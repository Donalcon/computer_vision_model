from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
import re
from typing import Union
import numpy as np
import pandas as pd
import json
import random
import torch
import kornia
from PIL import Image
from torch.utils.data import Dataset
import torchvision
from torch._six import string_classes
import collections
from operator import itemgetter
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import os

from homography.calibration.utils.mapping import class_to_model3d_mapping


class BaseDataset(Dataset):
    def __init__(
            self,
            annotation_file,
            image_dir,
            model3d,
    ):
        super().__init__()
        self.coco = COCO(annotation_file)
        self.image_dir = Path(image_dir)
        self.image_ids = self.coco.getImgIds()
        self.model3d = model3d
        self.id_to_class_name = {category['id']: category['name'] for category in
                                 self.coco.loadCats(self.coco.getCatIds())}

        # Create DataFrame
        data_list = []
        for img_id in self.image_ids:
            img_info = self.coco.loadImgs(img_id)[0]  # Assuming each ID returns a list with a single dictionary
            ann_ids = self.coco.getAnnIds(imgIds=img_info['id'])
            annotations = self.coco.loadAnns(ann_ids)
            # Populate more fields if needed
            data_list.append({
                'image_id': img_id,
                'file_name': img_info['file_name'],
                'annotations': annotations,
            })
        self.df_match_info = pd.DataFrame(data_list).set_index('image_id')
        self.df_match_info["image_id"] = self.df_match_info.index

    def __len__(self):
        return len(self.df_match_info)

    def __getitem__(self, idx):
        candidates_meta = self.df_match_info.iloc[idx].to_dict()
        candidates_meta["image_id"] = [self.df_match_info.index[idx]]
        # List of class names you want to keep
        allowed_classes = ['0A', '0B', '1A', '1B', '1C', '1D', '1E', '1F', '1G', '1H', '1I', '1J', '1K', '1L', '1M',
                           '1N', '1O', '1P', '1Q', '1R', '1S', '1T', '2A', '2B', '2C', '2D', '2E', '2F', '2G', '2H',
                           '2I', '2J', '2K', '2L', '2M', '2N', '2O', '2P', '2Q', '2R', '2S', '2T', '1GPA', '1GPB',
                           '2GPA', '2GPB']

        # Initialize the list that will be populated and added to candidates_meta
        candidates_meta["keypoints_raw"] = []
        # Populate the keypoints_dict based on the annotations
        keypoints_dict = {}
        for i in [candidates_meta["image_id"]]:

            for annotation in candidates_meta["annotations"]:
                original_class_name = self.id_to_class_name.get(annotation.get('category_id', None), "Unknown")
                if original_class_name in allowed_classes:  # Check before mapping
                    class_name = class_to_model3d_mapping.get(original_class_name, None)
                    if class_name not in keypoints_dict:
                        keypoints_dict[class_name] = []

                    bbox = annotation['bbox']
                    x1, y1, w, h = bbox
                    x2, y2 = x1 + w, y1 + h

                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2

                    keypoints_dict[class_name].append([x_center, y_center])
            # Add empty entries for non-visible segments
            for l in self.model3d.keypoint_names:
                if l not in keypoints_dict:
                    keypoints_dict[l] = []
        # Add the populated keypoints_dict to candidates_meta["keypoints_raw"]
        candidates_meta["keypoints_raw"].append(keypoints_dict)
        return {"meta": candidates_meta}


class KeypointDetectionDataset(BaseDataset):
    def __init__(
            self,
            annotation_file,
            image_dir,
            model3d,
            image_width,
            image_height,
            return_image=True,
            image_tfms=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
    ) -> None:
        super().__init__(
            model3d=model3d,
            annotation_file=annotation_file,
            image_dir=image_dir,
        )
        self.pad_pixel_position_xy = 0.0
        self.image_width_source = image_width
        self.image_height_source = image_height
        self.return_image = return_image
        self.image_tfms = image_tfms

    def __getitem__(self, idx):
        meta_dict = super().__getitem__(idx)

        image_ids = meta_dict["meta"]["image_id"]
        keypoints_raw = meta_dict["meta"]["keypoints_raw"]
        # Prepare each sample using your custom function
        per_sample_output = [
            self.prepare_per_sample(keypoints_raw[i], image_ids[i]) for i in range(len(image_ids))
        ]
        for k in per_sample_output[0].keys():
            meta_dict[k] = torch.stack([per_sample_output[i][k] for i in range(len(image_ids))])
        del meta_dict["meta"]["keypoints_raw"]

        meta_dict["image_id"] = image_ids
        del meta_dict["meta"]["image_id"]

        return meta_dict

    def prepare_per_sample(self, keypoints_raw: dict, image_id: str):
        r = {}
        pixel_stacked_dictionary = {}
        # Initial pre-processing to gather and scale keypoints
        for label, points in keypoints_raw.items():
            points_sel = points  # all points are selected
            if len(points_sel) > 0:
                xx = torch.tensor([a[0] for a in points_sel])
                yy = torch.tensor([a[1] for a in points_sel])
                temp = torch.stack([xx, yy], dim=-1)

                # Scale keypoints
                temp[:, 0] *= (self.image_width_source - 1)
                temp[:, 1] *= (self.image_height_source - 1)

                pixel_stacked_dictionary[label] = temp

        # Add padding logic
        num_keypoints = len(self.model3d.keypoints)
        num_points_per_keypoint = 2  # Assuming each keypoint has an x and y point
        px_projected_selection = torch.zeros((num_keypoints, num_points_per_keypoint, 2)) + self.pad_pixel_position_xy

        # Populate the px_projected_selection tensor
        for keypoint_index, keypoint_name in enumerate(self.model3d.keypoint_names):
            if keypoint_name in pixel_stacked_dictionary:
                num_points_available = pixel_stacked_dictionary[keypoint_name].shape[0]
                px_projected_selection[keypoint_index, :num_points_available, :] = \
                    pixel_stacked_dictionary[keypoint_name][:num_points_available]

        # Mask logic
        is_keypoint_mask = (
                                   (0.0 <= px_projected_selection[:, 0]) & (
                                   px_projected_selection[:, 0] < self.image_width_source)
                           ) & (
                                   (0 <= px_projected_selection[:, 1]) & (
                                   px_projected_selection[:, 1] < self.image_height_source)
                           )
        r["kp__is_keypoint_mask"] = is_keypoint_mask.unsqueeze(0)  # (1, num_keypoints)

        # Homogeneous conversion
        px_projected_homogeneous = kornia.geometry.conversions.convert_points_to_homogeneous(px_projected_selection)

        # This is the equivalent variable to store transformed keypoints
        r["kp__px_projected"] = px_projected_homogeneous  # (num_keypoints, 3)

        # Normalized Device Coordinates (NDC) conversion
        ndc_projected = px_projected_homogeneous.clone()
        ndc_projected[:, 0] /= self.image_width_source
        ndc_projected[:, 1] /= self.image_height_source
        ndc_projected[:, 1] = ndc_projected[:, 1] * 2.0 - 1
        ndc_projected[:, 0] = ndc_projected[:, 0] * 2.0 - 1

        r["kp__ndc_projected"] = ndc_projected  # (num_keypoints, 3)
        return r


def custom_list_collate(batch):
    r"""
    Function that takes in a batch of data and puts the elements within the batch
    into a tensor with an additional outer dimension - batch size. The exact output type can be
    a :class:`torch.Tensor`, a `Sequence` of :class:`torch.Tensor`, a
    Collection of :class:`torch.Tensor`, or left unchanged, depending on the input type.
    This is used as the default function for collation when
    `batch_size` or `batch_sampler` is defined in :class:`~torch.utils.data.DataLoader`.
    Here is the general input type (based on the type of the element within the batch) to output type mapping:
    * :class:`torch.Tensor` -> :class:`torch.Tensor` (with an added outer dimension batch size)
    * NumPy Arrays -> :class:`torch.Tensor`
    * `float` -> :class:`torch.Tensor`
    * `int` -> :class:`torch.Tensor`
    * `str` -> `str` (unchanged)
    * `bytes` -> `bytes` (unchanged)
    * `Mapping[K, V_i]` -> `Mapping[K, default_collate([V_1, V_2, ...])]`
    * `NamedTuple[V1_i, V2_i, ...]` -> `NamedTuple[default_collate([V1_1, V1_2, ...]), default_collate([V2_1, V2_2, ...]), ...]`
    * `Sequence[V1_i, V2_i, ...]` -> `Sequence[default_collate([V1_1, V1_2, ...]), default_collate([V2_1, V2_2, ...]), ...]`
    Args:
        batch: a single batch to be collated
    Examples:
        >>> # Example with a batch of `int`s:
        >>> default_collate([0, 1, 2, 3])
        tensor([0, 1, 2, 3])
        >>> # Example with a batch of `str`s:
        >>> default_collate(['a', 'b', 'c'])
        ['a', 'b', 'c']
        >>> # Example with `Map` inside the batch:
        >>> default_collate([{'A': 0, 'B': 1}, {'A': 100, 'B': 100}])
        {'A': tensor([  0, 100]), 'B': tensor([  1, 100])}
        >>> # Example with `NamedTuple` inside the batch:
        >>> Point = namedtuple('Point', ['x', 'y'])
        >>> default_collate([Point(0, 0), Point(1, 1)])
        Point(x=tensor([0, 1]), y=tensor([0, 1]))
        >>> # Example with `Tuple` inside the batch:
        >>> default_collate([(0, 1), (2, 3)])
        [tensor([0, 2]), tensor([1, 3])]

        >>> # modification
        >>> # Example with `List` inside the batch:
        >>> default_collate([[0, 1, 2], [2, 3, 4]])
        >>> [[0, 1, 2], [2, 3, 4]]
        >>> # original behavior
        >>> [[0, 2], [1, 3], [2, 4]]
    """

    np_str_obj_array_pattern = re.compile(r"[SaUO]")
    default_collate_err_msg_format = "default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found {}"

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):

        # Find out the maximum shape
        max_shape = [0] * len(elem.size())
        for tensor in batch:
            for i, size_i in enumerate(tensor.size()):
                max_shape[i] = max(max_shape[i], size_i)

        # Pad the tensors to have the same shape
        padded_batch = []
        for tensor in batch:
            pad_amount = [(max_shape[i] - tensor.size(i)) for i in reversed(range(len(max_shape)))]
            pad_amount = [item for sublist in zip([0] * len(pad_amount), pad_amount) for item in sublist]
            padded_tensor = pad(tensor, pad_amount)
            padded_batch.append(padded_tensor)

        # Handle the shared memory if in a background process
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in padded_batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage).resize_(len(padded_batch), *list(max_shape))

        return torch.stack(padded_batch, 0, out=out)
    elif (
            elem_type.__module__ == "numpy"
            and elem_type.__name__ != "str_"
            and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return [torch.as_tensor(b) for b in batch]
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.Mapping):
        try:
            return elem_type({key: custom_list_collate([d[key] for d in batch]) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: custom_list_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(custom_list_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.Sequence):
        # Calculate the maximum length of sequences in the batch
        max_length = max(len(e) for e in batch)

        # Pad each sequence to have the same length as the longest sequence in the batch
        padded_batch = [e + [0] * (max_length - len(e)) for e in batch]
        return padded_batch

        # if isinstance(elem, tuple):
        #     return [
        #         custom_list_collate(samples) for samples in transposed
        #     ]  # Backwards compatibility.
        # else:
        #     try:
        #         return elem_type([custom_list_collate(samples) for samples in transposed])
        #     except TypeError:
        #         # The sequence type may not support `__init__(iterable)` (e.g., `range`).
        #         return [custom_list_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))
