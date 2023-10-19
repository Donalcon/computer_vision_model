from pathlib import Path
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

class KeypointDetectionDataset(Dataset):
    def __init__(self,
                 file_match_info: Union[str, Path],
                 annotations: Union[str, Path],
                 model3d,
                 image_width: int,
                 image_height: int,
                 filter_cam_type=None,
                 return_image=True,
                 annotations_prefix="",
                 image_tfms=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
                 constant_cam_position=1,
                 remove_invalid=False,
                 ) -> None:
        self.df_match_info = None
        self.model3d = model3d
        self.image_width_source = image_width
        self.image_height_source = image_height
        self.pad_pixel_position_xy = 0.0
        self.return_image = return_image
        self.image_tfms = image_tfms

        self.annotations_prefix = annotations_prefix
        self.dir_annotations = Path(annotations)
        self.dir_images = file_match_info.parent
        self.df_match_info = pd.read_json(file_match_info).T
        self.df_match_info["image_id"] = self.df_match_info.index

        if filter_cam_type and "camera" in self.df_match_info.columns:
            self.df_match_info_filter = self.df_match_info.loc[
                self.df_match_info["camera"] == filter_cam_type
                ]
            if len(self.df_match_info_filter) == 0:
                print(self.df_match_info.head(5))
                print(
                    "requested cam type:",
                    filter_cam_type,
                    "given:",
                    self.df_match_info["camera"].unique().tolist(),
                )
                raise RuntimeError("No elements in dataset")
            self.df_match_info = self.df_match_info_filter
        if constant_cam_position > 1:
            self.df_match_info = (
                self.df_match_info.groupby(["league", "season", "match"]).agg(list).reset_index()
            )

            if not remove_invalid:
                if not (self.df_match_info["image_id"].agg(len) >= constant_cam_position).all():
                    print(self.df_match_info["image_id"].agg(len))
                    raise ValueError(
                        f"Tried to sample constant_cam_position={constant_cam_position} but this assumption does not hold for all samples"
                    )

            self.df_match_info["number_of_samples"] = self.df_match_info["image_id"].apply(
                lambda l: len(l)
            )
            self.df_match_info = self.df_match_info.loc[
                self.df_match_info["number_of_samples"] >= constant_cam_position
                ]

            self.df_match_info = self.df_match_info.apply(pd.Series.explode).reset_index()
            self.df_match_info = self.df_match_info.groupby(["league", "season", "match"]).sample(
                n=constant_cam_position, random_state=10
            )

            self.df_match_info = (
                self.df_match_info.groupby(["league", "season", "match"]).agg(list).reset_index()
            )

            self.df_match_info.drop(labels=["number_of_samples", "index"], inplace=True, axis=1)

        self.filter_cam_type = filter_cam_type
        self.constant_cam_position = constant_cam_position
        self.model3d = model3d

    def __len__(self):
        return len(self.df_match_info)

    def __getitem__(self, idx):
        candidates_meta = self.df_match_info.iloc[idx].to_dict()

        if self.constant_cam_position == 1:
            candidates_meta = {k: [v] for k, v in candidates_meta.items()}

        candidates_meta["keypoints_raw"] = []
        for image_id in candidates_meta["image_id"]:
            file_annotation = self.dir_annotations / image_id
            file_annotation = (
                    file_annotation.parent / f"{self.annotations_prefix}{file_annotation.stem}.json"
            )
            with open(file_annotation) as fr:
                keypoints_dict = json.load(fr)


            # Add empty entries for non-visible segments
            for l in self.model3d.keypoint_names:
                if l not in keypoints_dict:
                    keypoints_dict[l] = []
            candidates_meta["keypoints_raw"].append(keypoints_dict)

        meta_dict = {"meta": candidates_meta}

        return meta_dict
