import json
import csv
import numpy as np
import pickle
from argparse import ArgumentParser
from pathlib import Path
from time import time
from datetime import datetime
import torch
import matplotlib.pyplot as plt
import pandas as pd
from pytorch_lightning import seed_everything
from homography.calibration.cam_distribution.main_centre import get_cam_distr, get_dist_distr
from homography.calibration.data_loader import KeypointDetectionDataset, custom_list_collate
from homography.calibration.module import TVCalibModule
from homography.calibration.utils.objects_3d import pitch3D
import homography.calibration.utils.io as io
from homography.calibration.utils.visualization_mpl import (
    plot_loss_dataset,
    plot_per_stadium_loss,
    plot_per_step_loss,
    plot_per_step_lr,
)
seed_everything(seed=10, workers=True)

args = ArgumentParser()
args.add_argument("--hparams", type=Path)
args.add_argument("--log_per_step", action="store_true")
args.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
args.add_argument("--output_dir", type=Path, default="experiments")
args.add_argument("--exp_timestmap", action="store_true")
args = args.parse_args()

with open(args.hparams) as fw:
    hparams = json.load(fw)

distr_lens_disto = None
distr_cam = get_cam_distr(hparams["sigma_scale"], hparams["batch_dim"], hparams["temporal_dim"])
if hparams["lens_distortion"] == True:
    distr_lens_disto = get_dist_distr(hparams["batch_dim"], hparams["temporal_dim"])
hparams["distr_cam"] = distr_cam
hparams["distr_lens_disto"] = distr_lens_disto

output_dir = args.output_dir / args.hparams.stem
if args.exp_timestmap:
    output_dir = output_dir / datetime.now().strftime("%y%m%d-%H%M")
output_dir.mkdir(exist_ok=True, parents=True)
print("output directory", output_dir)
model3d = pitch3D()
# ANNOTATION_FILE = 'football-id-2-6/_annotations.coco.json'
# IMG_DIR = 'football-id-2-6/train'
print("Init Dataset")
dataset = KeypointDetectionDataset(
    model3d=model3d,
    image_width=hparams["image_width"],
    image_height=hparams["image_height"],
    **hparams["dataset"],
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=hparams["batch_dim"],
    num_workers=0,
    shuffle=False,
    collate_fn=custom_list_collate,
)

print("Init TVCalibModule")
model = TVCalibModule(
    model3d,
    distr_cam,
    distr_lens_disto,
    (hparams["image_height"], hparams["image_width"]),
    hparams["optim_steps"],
    device="cpu",
    log_per_step=args.log_per_step,
    tqdm_kwqargs={"ncols": 100},
)
# TODO continue from here, SNprojectiveCamera is not defined correctly also
hparams["TVCalibModule"] = model.hparams
print(output_dir / "hparams.yml")
io.write_yaml(hparams, output_dir / "hparams.yml")

dataset_dict_stacked = {}
dataset_dict_stacked["batch_idx"] = []
for batch_idx, x_dict in enumerate(dataloader):

    print(f"{batch_idx}/{len(dataloader) - 1}")
    points = x_dict["kp__px_projected"].clone().detach()
    batch_size = points.shape[0]

    fout_prefix = f"batch_{batch_idx}"

    start = time()
    per_sample_loss, cam, per_step_info = model.self_optim_batch(x_dict)
    with open('cam_object.pkl', 'wb') as f:
        pickle.dump(cam, f)


    output_dict = {"image_ids": x_dict["image_id"],
                   "camera": cam.get_parameters(batch_size),
                   **per_sample_loss,
                   "meta": x_dict["meta"],
                   "batch_idx": batch_idx,
                   }
    print(output_dict.keys())
    torch.save(io.detach_dict(output_dict), output_dir / f"{fout_prefix}.pt")


    # format for per_sample_output.json
    if "per_step_lr" in output_dict:
        del output_dict["per_step_lr"]
    # max distance over all given points
    output_dict["loss_ndc_total_max"] = (
        output_dict["loss_ndc_total"].amax(dim=[-2, -1])
    )
    output_dict["loss_ndc_total_max"] = output_dict["loss_ndc_total_max"].max(dim=-1)[0]
    del output_dict["loss_ndc_total"]
    del output_dict["mask_keypoints"]
    del output_dict["keypoint_distances_raw"]

    output_dict = io.tensor2list(output_dict)
    output_dict["batch_idx"] = [[str(batch_idx)]] * batch_size

    # output_dict["time_s"] /= batch_size

    for k in output_dict.keys():
        if k not in dataset_dict_stacked:
            dataset_dict_stacked[k] = output_dict[k]
        elif isinstance(dataset_dict_stacked[k], list):
            dataset_dict_stacked[k].extend(output_dict[k])
        else:
            dataset_dict_stacked[k] = output_dict[k]

    print(output_dir / f"{fout_prefix}.pt")

print(dataset_dict_stacked.keys())

csv_file = "homographies.csv"
with open(csv_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write the header
    writer.writerow(["frame_name", "homography_matrix"])

    # Iterate through each entry in dataset_dict_stacked to write to the CSV file
    for meta, homography in zip(dataset_dict_stacked["meta"], dataset_dict_stacked["homography"]):
        # Extract file name and homography matrix, assuming meta is a dictionary with a 'file_name' key
        # and homography is in its raw form
        file_name = meta['file_name']
        homography_str = str(homography)  # This will convert the list or numpy array to a string

        # Write the row to the CSV
        writer.writerow([file_name, homography_str])

print(f"CSV file '{csv_file}' written successfully.")
for key, value in dataset_dict_stacked.items():
    print(f"{key}: {type(value)}, Shape: {np.array(value).shape}")
df = pd.DataFrame.from_dict(dataset_dict_stacked)

# Extract individual x, y, z components from each item in 'position_meters'
df["pos_x"] = [pos[2][0] for pos in df["position_meters"]]
df["pos_y"] = [pos[2][1] for pos in df["position_meters"]]
df["pos_z"] = [pos[2][2] for pos in df["position_meters"]]
df["pincipal_point_x"] = [pos[2][0] for pos in df["principal_point"]]
df["pincipal_point_y"] = [pos[2][1] for pos in df["principal_point"]]

if 'meta' in df.columns:
    df.drop(columns=["meta"], inplace=True)
if 'camera'in df.columns:
    df.drop(columns=["camera"], inplace=True)
if 'position_meters'in df.columns:
    df.drop(columns=["position_meters"], inplace=True)
if 'principal_point' in df.columns:
    df.drop(columns=["principal_point"], inplace=True)
# Ensure that 'homography' is actually in df.columns before attempting operations on it
if 'homography' in df.columns:
    # Flatten each 3x3 homography matrix into a list of 9 elements
    df['homography_flattened'] = df['homography'].apply(lambda x: x.flatten().tolist() if isinstance(x, np.ndarray) else x)
    # Remove the original 'homography' column from the DataFrame
    df.drop(columns=["homography"], inplace=True)

explode_cols = [k for k, v in dataset_dict_stacked.items() if isinstance(v, list)]
df = df.explode(column=explode_cols)  # explode over t
df["image_id"] = df["image_ids"].apply(lambda l: l.split(".jpg")[0])
df.set_index("image_id", inplace=True)

fout = output_dir / "per_sample_output.json"
df.to_json(fout, orient="records", lines=True)

if "match" in df.columns:
    plot_per_stadium_loss(df, output_dir)

plot_loss_dataset(df, output_dir)
