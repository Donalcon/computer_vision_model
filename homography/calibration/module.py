from functools import partial
from typing import Tuple
import torch
from tqdm.auto import tqdm
from kornia.geometry.conversions import convert_points_to_homogeneous
from homography.calibration.cam_modules import CameraParameterWLensDistDictZScore, SNProjectiveCamera
from homography.calibration.utils.helper_functions import calculate_bbox_centre
from homography.calibration.utils.linalg import distance_point_pointcloud


class TVCalibModule(torch.nn.Module):
    def __init__(
        self,
        pitch3d,
        cam_distr,
        dist_distr,
        image_dim: Tuple[int, int],
        optim_steps: int,
        device="cpu",
        tqdm_kwqargs=None,
        log_per_step=False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.image_height, self.image_width = image_dim
        self.principal_point = (self.image_width / 2, self.image_height / 2)
        self.pitch3d = pitch3d
        self.cam_param_dict = CameraParameterWLensDistDictZScore(
            cam_distr, dist_distr, device=device
        )

        self.lens_distortion_active = False if dist_distr is None else True
        self.optim_steps = optim_steps
        self._device = device

        self.optim = torch.optim.AdamW(
            self.cam_param_dict.param_dict.parameters(), lr=0.1, weight_decay=0.01
        )
        self.Scheduler = partial(
            torch.optim.lr_scheduler.OneCycleLR,
            max_lr=0.05,
            total_steps=self.optim_steps,
            pct_start=0.5,
        )

        if self.lens_distortion_active:
            self.optim_lens_distortion = torch.optim.AdamW(
                self.cam_param_dict.param_dict_dist.parameters(), lr=1e-3, weight_decay=0.01
            )
            self.Scheduler_lens_distortion = partial(
                torch.optim.lr_scheduler.OneCycleLR,
                max_lr=1e-3,
                total_steps=self.optim_steps,
                pct_start=0.33,
                optimizer=self.optim_lens_distortion,
            )

        self.tqdm_kwqargs = tqdm_kwqargs
        if tqdm_kwqargs is None:
            self.tqdm_kwqargs = {}

        self.hparams = {"optim": str(self.optim), "scheduler": str(self.Scheduler)}
        self.log_per_step = log_per_step

    def forward(self, x):

        # individual camera parameters & distortion parameters
        phi_hat, psi_hat = self.cam_param_dict()

        cam = SNProjectiveCamera(
            phi_hat,
            psi_hat,
            self.principal_point,
            self.image_width,
            self.image_height,
            device=self._device,
            nan_check=False,
        )

        # (batch_size, num_views_per_cam, 3, num_segments, num_points)
        keypoints_px = x["bounding_boxes"].to(self._device)
        batch_size, T_k, _, S_k, N_k = keypoints_px.shape # may need to change variables this goes into

        # Calculate keypoint distances
        keypoints_px_true = calculate_bbox_centre(keypoints_px)

        points3d = self.pitch3d.keypoints  # (3, S_l, 2) to (S_l * 2, 3)
        points3d = points3d.reshape(3, S_k * 2).transpose(0, 1)
        points_px = convert_points_to_homogeneous(
            cam.project_point2ndc(points3d, lens_distortion=False)
        )

        if batch_size < cam.batch_dim:  # actual batch_size smaller than expected, i.e. last batch
            points_px = points_px[:batch_size]

        points_px = points_px.view(batch_size, T_k, S_k, 2, 3)
        pc = (
            keypoints_px_true.view(batch_size, T_k, 3, S_k * N_k)
            .transpose(2, 3)
            .view(batch_size, T_k, S_k, N_k, 3)
        )
        if self.lens_distortion_active:
            # undistort given points
            pc = pc.view(batch_size, T_k, S_k * N_k, 3)
            pc = pc.detach().clone()
            pc[..., :2] = cam.undistort_points(
                pc[..., :2], cam.intrinsics_ndc, num_iters=1
            )  # num_iters=1 might be enough for a good approximation
            pc = pc.view(batch_size, T_k, S_k, N_k, 3)
        distances_px_keypoints_raw = distance_point_pointcloud(
            keypoints_px_true, points_px
        )  # (batch_size, T_l, S_l, N_l)
        distances_px_keypoints_raw = distances_px_keypoints_raw.unsqueeze(-3)

        distances_dict = {
            "loss_ndc_lines": distances_px_keypoints_raw,  # (batch_size, T_l, 1, S_l, N_l)
        }
        return distances_dict, cam

    def self_optim_batch(self, x, *args, **kwargs):

        scheduler = self.Scheduler(self.optim)  # re-initialize lr scheduler for every batch
        if self.lens_distortion_active:
            scheduler_lens_distortion = self.Scheduler_lens_distortion()

        # TODO possibility to init from x; -> modify dataset that yields x
        self.cam_param_dict.initialize(None)
        self.optim.zero_grad()
        if self.lens_distortion_active:
            self.optim_lens_distortion.zero_grad()

        keypoint_masks = {
            "loss_ndc_keypoints": x["kp__is_keypoint_mask"].to(self._device),
        }
        num_actual_points = {
            "loss_ndc_keypoints": keypoint_masks["loss_ndc_keypoints"].sum(dim=(-1, -2)),
        }
        # print({f"{k} {v}" for k, v in num_actual_points.items()})

        per_sample_loss = {}
        per_sample_loss["mask_keypoints"] = keypoint_masks["loss_ndc_keypoints"]

        per_step_info = {"loss": [], "lr": []}
        # with torch.autograd.detect_anomaly():
        with tqdm(range(self.optim_steps), **self.tqdm_kwqargs) as pbar:
            for step in pbar:

                self.optim.zero_grad()
                if self.lens_distortion_active:
                    self.optim_lens_distortion.zero_grad()

                # forward pass
                distances_dict, cam = self(x)

                # create mask for batch dimension indicating whether distances and loss are computed
                # based on per-sample distance

                # distance calculate with masked input and output
                losses = {}
                for key_dist, distances in distances_dict.items():
                    # for padded points set distance to 0.0
                    # then sum over dimensions (S, N) and divide by number of actual given points
                    distances[~keypoint_masks[key_dist]] = 0.0

                    # log per-point distance
                    per_sample_loss[f"{key_dist}_distances_raw"] = distances

                    # sum px distance over S and number of points, then normalize given the number of annotations
                    distances_reduced = distances.sum(dim=(-1, -2))  # (B, T, 1, S, M) -> (B, T, 1)
                    distances_reduced = distances_reduced / num_actual_points[key_dist]

                    # num_actual_points == 0 -> set loss for this segment to 0.0 to prevent division by zero
                    distances_reduced[num_actual_points[key_dist] == 0] = 0.0

                    distances_reduced = distances_reduced.squeeze(-1)  # (B, T, 1) -> (B, T,)
                    per_sample_loss[key_dist] = distances_reduced

                    loss = distances_reduced.mean(dim=-1)  # mean over T dimension: (B, T, )-> (B,)
                    # only relevant for autograd:
                    # sum over batch dimension
                    # --> different batch sizes do not change the per sample loss and its gradients
                    loss = loss.sum()

                    losses[key_dist] = loss

                # each segment and annotation contributes equal to the loss -> no need for weighting segment types
                loss_total_dist = losses["loss_ndc_keypoints"]
                loss_total = loss_total_dist

                if self.log_per_step:
                    per_step_info["lr"].append(scheduler.get_last_lr())
                    per_step_info["loss"].append(distances_reduced)  # log per sample loss
                if step % 50 == 0:
                    pbar.set_postfix(
                        loss=f"{loss_total_dist.detach().cpu().tolist():.5f}",
                    )

                loss_total.backward()
                self.optim.step()
                scheduler.step()
                if self.lens_distortion_active:
                    self.optim_lens_distortion.step()
                    scheduler_lens_distortion.step()

        per_sample_loss["loss_ndc_total"] = torch.sum(
            torch.stack([per_sample_loss[key_dist] for key_dist in distances_dict.keys()], dim=0),
            dim=0,
        )

        if self.log_per_step:
            per_step_info["loss"] = torch.stack(
                per_step_info["loss"], dim=-1
            )  # (n_steps, batch_dim, temporal_dim)
            per_step_info["lr"] = torch.tensor(per_step_info["lr"])
        return per_sample_loss, cam, per_step_info