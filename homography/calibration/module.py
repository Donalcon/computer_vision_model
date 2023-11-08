from functools import partial
from typing import Tuple
import torch
from tqdm.auto import tqdm
from kornia.geometry.conversions import convert_points_to_homogeneous
from homography.calibration.cam_modules import CameraParameterWLensDistDictZScore, SNProjectiveCamera
from homography.calibration.utils.linalg import distance_point_to_point


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

        # (batch_size, num_views_per_cam, keypoints, num_points)
        points_px_true = x["kp__ndc_projected"].to(self._device)
        batch_size, CV_k, T_k, _, N_k = points_px_true.shape
        points3d = self.pitch3d.keypoints
        points3d = torch.Tensor(points3d)

        points_px = convert_points_to_homogeneous(
            cam.project_point2ndc(points3d, lens_distortion=False)
        )

        if batch_size < cam.batch_dim:  # actual batch_size smaller than expected, i.e. last batch
            points_px = points_px[:batch_size]

        if self.lens_distortion_active:
            # undistort given points
            pc = pc.view(batch_size, T_k, S_k * N_k, 3)
            pc = pc.detach().clone()
            pc[..., :2] = cam.undistort_points(
                pc[..., :2], cam.intrinsics_ndc, num_iters=1
            )  # num_iters=1 might be enough for a good approximation
            pc = pc.view(batch_size, T_k, S_k, N_k, 3)

        points_px_true.squeeze(3)
        distances_px_keypoints_raw = distance_point_to_point(
            points_px_true, points_px
        )  # (batch_size, T_l, S_l, N_l)
        distances_px_keypoints_raw = distances_px_keypoints_raw.unsqueeze(-3)

        distances_dict = {
            "loss_ndc_keypoints": distances_px_keypoints_raw,  # (batch_size, T_l, 1, S_l, N_l)
        }
        return distances_dict, cam

    def self_optim_batch(self, x, *args, **kwargs):
        scheduler = self.Scheduler(self.optim)  # re-initialize lr scheduler for every batch
        self.optim.zero_grad()

        keypoint_mask = x["kp__is_keypoint_mask"].to(self._device)  # Updated mask
        num_actual_points = keypoint_mask.sum(dim=(-1, -2))  # Updated point calculation

        per_sample_loss = {}
        per_sample_loss["mask_keypoints"] = keypoint_mask  # Updated mask storage

        per_step_info = {"loss": [], "lr": []}

        with tqdm(range(self.optim_steps), **self.tqdm_kwqargs) as pbar:
            for step in pbar:
                self.optim.zero_grad()

                # forward pass
                distances_dict, cam = self(x)

                # distance calculate with masked input and output
                losses = {}
                distances = distances_dict.get("loss_ndc_keypoints", None)  # Simplified key name
                if distances is not None:
                    # Apply mask
                    distances[keypoint_mask == 0] = 0.0

                    # Log per-point distance
                    per_sample_loss["keypoint_distances_raw"] = distances

                    # sum pixel distance and normalize
                    distances_reduced = distances.sum(dim=(-1, -2))
                    distances_reduced = distances_reduced / num_actual_points

                    # Handle num_actual_points == 0
                    distances_reduced[num_actual_points == 0] = 0.0

                    loss = distances_reduced.mean(dim=-1)
                    loss = loss.sum()

                    losses["loss_ndc_keypoints"] = loss

                # Total loss
                loss_total = losses.get("loss_ndc_keypoints", 0)

                if self.log_per_step:
                    per_step_info["lr"].append(scheduler.get_last_lr())
                    per_step_info["loss"].append(distances_reduced)

                if step % 50 == 0:
                    pbar.set_postfix(
                        loss=f"{loss_total.detach().cpu().tolist():.5f}",
                    )

                loss_total.backward()
                self.optim.step()
                scheduler.step()

        per_sample_loss["loss_ndc_total"] = torch.sum(
            torch.stack([per_sample_loss["keypoint_distances_raw"]], dim=0),
            dim=0,
        )

        if self.log_per_step:
            per_step_info["loss"] = torch.stack(
                per_step_info["loss"], dim=-1
            )
            per_step_info["lr"] = torch.tensor(per_step_info["lr"])

        return per_sample_loss, cam, per_step_info
