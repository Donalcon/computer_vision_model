import numpy as np
from norfair.camera_motion import HomographyTransformationGetter

from homography.homography_utils import get_dst_points, check_num_points


class FieldHomographyEstimator:
    def __init__(self, method=None, ransac_reproj_threshold=3, max_iters=2000, confidence=0.995):
        self.homography_getter = HomographyTransformationGetter(
            method=method,
            ransac_reproj_threshold=ransac_reproj_threshold,
            max_iters=max_iters,
            confidence=confidence
        )
        self.current_homography = None
        self.dst_points = get_dst_points()

    def update_with_detections(self, detections):
        src_points = {}
        for detection in detections:
            name = detection.data["name"]
            xy = detection.data["xy"].tolist()  # Convert numpy array to list
            src_points[name] = xy

        # Convert dictionary values to lists
        matched_src_points = [src_points[key] for key in src_points.keys() if key in self.dst_points]
        matched_dst_points = [self.dst_points[key] for key in src_points.keys() if key in self.dst_points]

        # Convert lists to numpy arrays
        matched_src_points = np.array(matched_src_points)
        matched_dst_points = np.array(matched_dst_points)

        # Now call update

        self.update(matched_src_points, matched_dst_points)

    def update(self, src_points, dst_points):
        point_type = check_num_points(src_points, dst_points)
        if point_type == "insufficient" or "collinear":
            return
        try:
            update_prvs, homography = self.homography_getter(src_points, dst_points)
            if update_prvs:
                self.current_homography = homography

            self.current_homography = homography
        except np.linalg.LinAlgError as e:
            # Handle the error here
            print(f"Error encountered: {e}. Skipping this frame's homography update.")
            pass

    def apply_to_player(self, players):
        # Make sure that the current_homography exists
        if self.current_homography is None:
            print("No homography matrix has been calculated yet.")
            return None

        # Initialize a list to store all transformed x,y coordinates
        transformed_coords = []

        # Go through each player object and apply homography to its 'ground_center'
        for player in players:
            if hasattr(player, 'xy') and (isinstance(player.xy, list) or isinstance(player.xy, np.ndarray)):
                transformed_xy = self.current_homography.rel_to_abs(np.array([player.xy]))
                player.txy = transformed_xy.tolist()[0]

                # Append the transformed coordinates to the list
                transformed_coords.append(player.txy)

                player.detection.data["txy"] = player.txy
            else:
                print(f"Skipping player {player.id} due to missing or incorrectly formatted 'ground_center'.")

        for player in players:
            print(player.txy)

        return players

    def compute_homography(self, keypoints, dst_points_mapping):
        src_points = []
        dst_points_list = []

        # Map keypoints to the expected src and dst points for homography calculation
        for keypoint in keypoints:
            category_id = keypoint['category_id']
            xy = keypoint['keypoint']
            if category_id in dst_points_mapping:
                src_points.append(xy)
                dst_points_list.append(dst_points_mapping[category_id])

        # Convert the lists to numpy arrays
        src_points_np = np.array(src_points)
        dst_points_np = np.array(dst_points_list)

        # Check if there are enough points to compute the homography
        if len(src_points_np) >= 4:
            try:
                # Compute the homography matrix
                success_flag, homography_matrix = self.homography_getter(src_points_np, dst_points_np)
                if success_flag:
                    return homography_matrix
            except np.linalg.LinAlgError as e:
                print(f"Error encountered: {e}. Skipping this frame's homography update.")
        else:
            homography_matrix = None
            print("Not enough keypoints to compute homography.")
        return None

