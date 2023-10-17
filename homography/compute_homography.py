from norfair.camera_motion import HomographyTransformationGetter
import numpy as np
from norfair.camera_motion import HomographyTransformationGetter

from homography.homography_utils import get_dst_points, collinear, verify_distance_between_players, \
    verify_players_within_pitch, check_num_points


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
        if point_type == "insufficient":
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

            # Verification Clause 1
            if not verify_distance_between_players(transformed_coords):
                return None  # Or you could reset the current homography
            if not verify_players_within_pitch(transformed_coords):
                return None

        for player in players:
            print(player.txy)

        return players

    # add in relevant xy properties to player class, abs and relative

    # ensure data type, array, list or dict is correct for both src and dst.
    # display x,y over players head for video debug.
    # ensure alpha is present in draw_mask
