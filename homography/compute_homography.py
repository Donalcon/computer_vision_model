import cv2
from norfair.camera_motion import HomographyTransformationGetter
from homography.homography_utils import get_dst_points, get_perspective_transform
import numpy as np

class FieldHomographyEstimator:
    def __init__(self, method=None, ransac_reproj_threshold=3, max_iters=2000, confidence=0.995):
        self.homography_getter = HomographyTransformationGetter(
            method=method,
            ransac_reproj_threshold=ransac_reproj_threshold,
            max_iters=max_iters,
            confidence=confidence
        )
        self.current_homography = None
        self.dst_points = get_dst_points()  # Initialize dst_points during object creation

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
        num_points = len(src_points)
        if num_points < 4:
            print("Bad homography: Insufficient points")
            return
        elif num_points == 4:
            print("Good homography: Minimum points")
        elif num_points > 4:
            print("Great homography: Additional points for robustness")
        # Also add in check here for co-linearity?
        # This will update the current homography transformation
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

        # Go through each player object and apply homography to its 'ground_center'
        for player in players:
            # Ensure 'ground_center' exists and is in the expected format (list or numpy array)
            if hasattr(player, 'xy') and (isinstance(player.xy, list) or isinstance(player.xy, np.ndarray)):
                transformed_xy = self.current_homography.rel_to_abs(np.array([player.xy]))
                player.txy = transformed_xy.tolist()[0]  # Assuming you want it as a list
                # Assign the txy to the detection.data dictionary
                player.detection.data["txy"] = player.txy
            else:
                print(f"Skipping player {player.id} due to missing or incorrectly formatted 'ground_center'.")
        return players

    # add in relevant xy properties to player class, abs and relative

    # ensure data type, array, list or dict is correct for both src and dst.
    # display x,y over players head for video debug.
    # ensure alpha is present in draw_mask

