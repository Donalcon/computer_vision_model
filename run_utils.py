from typing import List, Tuple
import norfair
import numpy as np
from norfair import Detection
from norfair.camera_motion import MotionEstimator
from inference import Converter
from game import Referee, Ball, Match
from inference.sahi_detector import BaseSahiDetection


def create_mask(frame: np.ndarray, detections: List[norfair.Detection]) -> np.ndarray:
    """

    Creates mask in order to hide detections and goal counter for motion estimation

    Parameters
    ----------
    frame : np.ndarray
        Frame to create mask for.
    detections : List[norfair.Detection]
        Detections to hide.

    Returns
    -------
    np.ndarray
        Mask.
    """

    if not detections:
        mask = np.ones(frame.shape[:2], dtype=frame.dtype)
    else:
        detections_df = Converter.Detections_to_DataFrame(detections)
        mask = BaseSahiDetection.generate_predictions_mask(predictions=detections_df, img=frame, margin=40)

    # remove goal counter
    mask[69:200, 160:510] = 0

    return mask


def apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Applies a mask to an img

    Parameters
    ----------
    img : np.ndarray
        Image to apply the mask to
    mask : np.ndarray
        Mask to apply

    Returns
    -------
    np.ndarray
        img with mask applied
    """
    masked_img = img.copy()
    masked_img[mask == 0] = 0
    return masked_img


def update_motion_estimator(
    motion_estimator: MotionEstimator,
    detections: List[Detection],
    frame: np.ndarray,
) -> "CoordinatesTransformation":
    """

    Update coordinate transformations every frame

    Parameters
    ----------
    motion_estimator : MotionEstimator
        Norfair motion estimator class
    detections : List[Detection]
        List of detections to hide in the mask
    frame : np.ndarray
        Current frame

    Returns
    -------
    CoordinatesTransformation
        Coordinate transformation for the current frame
    """

    mask = create_mask(frame=frame, detections=detections)
    coord_transformations = motion_estimator.update(frame, mask=mask) # should have mask=mask in args
    return coord_transformations



def get_main_ref(detections: List[Detection]) -> Referee:
    """
    Gets the main referee from a list of referee detection

    Parameters
    ----------
    detections : List[Detection]
        List of detections
    Returns
    -------
    Referee
        Main referee
    """
    referee = Referee(detection=None)

    if detections:
        referee.detection = detections[0]

    return referee

def get_main_ball(detections: List[Detection], match: Match = None) -> Ball:
    """
    Gets the main ball from a list of balls detection

    The match is used in order to set the color of the ball to
    the color of the team in possession of the ball.

    Parameters
    ----------
    detections : List[Detection]
        List of detections
    match : Match, optional
        Match object, by default None

    Returns
    -------
    Ball
        Main ball
    """
    ball = Ball(detection=None)

    if match:
        ball.set_color(match)

    if detections:
        ball.detection = detections[0]

    return ball
