from typing import Tuple, Annotated, Any
from zenml import step
from norfair import Tracker
from norfair.camera_motion import MotionEstimator
from norfair.distances import mean_euclidean


@step
def trackers() -> Tuple[
    Annotated[Tracker, "player_tracker"],
    Annotated[Tracker, "ref_tracker"],
    Annotated[Tracker, "ball_tracker"],
    Annotated[MotionEstimator, "motion_estimator"],
    Annotated[Any, "coordinate_transformation"]
]:
    player_tracker = Tracker(
        distance_function=mean_euclidean,
        distance_threshold=250,
        initialization_delay=3,
        hit_counter_max=90,
    )

    referee_tracker = Tracker(
        distance_function=mean_euclidean,
        distance_threshold=250,
        initialization_delay=3,
        hit_counter_max=90,
    )

    ball_tracker = Tracker(
        distance_function=mean_euclidean,
        distance_threshold=250,
        initialization_delay=3,
        hit_counter_max=2000,
    )

    motion_estimator = MotionEstimator()

    coord_transformations = None

    return player_tracker, referee_tracker, ball_tracker, motion_estimator, coord_transformations
