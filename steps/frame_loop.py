from typing import List
import PIL
import numpy as np
from norfair.camera_motion import MotionEstimator
from zenml import step
from norfair import Tracker, Video
from run_utils import update_motion_estimator, get_main_ball, get_main_ref
from steps.annotations.annotation import draw_possession_counter, draw_passes_counter
from steps.annotations.paths import AbsolutePath
from steps.game import Team, Match, Player
from steps.inference import Converter
from steps.inference.sahi_detector import SahiBallDetection, SahiPersonDetection
from steps.inference.inertia_classifier import InertiaClassifier
# To Do: Once restructure is complete, reformat init files so imports above are cleaner




@step
def frame_loop(
    video: Video,
    ball_detector: "SahiBallDetection",
    person_detector: "SahiPersonDetection",
    player_tracker: "Tracker",
    ball_tracker: "Tracker",
    referee_tracker: "Tracker",
    motion_estimator: "MotionEstimator",
    classifier: "InertiaClassifier",
    teams: List["Team"],
    match: "Match",
    path: "AbsolutePath",
    fps: int,  # Replace Any with the specific type of config object
    possession_background,
    passes_background,
) -> None:
    for i, frame in enumerate(video):
        # Get Detections
        ball_predictions = ball_detector.predict(frame)
        person_predictions = person_detector.predict(frame)
        ball_detections = ball_detector.get_ball_detections(ball_predictions)
        player_detections, ref_detections = person_detector.get_detections(person_predictions)
        detections = ball_detections + player_detections + ref_detections

        # Update trackers
        coord_transformations = update_motion_estimator(
            motion_estimator=motion_estimator,
            detections=detections,
            frame=frame,
        )
        player_track_objects = player_tracker.update(
            detections=player_detections, coord_transformations=coord_transformations
        )
        ball_track_objects = ball_tracker.update(
            detections=ball_detections, coord_transformations=coord_transformations
        )
        ref_track_objects = referee_tracker.update(
            detections=ref_detections, coord_transformations=coord_transformations
        )

        # Integrate Detections & Tracks
        player_detections = Converter.TrackedObjects_to_Detections(player_track_objects)
        ball_detections = Converter.TrackedObjects_to_Detections(ball_track_objects)
        ref_detections = Converter.TrackedObjects_to_Detections(ref_track_objects)
        player_detections = classifier.predict_from_detections(
            detections=player_detections,
            img=frame,
        )

        # Match update
        ball = get_main_ball(ball_detections)
        referee = get_main_ref(ref_detections)
        players = Player.from_detections(detections=player_detections, teams=teams)
        match.update(players, ball)
        frame = PIL.Image.fromarray(frame)

        # Annotations & counters
        frame = Player.draw_players(
            players=players, frame=frame, confidence=False, id=True
        )
        frame = path.draw(
            img=frame,
            detection=ball.detection,
            coord_transformations=coord_transformations,
            color=match.possession.team_possession.color,
        )
        frame = draw_possession_counter(
            fps, match, frame, counter_background=possession_background, debug=False
        )
        if ball:
            frame = ball.draw(frame)
        if referee:
            frame = referee.draw(frame)
        frame = draw_passes_counter(
            match, frame, counter_background=passes_background, debug=False
        )
        # Write video
        frame = np.array(frame)
        video.write(frame)
