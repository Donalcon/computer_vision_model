import argparse
import supervision as sv
import cv2
import numpy as np
import PIL
from PIL import Image
from norfair import Tracker, Video
from norfair.camera_motion import MotionEstimator
from norfair.distances import mean_euclidean
from inference.nn_classifier import NNClassifier
from inference.ball_detector import BallDetection
from inference.sahi_ball_detector import SahiBallDetection
from inference.sahi_person_detector import SahiPersonDetection
from inference.sahi import SahiDetector
from inference import Converter, HSVClassifier, InertiaClassifier
from inference.filters import filters
from run_utils import (
    get_ball_detections,
    get_main_ball,
    get_main_ref,
    get_person_detections,
    update_motion_estimator,
    get_sahi_ball_detections,
    get_sahi_person_detections,
    get_sahi_ref_detections,
)
from game import Match, Player, Team
from game.draw import AbsolutePath
from game.pass_event import Pass

parser = argparse.ArgumentParser()
parser.add_argument(
    "--video",
    default="videos/soccer_possession.mp4",
    type=str,
    help="Path to the input video",
)
parser.add_argument(
    "--model", default="models/ball.pt", type=str, help="Path to the model"
)
parser.add_argument(
    "--passes",
    action="store_true",
    help="Enable pass detection",
)
parser.add_argument(
    "--possession",
    action="store_true",
    help="Enable possession counter",
)
args = parser.parse_args()

video = Video(input_path=args.video)
fps = video.video_capture.get(cv2.CAP_PROP_FPS)
print('fps:', fps)
# Object Detectors
person_detector = SahiPersonDetection()
ball_detector = SahiBallDetection()
referee_detector = SahiBallDetection()

# NN Classifier
nn_classifier = NNClassifier('model_path.pt', ['dublin', 'kerry', 'referee'])

# HSV Classifier
hsv_classifier = HSVClassifier(filters=filters)
# Add inertia to classifier
classifier = InertiaClassifier(classifier=nn_classifier, inertia=20)

# Match
kerry = Team(
    name="kerry",
    abbreviation="KER",
    color=(21, 107, 21),
    text_color=(109, 230, 240),
)
dublin = Team(
    name="dublin",
    abbreviation="DUB",
    color=(245, 206, 11),
    text_color=(128, 0, 0)
)
teams = [dublin, kerry]
match = Match(home=dublin, away=kerry, fps=fps)
match.team_possession = dublin

# Tracking
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
    distance_threshold=150,
    initialization_delay=3,
    hit_counter_max=2000,
)
motion_estimator = MotionEstimator()
coord_transformations = None

# Paths
path = AbsolutePath()

# Get Counter img
possession_background = match.get_possession_background()
passes_background = match.get_passes_background()

for i, frame in enumerate(video):

    # Get Detections
    player_detections = get_sahi_person_detections(person_detector, frame)
    ball_detections = get_sahi_ball_detections(ball_detector, frame)
    referee_detections = get_sahi_ref_detections(referee_detector, frame)
    detections = player_detections + ball_detections

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
        detections=referee_detections, coord_transformations=coord_transformations
    )

    player_detections = Converter.TrackedObjects_to_Detections(player_track_objects)
    ball_detections = Converter.TrackedObjects_to_Detections(ball_track_objects)
    referee_detections = Converter.TrackedObjects_to_Detections(ref_track_objects)

    player_detections = classifier.predict_from_detections(
        detections=player_detections,
        img=frame,
    )

    # Match update
    ball = get_main_ball(ball_detections)
    referee = get_main_ref(referee_detections)
    players = Player.from_detections(detections=player_detections, teams=teams)
    match.update(players, ball)
    frame = PIL.Image.fromarray(frame)

    if args.possession:
        frame = Player.draw_players(
            players=players, frame=frame, confidence=False, id=True
        )

        frame = path.draw(
            img=frame,
            detection=ball.detection,
            coord_transformations=coord_transformations,
            color=match.team_possession.color,
        )

        frame = match.draw_possession_counter(
            frame, counter_background=possession_background, debug=False
        )

        if ball:
            frame = ball.draw(frame)

    if args.passes:
        pass_list = match.passes

        frame = Pass.draw_pass_list(
            img=frame, passes=pass_list, coord_transformations=coord_transformations
        )

        frame = match.draw_passes_counter(
            frame, counter_background=passes_background, debug=False
        )

    frame = np.array(frame)
    print(frame.shape)

    # Write video
    video.write(frame)


# Generate summary stats
home_turnovers = match.home.get_turnovers()
away_turnovers = match.away.get_turnovers()
home_time_in_possession = match.home.get_time_possession(match.fps)
away_time_in_possession = match.away.get_time_possession(match.fps)

print(f"{match.home.name} turnovers:", home_turnovers)
print(f"{match.away.name} turnovers:", away_turnovers)
print(f"{match.home.name} time in possession: {home_time_in_possession}")
print(f"{match.away.name} time in possession: {away_time_in_possession}")

