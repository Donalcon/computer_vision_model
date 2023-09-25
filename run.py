from annotations import annotation
import numpy as np
import PIL
from norfair import Tracker, Video
from norfair.camera_motion import MotionEstimator
from norfair.distances import mean_euclidean
from config import Config
from inference import Converter, InertiaClassifier, NNClassifier
from inference.split_detector import SahiBallDetection, SahiPersonDetection
from run_utils import (
    get_main_ball,
    get_main_ref,
    update_motion_estimator,
)
from game import Match, Player, Team, MatchStats
from annotations.paths import AbsolutePath

config = Config.from_args()
video = Video(input_path=config.video_path)

# Object Detectors
ball_detector = SahiBallDetection()
ball_detector.load_model(model_path='seg-5epoch.pt', config_path='data.yaml')
person_detector = SahiPersonDetection()
person_detector.load_model(model_path='seg5ep-no-tile.pt', config_path='data.yaml')

# Color Classifier
nn_classifier = NNClassifier('model_path.pt', ['dublin', 'kerry', 'referee'])
classifier = InertiaClassifier(classifier=nn_classifier, inertia=20)
# Instantiate Match
home = Team(
    name=config.home['name'],
    color=config.home['color'],
    abbreviation=config.home['abbreviation'],
    text_color=config.home['text_color']
)
away = Team(
    name=config.away['name'],
    color=config.away['color'],
    abbreviation=config.away['abbreviation'],
    text_color=config.away['text_color']
)
teams = [home, away]
match = Match(home, away, fps=config.fps)
match.team_possession = home

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
    distance_threshold= 250,
    initialization_delay=3,
    hit_counter_max=2000,
)
motion_estimator = MotionEstimator()
coord_transformations = None

# Instantiate Ball Path
path = AbsolutePath()

possession_background = annotation.get_possession_background()
passes_background = annotation.get_passes_background()

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
        color=match.team_possession.color,
    )
    frame = annotation.draw_possession_counter(
        config.fps, match, frame, counter_background=possession_background, debug=False
    )
    if ball:
        frame = ball.draw(frame)
    if referee:
        frame = referee.draw(frame)
    frame = annotation.draw_passes_counter(
        match, frame, counter_background=passes_background, debug=False
    )
    # Write video
    frame = np.array(frame)
    video.write(frame)


# Generate summary stats
match_stats = MatchStats(match)
match_stats()
