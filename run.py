import os
import mlflow
from annotations import annotation
import numpy as np
import PIL
from norfair import Tracker, Video
from norfair.camera_motion import MotionEstimator
from norfair.distances import mean_euclidean
from config import Config
from homography.compute_homography import FieldHomographyEstimator
from homography.homography_utils import log_fail_counts_to_mlflow
from inference import Converter, InertiaClassifier, NNClassifier
from inference.detector import SahiBallDetection, Yolov8Detection
from run_utils import (
    get_main_ball,
    get_main_ref,
    update_motion_estimator
)
from game import Match, Player, Team, MatchStats
from annotations.paths import AbsolutePath

config = Config.from_args()
video = Video(input_path=config.video_path)

# Object Detectors
ball_detector = SahiBallDetection()
std_detector = Yolov8Detection()

# Color Classifier
nn_classifier = NNClassifier('models/model_path2.pt', ['dublin', 'kerry', 'referee'])
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
    distance_threshold=250,
    initialization_delay=3,
    hit_counter_max=2000,
)
keypoint_tracker = Tracker(
    distance_function=mean_euclidean,
    distance_threshold=50,
    initialization_delay=1,
    hit_counter_max=1000,
)
motion_estimator = MotionEstimator()
coord_transformations = None

# Instantiate Ball Path
path = AbsolutePath()

possession_background = annotation.get_possession_background()
#passes_background = annotation.get_passes_background()

# MLFlow
EXPERIMENT_NAME = "GAA CV Model"
client = mlflow.tracking.MlflowClient()
experiment_id = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment_id is None:
    EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)
else:
    EXPERIMENT_ID = experiment_id.experiment_id
RUN_NAME = f"run_2"
save_path = f'datasets/{RUN_NAME}/'
os.makedirs(save_path, exist_ok=True)
total_balls_detected = 0
with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=RUN_NAME) as run:

    for i, frame in enumerate(video):
        # Get Detections
        predictions, keypoint_predictions = std_detector.predict(frame)
        player_detections, ref_detections, ball_detections = std_detector.get_all_detections(predictions)
        if not ball_detections:
            ball_predictions = ball_detector.predict(frame)
            ball_detections = ball_detector.get_ball_detections(ball_predictions)
        keypoint_detections = std_detector.get_keypoint_detections(keypoint_predictions)
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
        keypoint_track_objects = keypoint_tracker.update(
            detections=keypoint_detections, coord_transformations=coord_transformations
        )
        # Integrate Detections & Tracks
        player_detections = Converter.TrackedObjects_to_Detections(player_track_objects)
        ball_detections = Converter.TrackedObjects_to_Detections(ball_track_objects)
        ref_detections = Converter.TrackedObjects_to_Detections(ref_track_objects)
        keypoint_detections = Converter.TrackedObjects_to_Detections(keypoint_track_objects)
        player_detections = classifier.predict_from_detections(
            detections=player_detections,
            img=frame,
        )
        # Compute Homography
        field_homography_estimator = FieldHomographyEstimator()
        field_homography_estimator.update_with_detections(keypoint_detections)
        # Match update
        ball = get_main_ball(ball_detections)
        referee = get_main_ref(ref_detections)
        players = Player.from_detections(detections=player_detections, teams=teams)
        # Apply Homography to Localize Players
        players = field_homography_estimator.apply_to_player(players)
        if players:
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
        frame = annotation.draw_possession_counter(
            config.fps, match, frame, counter_background=possession_background, debug=False
        )
        if ball:
            frame = ball.draw(frame)
            total_balls_detected += 1
        if referee:
            frame = referee.draw(frame)
        #frame = annotation.draw_passes_counter(
        #    match, frame, counter_background=passes_background, debug=False
        #)

        if i % 100 == 0:
            print(f"Metrics for frame {i}")
            frame.save(f'frame_{i}.png')  # Save to file

            metrics_dict = {}

            if players:
                for player in players:
                    # Create unique keys for each metric
                    x_key = f"Player_{player.detection.data['id']}_{player.team}_X"
                    y_key = f"Player_{player.detection.data['id']}_{player.team}_Y"

                    # Store the metrics in the dictionary
                    metrics_dict[x_key] = player.txy[0]
                    metrics_dict[y_key] = player.txy[1]
                    # Log player ID and team as params
                    param_key = f"Player_{player.detection.data['id']}_{i}"
                    mlflow.log_param(f"{param_key}_ID", player.detection.data['id'], )
                    mlflow.log_param(f"{param_key}_Team", player.team)

                # Add frame index to the metrics dictionary
                metrics_dict['Frame'] = i
                frame.save(f'{save_path}frame_{i}.png')
                # Add time in seconds
                metrics_dict['Timestamp'] = i / config.fps
                # Log metrics
                mlflow.log_metrics(metrics_dict)
        # Write video
        frame = np.array(frame)
        video.write(frame)


    # Generate summary stats
    match_stats = MatchStats(match)
    match_stats()
    mlflow.log_metric("Total balls detected", total_balls_detected)
    log_fail_counts_to_mlflow()
