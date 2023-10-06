from zenml import pipeline
from materializers import VideoMaterializer, TeamMaterializer

from steps.load_annotators import annotator
from steps.load_stats import generate_match_stats
from steps.instantiate_match import instantiate_match
from steps.load_detectors import detectors
from steps.load_classifier import classifier
from steps.load_tracker import trackers
from steps.frame_loop import frame_loop
from steps.config import game_config
import json


@pipeline()
def inference_pipeline():
    video, home, away, fps = game_config()
    print(video)
    ball_detector, person_detector = detectors()
    player_classifier = classifier()
    match, teams = instantiate_match(home, away, fps)
    player_tracker, referee_tracker, ball_tracker, motion_estimator, coord_transformations = trackers()
    ball_path, possession_background, passes_background = annotator()
    frame_loop(
        video,
        ball_detector,
        person_detector,
        player_tracker,
        ball_tracker,
        referee_tracker,
        motion_estimator,
        player_classifier,
        teams,
        match,
        ball_path,
        fps,
        possession_background,
        passes_background)
    generate_match_stats(match=match)

inference_pipeline()
