from annotations import annotation
import numpy as np
from PIL import Image
from norfair import Tracker
from norfair.camera_motion import MotionEstimator
from norfair.distances import mean_euclidean
from config import Config
from inference import Converter, InertiaClassifier, NNClassifier
from inference.sahi_detector import SahiBallDetection, SahiPersonDetection
from run_utils import (
    get_main_ball,
    get_main_ref,
    update_motion_estimator,
)
from game import Match, Player, Team, MatchStats
from annotations.paths import AbsolutePath

config = Config.from_args()

# Your single frame should be in NumPy array format
single_frame = np.array(Image.open('annotations/images/image_156.jpg'))

# Object Detectors
ball_detector = SahiBallDetection()
ball_detector.load_model(model_path='models/seg-5epoch.pt', config_path='data.yaml')
person_detector = SahiPersonDetection()
person_detector.load_model(model_path='models/seg5ep-no-tile.pt', config_path='data.yaml')

# Color Classifier
nn_classifier = NNClassifier('models/model_path.pt', ['dublin', 'kerry', 'referee'])
classifier = InertiaClassifier(classifier=nn_classifier, inertia=20)

# Instantiate Match
home = Team(name=config.home['name'], color=config.home['color'], abbreviation=config.home['abbreviation'], text_color=config.home['text_color'])
away = Team(name=config.away['name'], color=config.away['color'], abbreviation=config.away['abbreviation'], text_color=config.away['text_color'])
teams = [home, away]
match = Match(home, away, fps=config.fps)
match.team_possession = home

# Tracking setup
player_tracker = Tracker(distance_function=mean_euclidean, distance_threshold=250, initialization_delay=3, hit_counter_max=90)
referee_tracker = Tracker(distance_function=mean_euclidean, distance_threshold=250, initialization_delay=3, hit_counter_max=90)
ball_tracker = Tracker(distance_function=mean_euclidean, distance_threshold=250, initialization_delay=3, hit_counter_max=2000)

motion_estimator = MotionEstimator()
coord_transformations = None

# Instantiate Ball Path
path = AbsolutePath()

possession_background = annotation.get_possession_background()

# Process Single Frame
ball_predictions = ball_detector.predict(single_frame)
person_predictions = person_detector.predict(single_frame)
ball_detections = ball_detector.get_ball_detections(ball_predictions)
player_detections, ref_detections = person_detector.get_person_detections(person_predictions)
detections = ball_detections + player_detections + ref_detections

coord_transformations = update_motion_estimator(motion_estimator=motion_estimator, detections=detections, frame=single_frame)
player_track_objects = player_tracker.update(detections=player_detections, coord_transformations=coord_transformations)
ball_track_objects = ball_tracker.update(detections=ball_detections, coord_transformations=coord_transformations)
ref_track_objects = referee_tracker.update(detections=ref_detections, coord_transformations=coord_transformations)

player_detections = Converter.TrackedObjects_to_Detections(player_track_objects)
ball_detections = Converter.TrackedObjects_to_Detections(ball_track_objects)
ref_detections = Converter.TrackedObjects_to_Detections(ref_track_objects)
player_detections = classifier.predict_from_detections(detections=player_detections, img=single_frame)

ball = get_main_ball(ball_detections)
referee = get_main_ref(ref_detections)
players = Player.from_detections(detections=player_detections, teams=teams)
match.update(players, ball)
single_frame = Image.fromarray(single_frame)

single_frame = Player.draw_players(players=players, frame=single_frame, confidence=False, id=True)
single_frame = path.draw(img=single_frame, detection=ball.detection, coord_transformations=coord_transformations, color=match.team_possession.color)
single_frame = annotation.draw_possession_counter(config.fps, match, single_frame, counter_background=possession_background, debug=False)
if ball:
    single_frame = ball.draw(single_frame)
if referee:
    single_frame = referee.draw(single_frame)

single_frame.show()
