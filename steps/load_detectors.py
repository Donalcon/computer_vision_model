from typing import Tuple

from zenml import step
from steps.inference.sahi_detector import SahiBallDetection, SahiPersonDetection


@step
def detectors() -> Tuple[SahiBallDetection, SahiPersonDetection]:
    # Initialize SahiBallDetection and SahiPersonDetection
    ball_detector = SahiBallDetection()
    ball_detector.load_model(model_path='models/seg-5epoch.pt', config_path='data.yaml')

    person_detector = SahiPersonDetection()
    person_detector.load_model(model_path='models/seg5ep-no-tile.pt', config_path='data.yaml')

    return ball_detector, person_detector
