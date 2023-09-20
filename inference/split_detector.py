from typing import List
import torch
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
from norfair import Detection
# Include sahi & norfair in requirements.txt
# Should I make load_model & return_Detections classes static?


class BaseSahiDetection:
    def __init__(self, model_path, config_path, confidence_threshold):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        self.model = self.load_model(model_path, config_path, confidence_threshold)

    def load_model(self, model_path, config_path, confidence_threshold):
        model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=model_path,
            config_path=config_path,
            confidence_threshold=confidence_threshold
        )
        return model

    def return_detections(self, results):
        detections = []
        for detection in results.object_prediction_list:
            xyxy = (detection.bbox.minx, detection.bbox.miny, detection.bbox.maxx, detection.bbox.maxy)
            confidence = detection.score.value
            class_id = detection.category.id
            mask = detection.mask

            points = np.array([(xyxy[0], xyxy[1]), (xyxy[2], xyxy[3])])
            norfair_detection = Detection(points=points,
                                          scores=np.array([confidence]),
                                          label=class_id,
                                          data=mask)
            detections.append(norfair_detection)

        return detections

    def generate_predictions_mask(
        predictions: np.array, img: np.ndarray, margin: int = 0
    ) -> np.ndarray:
        """
        Generates a mask of the predictions bounding boxes

        Parameters
        ----------
        predictions : pd.DataFrame
            DataFrame containing the bounding boxes and the class of the objects
        img : np.ndarray
            Image where the predictions were made
        margin : int, optional
            Margin to add to the bounding box, by default 0

        Returns
        -------
        np.ndarray
            Mask of the predictions bounding boxes

        Raises
        ------
        TypeError
            If predictions type is not pd.DataFrame
        """
        print(type(predictions))
        if type(predictions) != np.array:
            raise TypeError("predictions must be a pandas dataframe")

        mask = np.ones(img.shape[:2], dtype=img.dtype)

        for index, row in predictions.iterrows():

            xmin = round(row["xmin"])
            ymin = round(row["ymin"])
            xmax = round(row["xmax"])
            ymax = round(row["ymax"])

            mask[ymin - margin: ymax + margin, xmin - margin: xmax + margin] = 0

        return mask


class SahiBallDetection(BaseSahiDetection):

    def __init__(self):
        super().__init__(model_path='seg-5epoch.pt', config_path='data.yaml', confidence_threshold=0.05)

    def predict(self, frame):
        height, width, _ = frame.shape
        slice_height = height / 4
        slice_width = width / 4
        results = get_sliced_prediction(
            frame,
            self.model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )
        return results

    def get_ball_detections(self, frame: np.ndarray) -> List[Detection]:
        ball = self.predict(frame)
        detections = self.return_detections(ball)
        # Filter out the detections based on class_id and confidence
        ball_detections = [
            detection for detection in detections
            if detection.label == 0 and detection.scores[0] > 0.1
        ]
        return ball_detections


class SahiPersonDetection(BaseSahiDetection):
    def __init__(self):
        super().__init__(model_path='seg5ep-no-tile.pt', config_path='data.yaml', confidence_threshold=0.3)

    def predict(self, frame):
        height, width, _ = frame.shape
        results = get_prediction(frame, self.model)
        return results

    def get_player_detections(self, detections: List[Detection]) -> List[Detection]:
        player_detections = [
            detection for detection in detections
            if detection.label == 3 and detection.scores[0] > 0.5
        ]
        return player_detections

    def get_referee_detections(self, detections: List[Detection]) -> List[Detection]:
        ref_detections = [
            detection for detection in detections
            if detection.label == 4 and detection.scores[0] > 0.5
        ]
        return ref_detections
