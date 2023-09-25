from typing import List, Tuple
import torch
import pandas as pd
import numpy as np
import norfair
from norfair import Detection
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction

# Include sahi & norfair in requirements.txt
# Should I make load_model & return_Detections classes static?


class BaseSahiDetection:
    def __init__(self, model_path: str = 'seg-5epoch.pt', config_path: str = 'data.yaml'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        self.model = self.load_model(model_path, config_path)

    def load_model(self, model_path, config_path):
        model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=model_path,
            config_path=config_path,
        )
        return model

    def return_detections(self, results):
        detections = []
        for detection in results.object_prediction_list:
            xmin = detection.bbox.minx
            ymin = detection.bbox.miny
            xmax = detection.bbox.maxx
            ymax = detection.bbox.maxy

            box = np.array(
                [
                    [xmin, ymin],
                    [xmax, ymax]
                ]
            )

            confidence = detection.score.value
            name = detection.category.name
            label = detection.category.id
            mask = detection.mask
            data = {
                "name": name,
                "confidence": confidence,
                "mask": mask
            }

            norfair_detection = Detection(points=box, label=label, data=data)
            detections.append(norfair_detection)

        return detections

    def generate_predictions_mask(self, predictions: pd.DataFrame, img: np.ndarray, margin: int = 0) -> np.ndarray:
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
        super().__init__(model_path='seg-5epoch.pt', config_path='data.yaml')

    def predict(self, frame: np.ndarray):
        height, width, _ = frame.shape
        results = get_sliced_prediction(
            frame,
            self.model,
            slice_height=270,
            slice_width=480,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )
        predictions = self.return_detections(results)
        return predictions

    def get_ball_detections(self, predictions: List[Detection]) -> List[norfair.Detection]:
        # Filter out the detections based on class_id and confidence
        ball_detections = [
            prediction for prediction in predictions
            if prediction.label == 0 and prediction.data['confidence'] > 0.3
        ]
        return ball_detections


class SahiPersonDetection(BaseSahiDetection):
    def __init__(self):
        super().__init__(model_path='seg5ep-no-tile.pt', config_path='data.yaml')

    def predict(self, frame: np.ndarray):
        height, width, _ = frame.shape
        results = get_prediction(frame, self.model)
        predictions = self.return_detections(results)
        return predictions

    def get_detections(self, predictions: List[Detection]) -> tuple[list[Detection], list[Detection]]:
        player_detections = [
            prediction for prediction in predictions
            if prediction.label == 3 and prediction.data['confidence'] > 0.5
        ]
        ref_detections = [
            prediction for prediction in predictions
            if prediction.label == 4 and prediction.data['confidence'] > 0.5
        ]
        # Can we access 2nd highest class prediction? Use to filter all players for low pred ref scores, this should lead to stronger ref class preds
        return player_detections, ref_detections
