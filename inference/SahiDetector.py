from typing import List

import pandas as pd
import torch
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from norfair import Detection


class SahiDetection:
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
            # Optionally add any other data like color, label, etc.

            norfair_detection = Detection(points=box, data=data, label=label)

            detections.append(norfair_detection)

        return detections

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
        print(results.object_prediction_list)
        return results

    def get_detections(self, detections):
        ball_detections = [
            detection for detection in detections if detection.label == 0 and detection.data["confidence"] > 0.1
        ]
        player_detections = [
            detection for detection in detections if detection.label == 3 and detection.data["confidence"] > 0.5
        ]
        ref_detections = [
            detection for detection in detections if detection.label == 4 and detection.data["confidence"] > 0.5
        ]
        return ball_detections, player_detections, ref_detections

    def generate_predictions_mask(
            predictions: pd.DataFrame, img: np.ndarray, margin: int = 0
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
        if type(predictions) != pd.DataFrame:
            raise TypeError("predictions must be a pandas dataframe")

        mask = np.ones(img.shape[:2], dtype=img.dtype)

        for index, row in predictions.iterrows():
            xmin = round(row["xmin"])
            ymin = round(row["ymin"])
            xmax = round(row["xmax"])
            ymax = round(row["ymax"])

            mask[ymin - margin: ymax + margin, xmin - margin: xmax + margin] = 0

        return mask
