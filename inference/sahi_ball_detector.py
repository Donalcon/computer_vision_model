import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
from inference.sahi import SahiDetector
import supervision as sv
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict

from dataclasses import dataclass
from typing import List

@dataclass
class DetectionInfo:
    xyxy: List[int]
    confidence: List[float]
    class_id: List[int]

class SahiBallDetection:

    def __init__(self):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

        self.model = self.load_model()

    def load_model(self):

        model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path='seg-5epoch.pt',
            config_path='data.yaml',
            confidence_threshold=0.15,
        )
        return model

    def predict(self, frame):
        height, width, _ = frame.shape
        results = get_sliced_prediction(
            frame,
            self.model,
            slice_height=270,
            slice_width=480,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )

        return results

    def return_Detections(self, results):
        detection_list = []
        for pred in results.object_prediction_list:
            xyxy = (pred.bbox.minx, pred.bbox.miny, pred.bbox.maxx, pred.bbox.maxy)
            confidence = pred.score.value
            class_id = pred.category.id
            mask = pred.mask
            detection_list.append((xyxy, confidence, class_id, mask))

        return detection_list

    @staticmethod
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

            mask[ymin - margin : ymax + margin, xmin - margin : xmax + margin] = 0

        return mask
