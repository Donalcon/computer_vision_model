import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
from inference.sahi import SahiDetector
import supervision as sv
from sahi import AutoDetectionModel
from sahi.models.custom import Yolov8DetectionModel
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

        #self.capture_index = capture_index

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

        self.model = self.load_model()



        self.box_annotator = sv.BoxAnnotator(sv.ColorPalette.default(), thickness=3, text_thickness=3, text_scale=1.5)

    def load_model(self):

        model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path='seg5ep-notile.pt',
            confidence_threshold=0.7,
        )

        return model

    def predict(self, frame):
        height, width, _ = frame.shape
        results = get_prediction(
            frame,
            self.model,
        )

        return results

    def return_Detections(self, results):
        detection_list = []
        print(results.object_prediction_list)
        print(dir(results.object_prediction_list))
        for pred in results.object_prediction_list:
            xyxy = (pred.bbox.minx, pred.bbox.miny, pred.bbox.maxx, pred.bbox.maxy)
            confidence = pred.score.value
            class_id = pred.category.id
            mask = pred.mask

            detection_list.append((xyxy, confidence, class_id), mask)

            # Create an instance of the DetectionInfo class with the collected variables
            # detection_info = DetectionInfo(xyxy=xyxy, confidence=confidence, class_id=class_id)
            # detection_list.append(detection_info)

        return detection_list

        # Setup detections for visualization
        #detections = sv.Detections(
        #            xyxy=pred_list[0].bbox,
        #            confidence=pred_list[0].score,
        #            class_id=pred_list[0].category,
        #            )

        # Format custom labels
        # self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        #                for class_id, confidence in zip(detections.class_id, detections.confidence)]
        #return detections

    #def annotate(self):

        # Annotate and display frame
        # frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=self.labels)

        #return frame

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


    def __call__(self, frame):
        results = self.predict(frame)
        detections = self.plot_bboxes(results, frame)
        return detections
