from typing import List, Any
import torch
import pandas as pd
import numpy as np
import norfair
from norfair import Detection
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO
import pandas


class BaseDetection:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

    @staticmethod
    def return_detections(results):
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

    @staticmethod
    def generate_predictions_mask(predictions: pd.DataFrame, img: np.ndarray, margin: int = 0) -> np.ndarray:
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


class SahiBallDetection(BaseDetection):
    MODEL_PATH = 'models/seg-5epoch.pt'  # Replace with your actual model path
    CONFIG_PATH = 'models/data-tile.yaml'

    def __init__(self):
        super().__init__()
        self.load_model()

    def load_model(self):
        self.model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=self.MODEL_PATH,
            config_path=self.CONFIG_PATH,
        )

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

    @staticmethod
    def get_ball_detections(predictions: List[Detection]) -> List[norfair.Detection]:
        # Filter out the detections based on class_id and confidence
        ball_detections = [
            prediction for prediction in predictions
            if prediction.label == 0 and prediction.data['confidence'] > 0.3
        ]
        return ball_detections

    def __call__(self):
        if self.model is None:
            self.load_model(model_path=self.MODEL_PATH, config_path=self.CONFIG_PATH)


class Yolov8Detection(BaseDetection):
    MODEL_PATH = 'models/key-seg.pt'

    def __init__(self):
        super().__init__()
        self.load_model()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

    def load_model(self):
        self.model = YOLO(self.MODEL_PATH, 'segment')

    def predict(self, frame: np.ndarray):
        results = self.model.predict(frame)
        self.assign_names(results)
        person_predictions = return_detections(results, self.model)
        keypoint_predictions = self.return_keypoint_detections(results, self.model)
        return person_predictions, keypoint_predictions

    def assign_names(self, results):
        """Update the class_mappings dictionary."""
        if not hasattr(self.model, 'class_mappings'):
            # Assign each class name from the model to its corresponding ID
            self.model.class_mappings = {name: idx for idx, name in enumerate(self.model.names)}
            print(self.model.class_mappings)

    @staticmethod
    def return_keypoint_detections(results, model):
        detections = []

        for r in results:
            # xy seems to be the shape of the mask?
            masks = r.masks.data
            # Assuming this is a list
            boxes = r.boxes.cpu().numpy()  # Assuming this is a list

            for mask, box in zip(masks, boxes):
                points = box.xyxy
                xmin, ymin, xmax, ymax = points[0]
                points = np.array(
                    [
                        [xmin, ymin],
                        [xmax, ymax]
                    ]
                )
                confidence = box.conf
                label = int(box.cls[0])
                class_name = model.names[label]
                mask = mask
                keypoint_x = (xmin + xmax) / 2
                keypoint_y = (ymin + ymax) / 2
                xy = np.array([keypoint_x, keypoint_y])
                data = {
                    "label": label,
                    "name": class_name,
                    "confidence": confidence,
                    "mask": mask,  # Now each mask corresponds to the right box
                    "xy": xy,
                }
                norfair_detection = Detection(points=points, label=label, data=data)
                detections.append(norfair_detection)

        return detections

    @staticmethod
    def get_all_detections(predictions: List[Detection]) -> tuple[list[Detection], list[Detection], list[Detection]]:
        player_detections = []
        ref_detections = []
        ball_detections = []

        for prediction in predictions:
            if prediction.data['confidence'] > 0.5:
                if prediction.data["name"] == 'players':
                    player_detections.append(prediction)
                elif prediction.data["name"] == 'referee':
                    ref_detections.append(prediction)
            else:
                if prediction.data["confidence"] > 0.3:
                    if prediction.data["name"] == 'ball':
                        ball_detections.append(prediction)

        return player_detections, ref_detections, ball_detections

    # labels are in class numbers, not class names!!
    @staticmethod
    def get_keypoint_detections(predictions: List[Detection]) -> list[Detection]:
        keypoint_labels = [
            "0A", "0B", "1E", "1F", "1L", "1N", "1O", "1P", "1R", "1T",
            "2B", "2C", "2D", "2E", "2F", "2G", "2GPA", "2GPB", "2H", "2J",
            "2K", "2L", "2N", "2O", "2P", "2Q", "2R", "2T"
        ]
        keypoint_detections = [
            prediction for prediction in predictions
            if prediction.data["name"] in keypoint_labels and prediction.data['confidence'] > 0.05
        ]
        return keypoint_detections

    def __call__(self):
        if self.model is None:
            self.load_model(self.MODEL_PATH)


def return_detections(results: Any, model: YOLO) -> List[norfair.Detection]:
    detections = []

    for idx, r in enumerate(results):
        masks = r.masks.data
        boxes = r.boxes.cpu().numpy()
        for mask, box in zip(masks, boxes):
            points = box.xyxy
            xmin, ymin, xmax, ymax = points[0]
            points = np.array(
                [
                    [xmin, ymin],
                    [xmax, ymax]
                ]
            )
            confidence = box.conf
            label = int(box.cls[0])
            class_name = model.names[label]
            mask = mask
            data = {
                "label": label,
                "name": class_name,
                "confidence": confidence,
                "mask": mask,
                "txy": [],
            }
            norfair_detection = Detection(points=points, label=label, data=data)
            detections.append(norfair_detection)
            print(norfair_detection.data["mask"][0]) # how can i save this as a variable to view? make sure not all 0s

    return detections
