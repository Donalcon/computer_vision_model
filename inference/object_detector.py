import torch
import numpy as np
from ultralytics import YOLO
import supervision as sv


class ObjectDetection:

    def __init__(self):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        self.model = self.load_model()
        self.box_annotator = sv.BoxAnnotator(sv.ColorPalette.default(), thickness=3, text_thickness=3, text_scale=1.5)

    @staticmethod
    def load_model():
        model = YOLO("yolov8m-football.pt")
        model.fuse()
        print(model.names)

        return model

    def predict(self, frame):

        results = self.model(frame)

        return results

    @staticmethod
    def return_detections(results):
        detections = sv.Detections(
                    xyxy=results[0].boxes.xyxy.cpu().numpy(),
                    confidence=results[0].boxes.conf.cpu().numpy(),
                    class_id=results[0].boxes.cls.cpu().numpy().astype(int),
                    )

        # Format custom labels
        # self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        #                for class_id, confidence in zip(detections.class_id, detections.confidence)]
        return detections

    def annotate(self):

        # Annotate and display frame
        frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=self.labels)

        return frame

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
        #print(type(predictions))
        #if type(predictions) != np.array:
        #    raise TypeError("predictions must be a pandas dataframe")

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
