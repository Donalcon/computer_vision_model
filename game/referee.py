import cv2
import norfair
import numpy as np

from game.draw import Draw


class Referee:
    def __init__(self, detection: norfair.Detection):
        """
        Initialize Ball

        Parameters
        ----------
        detection : norfair.Detection
            norfair.Detection containing the ball
        """
        self.detection = None
        self.last_detection = None
        self.color = None

        # Assign the initial detection
        self.update_detection(detection)

    def update_detection(self, new_detection: norfair.Detection):
        """
        Update the ball detection and store the previous detection.

        Parameters
        ----------
        new_detection : norfair.Detection
            New detection of the ball
        """
        self.last_detection = self.detection
        self.detection = new_detection


    def draw(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw the ball on the frame

        Parameters
        ----------
        frame : np.ndarray
            Frame to draw on

        Returns
        -------
        np.ndarray
            Frame with ball drawn
        """
        if self.detection is None:
            return frame

        return Draw.draw_detection(self.detection, frame)

