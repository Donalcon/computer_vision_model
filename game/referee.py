import norfair
import numpy as np
from annotations.draw import draw_detection


class Referee:
    def __init__(self, detection: norfair.Detection):
        """
        Initialize Ball

        Parameters
        ----------
        detection : norfair.Detection
            norfair.Detection containing the ball
        """
        self.detection = detection

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

        return draw_detection(self.detection, frame)
