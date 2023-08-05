from array import array
from typing import List

import numpy as np
import PIL
import norfair
from norfair import Detection

from soccer.ball import Ball
from soccer.draw import Draw
from soccer.team import Team


class Referee:
    def __init__(self, detection: Detection):
        """

        Initialize Referee

        Parameters
        ----------
        detection : Detection
            Detection containing the referee
        """
        self.detection = detection

    def draw(
        self, frame: PIL.Image.Image, confidence: bool = False, id: bool = False
    ) -> PIL.Image.Image:
        """
        Draw the player on the frame

        Parameters
        ----------
        frame : PIL.Image.Image
            Frame to draw on
        confidence : bool, optional
            Whether to draw confidence text in bounding box, by default False
        id : bool, optional
            Whether to draw id text in bounding box, by default False

        Returns
        -------
        PIL.Image.Image
            Frame with player drawn
        """
        if self.detection is None:
            return frame

        self.detection.data["color"] = (0, 0, 0)

        return Draw.draw_detection(self.detection, frame, confidence=confidence, id=id)

