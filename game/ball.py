import norfair
import numpy as np
from annotations.draw import draw_detection



class Ball:
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

    def set_color(self, match: "Match"):
        """
        Sets the color of the ball to the team color with the ball possession in the match.

        Parameters
        ----------
        match : Match
            Match object
        """
        if match.team_possession is None:
            return

        self.color = match.team_possession.color

        if self.detection:
            self.detection.data["color"] = match.team_possession.color

    def get_center(self, points: np.array) -> tuple:
        """
        Returns the center of the points

        Parameters
        ----------
        points : np.array
            2D points

        Returns
        -------
        tuple
            (x, y) coordinates of the center
        """
        x1, y1 = points[0]
        x2, y2 = points[1]

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        return (center_x, center_y)

    @property
    def center(self) -> tuple:
        """
        Returns the center of the ball

        Returns
        -------
        tuple
            Center of the ball (x, y)
        """
        if self.detection is None:
            if self.last_detection is None:
                return None

            center = self.get_center(self.last_detection.points)
            round_center = np.round_(center)

            return round_center

        center = self.get_center(self.detection.points)
        round_center = np.round_(center)

        return round_center

    @property
    def center_abs(self) -> tuple:
        """
        Returns the center of the ball in absolute coordinates

        Returns
        -------
        tuple
            Center of the ball (x, y)
        """
        if self.detection is None:
            return None

        center = self.get_center(self.detection.absolute_points)
        round_center = np.round_(center)

        return round_center

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

        return draw_detection(self.detection, frame, confidence=True)

    def __str__(self):
        return f"Ball: {self.center}"

    def from_xyxy(cls, xyxy_points):
        """
        Create a Ball instance from xyxy points.

        Parameters
        ----------
        xyxy_points : list or tuple
            List or tuple containing four xyxy coordinates.

        Returns
        -------
        Ball
            Ball instance created from the provided xyxy points.
        """
        detection = norfair.Detection(
            points=np.array([xyxy_points]),
            absolute_points=np.array([xyxy_points]),
        )
        return cls(detection)
