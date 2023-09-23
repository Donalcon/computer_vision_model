from typing import List
from math import sqrt
import PIL
import norfair
import numpy as np
from annotations.path_points import PathPoint


class AbsolutePath:
    def __init__(self) -> None:
        self.past_points = []
        self.color_by_index = {}

    def center(self, points: np.ndarray) -> tuple:
        """
        Get the center of a Norfair Bounding Box Detection point

        Parameters
        ----------
        points : np.ndarray
            Norfair Bounding Box Detection point

        Returns
        -------
        tuple
            Center of the point (x, y)
        """
        return (
            int((points[0][0] + points[1][0]) / 2),
            int((points[0][1] + points[1][1]) / 2),
        )

    @property
    def path_length(self) -> int:
        return len(self.past_points)

    def draw_path_slow(
        self,
        img: PIL.Image.Image,
        path: List[PathPoint],
        thickness: int = 4,
    ) -> PIL.Image.Image:
        """
        Draw a path with alpha

        Parameters
        ----------
        img : PIL.Image.Image
            Image
        path : List[PathPoint]
            List of points to draw
        thickness : int, optional
            Thickness of the path, by default 4

        Returns
        -------
        PIL.Image.Image
            Image with the path drawn
        """
        draw = PIL.ImageDraw.Draw(img, "RGBA")

        for i in range(len(path) - 1):
            draw.line(
                [path[i].center, path[i + 1].center],
                fill=path[i].color_with_alpha,
                width=thickness,
            )
        return img

    def draw_arrow_head(
        self,
        img: PIL.Image.Image,
        start: tuple,
        end: tuple,
        color: tuple = (255, 255, 255),
        length: int = 10,
        height: int = 6,
        thickness: int = 4,
        alpha: int = 255,
    ) -> PIL.Image.Image:

        # https://stackoverflow.com/questions/43527894/drawing-arrowheads-which-follow-the-direction-of-the-line-in-pygame
        draw = PIL.ImageDraw.Draw(img, "RGBA")

        dX = end[0] - start[0]
        dY = end[1] - start[1]

        # vector length
        Len = sqrt(dX * dX + dY * dY)  # use Hypot if available

        if Len == 0:
            return img

        # normalized direction vector components
        udX = dX / Len
        udY = dY / Len

        # perpendicular vector
        perpX = -udY
        perpY = udX

        # points forming arrowhead
        # with length L and half-width H
        arrowend = end

        leftX = end[0] - length * udX + height * perpX
        leftY = end[1] - length * udY + height * perpY

        rightX = end[0] - length * udX - height * perpX
        rightY = end[1] - length * udY - height * perpY

        if len(color) <= 3:
            color += (alpha,)

        draw.line(
            [(leftX, leftY), arrowend],
            fill=color,
            width=thickness,
        )

        draw.line(
            [(rightX, rightY), arrowend],
            fill=color,
            width=thickness,
        )

        return img

    def draw_path_arrows(
        self,
        img: PIL.Image.Image,
        path: List[PathPoint],
        thickness: int = 4,
        frame_frequency: int = 30,
    ) -> PIL.Image.Image:
        """
        Draw a path with arrows every 30 points

        Parameters
        ----------
        img : PIL.Image.Image
            Image
        path : List[PathPoint]
            Path
        thickness : int, optional
            Thickness of the path, by default 4

        Returns
        -------
        PIL.Image.Image
            Image with the arrows drawn
        """

        for i, point in enumerate(path):

            if i < 4 or i % frame_frequency:
                continue

            end = path[i]
            start = path[i - 4]

            img = self.draw_arrow_head(
                img=img,
                start=start.center,
                end=end.center,
                color=start.color_with_alpha,
                thickness=thickness,
            )

        return img

    def draw_path_fast(
        self,
        img: PIL.Image.Image,
        path: List[PathPoint],
        color: tuple,
        width: int = 2,
        alpha: int = 255,
    ) -> PIL.Image.Image:
        """
        Draw a path without alpha (faster)

        Parameters
        ----------
        img : PIL.Image.Image
            Image
        path : List[PathPoint]
            Path
        color : tuple
            Color of the path
        with : int
            Width of the line
        alpha : int
            Color alpha (0-255)

        Returns
        -------
        PIL.Image.Image
            Image with the path drawn
        """
        draw = PIL.ImageDraw.Draw(img, "RGBA")

        path_list = [point.center for point in path]

        color += (alpha,)

        draw.line(
            path_list,
            fill=color,
            width=width,
        )

        return img

    def draw_arrow(
        self,
        img: PIL.Image.Image,
        points: List[PathPoint],
        color: tuple,
        width: int,
        alpha: int = 255,
    ) -> PIL.Image.Image:
        """Draw arrow between two points

        Parameters
        ----------
        img : PIL.Image.Image
            image to draw
        points : List[PathPoint]
            start and end points
        color : tuple
            color of the arrow
        width : int
            width of the arrow
        alpha : int, optional
            color alpha (0-255), by default 255

        Returns
        -------
        PIL.Image.Image
            Image with the arrow
        """

        img = self.draw_path_fast(
            img=img, path=points, color=color, width=width, alpha=alpha
        )
        img = self.draw_arrow_head(
            img=img,
            start=points[0].center,
            end=points[1].center,
            color=color,
            length=30,
            height=15,
            alpha=alpha,
        )

        return img

    def add_new_point(
        self, detection: norfair.Detection, color: tuple = (255, 255, 255)
    ) -> None:
        """
        Add a new point to the path

        Parameters
        ----------
        detection : norfair.Detection
            Detection
        color : tuple, optional
            Color of the point, by default (255, 255, 255)
        """

        if detection is None:
            return

        self.past_points.append(detection.absolute_points)

        self.color_by_index[len(self.past_points) - 1] = color

    def filter_points_outside_frame(
        self, path: List[PathPoint], width: int, height: int, margin: int = 0
    ) -> List[PathPoint]:
        """
        Filter points outside the frame with a margin

        Parameters
        ----------
        path : List[PathPoint]
            List of points
        width : int
            Width of the frame
        height : int
            Height of the frame
        margin : int, optional
            Margin, by default 0

        Returns
        -------
        List[PathPoint]
            List of points inside the frame with the margin
        """

        return [
            point
            for point in path
            if point.center[0] > 0 - margin
            and point.center[1] > 0 - margin
            and point.center[0] < width + margin
            and point.center[1] < height + margin
        ]

    def draw(
        self,
        img: PIL.Image.Image,
        detection: norfair.Detection,
        coord_transformations,
        color: tuple = (255, 255, 255),
    ) -> PIL.Image.Image:
        """
        Draw the path

        Parameters
        ----------
        img : PIL.Image.Image
            Image
        detection : norfair.Detection
            Detection
        coord_transformations : _type_
            Coordinate transformations
        color : tuple, optional
            Color of the path, by default (255, 255, 255)

        Returns
        -------
        PIL.Image.Image
            Image with the path drawn
        """

        self.add_new_point(detection=detection, color=color)

        if len(self.past_points) < 2:
            return img

        path = [
            PathPoint.from_abs_bbox(
                id=i,
                abs_point=point,
                coord_transformations=coord_transformations,
                alpha=i / (1.2 * self.path_length),
                color=self.color_by_index[i],
            )
            for i, point in enumerate(self.past_points)
        ]

        path_filtered = self.filter_points_outside_frame(
            path=path,
            width=img.size[0],
            height=img.size[0],
            margin=250,
        )

        img = self.draw_path_slow(img=img, path=path_filtered)
        img = self.draw_path_arrows(img=img, path=path)

        return img