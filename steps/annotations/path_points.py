import numpy as np


class PathPoint:
    def __init__(
        self, id: int, center: tuple, color: tuple = (255, 255, 255), alpha: float = 1
    ):
        """
        Path point
        Move to annotation?
        Parameters
        ----------
        id : int
            Id of the point
        center : tuple
            Center of the point (x, y)
        color : tuple, optional
            Color of the point, by default (255, 255, 255)
        alpha : float, optional
            Alpha value of the point, by default 1
        """
        self.id = id
        self.center = center
        self.color = color
        self.alpha = alpha

    def __str__(self) -> str:
        return str(self.id)

    @property
    def color_with_alpha(self) -> tuple:
        return (self.color[0], self.color[1], self.color[2], int(self.alpha * 255))

    @staticmethod
    def get_center_from_bounding_box(bounding_box: np.ndarray) -> tuple:
        """
        Get the center of a bounding box

        Parameters
        ----------
        bounding_box : np.ndarray
            Bounding box [[xmin, ymin], [xmax, ymax]]

        Returns
        -------
        tuple
            Center of the bounding box (x, y)
        """
        return (
            int((bounding_box[0][0] + bounding_box[1][0]) / 2),
            int((bounding_box[0][1] + bounding_box[1][1]) / 2),
        )

    @staticmethod
    def from_abs_bbox(
        id: int,
        abs_point: np.ndarray,
        coord_transformations,
        color: tuple = None,
        alpha: float = None,
    ) -> "PathPoint":
        """
        Create a PathPoint from an absolute bounding box.
        It converts the absolute bounding box to a relative one and then to a center point

        Parameters
        ----------
        id : int
            Id of the point
        abs_point : np.ndarray
            Absolute bounding box
        coord_transformations : "CoordTransformations"
            Coordinate transformations
        color : tuple, optional
            Color of the point, by default None
        alpha : float, optional
            Alpha value of the point, by default None

        Returns
        -------
        PathPoint
            PathPoint
        """

        rel_point = coord_transformations.abs_to_rel(abs_point)
        center = PathPoint.get_center_from_bounding_box(rel_point)

        return PathPoint(id=id, center=center, color=color, alpha=alpha)
