import os
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.io import fileio

from typing import Any, Type
from zenml.enums import ArtifactType
from norfair import Video
import norfair
import cv2
import os
import pickle
import json


class VideoMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (norfair.Video,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: json) -> Video:
        """Read from artifact store."""
        with fileio.open(os.path.join(self.uri,'video_path.json'), 'r') as f:
            video_path = json.load(f)
        return Video(input_path=video_path)

    def save(self, data_type: Video) -> None:
        """Write to artifact store"""
        with fileio.open(os.path.join(self.uri, 'video_path.json'), 'w') as f:
            json.dump(data_type.input_path, f)


