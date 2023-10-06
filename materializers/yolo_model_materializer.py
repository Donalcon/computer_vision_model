import os
import torch
from typing import Any, Dict, Type

from zenml.materializers.base_materializer import BaseMaterializer
from zenml.enums import ArtifactType, VisualizationType
from zenml.metadata.metadata_types import MetadataType


class BallModelMaterializer(BaseMaterializer):
    """Materializer for handling YOLOv8 .pt files."""

    ASSOCIATED_ARTIFACT_TYPE: (ArtifactType.MODEL,)
    ASSOCIATED_TYPES: (torch.nn.Module,)

    def load(self, data_type: Type[Any]) -> Any:
        """Load YOLOv8 .pt model from disk."""
        model_path = os.path.join(self.uri, "models/seg-5epoch.pt")
        model = torch.load(model_path)
        return model

    def save(self, data: Any) -> None:
        """Save YOLOv8 .pt model to disk."""
        model_path = os.path.join(self.uri, "models/seg-5epoch.pt")
        torch.save(data, model_path)

    def save_visualizations(self, data: Any) -> Dict[str, VisualizationType]:
        """Save visualizations of the model (if any).

        Currently, this is left as an empty stub. Implement this if you
        have specific visualizations for your YOLOv8 models.
        """
        return {}

    def extract_metadata(self, data: Any) -> Dict[str, MetadataType]:
        """Extract metadata from YOLOv8 model.

        Here we might want to store some basic metadata about the model.
        For demonstration purposes, I'll assume the model has an attribute
        `num_classes` that tells us the number of object classes it can detect.
        """
        return {
            "names_": data.names if hasattr(data, 'names') else "Unknown",
            "nc": data.nc if hasattr(data, 'nc') else "Unknown"
        }


class PersonModelMaterializer(BaseMaterializer):
    """Materializer for handling YOLOv8 .pt files."""

    ASSOCIATED_ARTIFACT_TYPE: (ArtifactType.MODEL,)
    ASSOCIATED_TYPES: (torch.nn.Module,)

    def load(self, data_type: Type[Any]) -> Any:
        """Load YOLOv8 .pt model from disk."""
        model_path = os.path.join(self.uri, "models/seg5ep-no-tile.pt")
        model = torch.load(model_path)
        return model

    def save(self, data: Any) -> None:
        """Save YOLOv8 .pt model to disk."""
        model_path = os.path.join(self.uri, "models/seg5ep-no-tile.pt")
        torch.save(data, model_path)

    def save_visualizations(self, data: Any) -> Dict[str, VisualizationType]:
        """Save visualizations of the model (if any).

        Currently, this is left as an empty stub. Implement this if you
        have specific visualizations for your YOLOv8 models.
        """
        return {}

    def extract_metadata(self, data: Any) -> Dict[str, MetadataType]:
        """Extract metadata from YOLOv8 model.

        Here we might want to store some basic metadata about the model.
        For demonstration purposes, I'll assume the model has an attribute
        `num_classes` that tells us the number of object classes it can detect.
        """
        return {
            "names_": data.names if hasattr(data, 'names') else "Unknown",
            "nc": data.nc if hasattr(data, 'nc') else "Unknown"
        }
