"""Abstract base class for pose exporters."""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np


class BaseExporter(ABC):
    """Abstract base for exporting retargeted poses.

    Subclasses implement format-specific export logic.
    """

    @abstractmethod
    def export(
        self,
        rotations: Dict[str, np.ndarray],
        scales: Dict[str, float],
        output_path: str,
        positions: Optional[Dict[str, np.ndarray]] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Export retargeted pose to file.

        Args:
            rotations: Dict mapping joint name -> quaternion [w, x, y, z]
            scales: Dict mapping joint name -> scale factor
            output_path: Path to save the file
            positions: Optional joint positions for debugging
            metadata: Optional metadata (source_image, etc.)

        Returns:
            Path to saved file
        """
        pass

    @abstractmethod
    def to_dict(
        self,
        rotations: Dict[str, np.ndarray],
        scales: Dict[str, float],
        positions: Optional[Dict[str, np.ndarray]] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Convert retargeted pose to dict representation.

        Args:
            rotations: Dict mapping joint name -> quaternion [w, x, y, z]
            scales: Dict mapping joint name -> scale factor
            positions: Optional joint positions for debugging
            metadata: Optional metadata

        Returns:
            Dict representation of the pose
        """
        pass

    @property
    @abstractmethod
    def file_extension(self) -> str:
        """Get the file extension for this exporter."""
        pass
