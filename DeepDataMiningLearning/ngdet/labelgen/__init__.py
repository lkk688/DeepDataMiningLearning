"""ngdet.labelgen — dense 2-D auto-labeling: open-vocab semantic seg (Grounded-SAM) + segment-level
metric depth (DepthAnything + LiDAR). Frozen foundation models, no training. See TUTORIAL.md §24."""
from .labeler import GroundedLabeler, Taxonomy, NUSC_TAXONOMY
from .sources import (NuScenesSource, WaymoSource, KittiSource, ImageFolderSource,
                      project_pinhole, project_ftheta)
from .physicalai import PhysicalAISource
from .av2 import AV2Source
__all__ = ["GroundedLabeler", "Taxonomy", "NUSC_TAXONOMY", "NuScenesSource", "WaymoSource",
           "KittiSource", "ImageFolderSource", "project_pinhole", "project_ftheta", "PhysicalAISource", "AV2Source"]
