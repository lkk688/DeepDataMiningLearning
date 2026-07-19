from .base import TeacherTarget, CLASS_NAMES, TAIL_CLASSES, FREE
from .voxel_teacher import VoxelTeacher
from .gaussian_teacher import GaussianTeacher

def build_teacher(kind, nusc, labelgen_cache, soft_cache=None):
    """Factory: kind in {voxel1, voxel10, gaussian, gaussian10}. `soft_cache` set = soft top-K
    distribution teacher (the {voxel,gaussian}×{hard,soft} 2×2); None = hard argmax."""
    if kind == "voxel1":     return VoxelTeacher(nusc, labelgen_cache, sweeps=1, soft_cache=soft_cache)
    if kind == "voxel10":    return VoxelTeacher(nusc, labelgen_cache, sweeps=10, soft_cache=soft_cache)
    if kind == "gaussian":   return GaussianTeacher(nusc, labelgen_cache, sweeps=1, soft_cache=soft_cache)
    if kind == "gaussian10": return GaussianTeacher(nusc, labelgen_cache, sweeps=10, soft_cache=soft_cache)
    raise ValueError(kind)
