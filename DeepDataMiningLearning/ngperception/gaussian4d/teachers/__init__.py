from .base import TeacherTarget, CLASS_NAMES, TAIL_CLASSES, FREE
from .voxel_teacher import VoxelTeacher
from .gaussian_teacher import GaussianTeacher

def build_teacher(kind, nusc, labelgen_cache):
    """Factory: kind in {voxel1, voxel10, gaussian, gaussian10}."""
    if kind == "voxel1":     return VoxelTeacher(nusc, labelgen_cache, sweeps=1)
    if kind == "voxel10":    return VoxelTeacher(nusc, labelgen_cache, sweeps=10)
    if kind == "gaussian":   return GaussianTeacher(nusc, labelgen_cache, sweeps=1)
    if kind == "gaussian10": return GaussianTeacher(nusc, labelgen_cache, sweeps=10)
    raise ValueError(kind)
