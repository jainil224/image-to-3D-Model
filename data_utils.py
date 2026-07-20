import numpy as np
from skimage import measure
import trimesh


def read_binvox(path):
    """Reads a .binvox file and returns a (D,H,W) numpy boolean array."""
    with open(path, "rb") as f:
        line = f.readline().strip()
        if not line.startswith(b"#binvox"):
            raise IOError("Not a binvox file")
        dims = list(map(int, f.readline().strip().split(b" ")[1:]))
        _ = f.readline()  # translate
        _ = f.readline()  # scale
        _ = f.readline()  # data
        raw_data = np.frombuffer(f.read(), dtype=np.uint8)

    values, counts = raw_data[::2], raw_data[1::2]
    voxels = np.repeat(values, counts).astype(bool)
    voxels = voxels.reshape(dims)
    return voxels


def voxels_to_mesh(voxel_grid, threshold=0.5):
    volume = voxel_grid.astype(float)
    if volume.size == 0:
        return trimesh.creation.icosphere(radius=0.2)

    volume_min = float(volume.min())
    volume_max = float(volume.max())
    if not np.isfinite(volume_min) or not np.isfinite(volume_max):
        volume = np.nan_to_num(volume, nan=0.0, posinf=1.0, neginf=0.0)
        volume_min = float(volume.min())
        volume_max = float(volume.max())

    if threshold <= volume_min or threshold >= volume_max:
        if np.count_nonzero(volume > 0.5) == 0:
            return trimesh.creation.box(extents=[0.5, 0.5, 0.5])
        volume = np.clip(volume, 0.0, 1.0)
        threshold = 0.5

    try:
        padded = np.pad(volume, 1, mode="constant")
        verts, faces, normals, _ = measure.marching_cubes(padded, level=threshold)
        return trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    except ValueError:
        return trimesh.creation.box(extents=[0.5, 0.5, 0.5])
