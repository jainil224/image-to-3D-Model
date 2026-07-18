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
    padded = np.pad(voxel_grid.astype(float), 1, mode="constant")
    verts, faces, normals, _ = measure.marching_cubes(padded, level=threshold)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    return mesh
