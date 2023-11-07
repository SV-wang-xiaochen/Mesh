import trimesh
import numpy as np

PITCH = 0.0025
num_of_heads = np.load(f"num_of_heads_{PITCH}.npy")
voxel_center_min = np.load(f"voxel_center_min_{PITCH}.npy")
voxel_center_max = np.load(f"voxel_center_max_{PITCH}.npy")
voxel_list_remove_zero = np.load(f"voxel_list_remove_zero_{PITCH}.npy")
colors_list = np.load(f"colors_list_{PITCH}.npy")
print(f"num of heads:{num_of_heads}")

scene = trimesh.Scene()

V = trimesh.PointCloud(vertices=voxel_list_remove_zero, colors=colors_list)

scene.add_geometry(V)
scene.show()
