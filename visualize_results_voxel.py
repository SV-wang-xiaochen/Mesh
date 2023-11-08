import trimesh
import numpy as np

PITCH = 0.0025
num_of_heads = np.load(f"voxel_results/num_of_heads_{PITCH}.npy")
voxel_center_min = np.load(f"voxel_results/voxel_center_min_{PITCH}.npy")
voxel_center_max = np.load(f"voxel_results/voxel_center_max_{PITCH}.npy")
voxel_list_remove_zero = np.load(f"voxel_results/voxel_list_remove_zero_{PITCH}.npy")
colors_list = np.load(f"voxel_results/colors_list_{PITCH}.npy")
accumulation_remove_zero = np.load(f"voxel_results/accumulation_remove_zero_{PITCH}.npy")
print(f"num of heads:{num_of_heads}")

scene = trimesh.Scene()

V = trimesh.PointCloud(vertices=voxel_list_remove_zero, colors=colors_list)

# show the voxels whose head hits >=3, this is the reference head that the lens must not intersect
valid_voxel = [voxel_list_remove_zero[i] for i in range(len(accumulation_remove_zero)) if accumulation_remove_zero[i] >= 3]
valid_hit = [colors_list[i] for i in range(len(accumulation_remove_zero)) if accumulation_remove_zero[i] >= 3]
V_valid = trimesh.PointCloud(vertices=valid_voxel, colors=valid_hit)

scene.add_geometry(V_valid)
scene.show(smooth=False, flags={'wireframe': True})
