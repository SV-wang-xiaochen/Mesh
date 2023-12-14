import trimesh
import numpy as np
import random
PITCH = 0.001
path = './voxel_results/p1.obj'
mesh = trimesh.load_mesh(path)
mesh.apply_scale(0.001)

voxelized_mesh = mesh.voxelized(PITCH).fill()
mesh_voxelization = np.around(np.array(voxelized_mesh.points), 4)
mesh_pcl = trimesh.PointCloud(vertices=np.array(mesh_voxelization), colors=[255, 0, 255, 100])

scene = trimesh.Scene()

scene.add_geometry(mesh_pcl)

scene.show(smooth=False)

