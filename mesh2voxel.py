import trimesh
import numpy as np
from bottom_vertex_index import bottom_vertex_index
import glob

MESH_NR = 0
path = r'C:\Users\xiaochen.wang\Projects\Dataset\FLORENCE'
obj_list = glob.glob(f'{path}/**/*.obj', recursive=True)

head_mesh = trimesh.load_mesh(obj_list[MESH_NR])
bottom_vertex = head_mesh.vertices[bottom_vertex_index]
bottom_vertex_pcl = trimesh.PointCloud(vertices=bottom_vertex)
hole_boundary_vertices = bottom_vertex

bottom_vertices_np = np.array(hole_boundary_vertices)
bottom_vertices_np_swap = np.empty(bottom_vertices_np.shape)
bottom_vertices_np_swap[:, [0, 1, 2]] = bottom_vertices_np[:, [0, 2, 1]]

bottom_vertices_selected = bottom_vertices_np_swap[:,:2]

from scipy.spatial import Delaunay
tri = Delaunay(bottom_vertices_selected)

faces = np.array(tri.simplices)
converted_faces = np.array([bottom_vertex_index[val] for row in faces for val in row]).reshape(faces.shape)

# Add the new faces to the mesh
head_mesh.faces = np.append(head_mesh.faces, converted_faces, axis=0)

voxelized_mesh = head_mesh.voxelized(0.005)
print(voxelized_mesh.filled_count)
voxelized_mesh.fill()
print(voxelized_mesh.filled_count)
print(voxelized_mesh.points_to_indices(voxelized_mesh.points).shape)
scene = trimesh.Scene()

scene.add_geometry(voxelized_mesh.as_boxes())

scene.show()
