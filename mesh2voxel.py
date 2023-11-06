import trimesh
import numpy as np
from bottom_vertex_index import bottom_vertex_index
import glob

import random


def generate_random_color():
    # Generate random values for red, green, and blue components
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)

    # Create the color list in the specified format [r, g, b, 100]
    color = [r, g, b, 100]
    return color

MESH_NR = 0
path = r'C:\Users\xiaochen.wang\Projects\Dataset\FLORENCE'
obj_list = glob.glob(f'{path}/**/*.obj', recursive=True)
scene = trimesh.Scene()

for mesh_nr in range(0, 10):
    head_mesh = trimesh.load_mesh(obj_list[mesh_nr])
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
    print(head_mesh.is_watertight)
    voxelized_mesh = head_mesh.voxelized(0.005)
    a = voxelized_mesh.points
    # print(a)
    # print(voxelized_mesh.points.shape)
    # print(voxelized_mesh.filled_count)
    voxelized_mesh.fill()
    b = voxelized_mesh.points
    # print(b)

    head_pcl = trimesh.PointCloud(vertices=b, colors = generate_random_color())
    scene.add_geometry(head_pcl)

# print(voxelized_mesh.points.shape)
# print(voxelized_mesh.filled_count)
# print(voxelized_mesh.points_to_indices(voxelized_mesh.points).shape)

# scene.add_geometry(voxelized_mesh.as_boxes())
origin = trimesh.PointCloud(vertices=[[0,0,0]], colors = [255,0,0])
scene.add_geometry(origin)

scene.show(smooth=False, flags={'wireframe': False})
