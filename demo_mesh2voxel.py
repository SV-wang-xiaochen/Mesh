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
LeftEyeFront = 4043
LeftEyeRear = 4463
Head1 = 1726
Head2 = 1335
Head3 = 1203
MOUTH_ABOVE = 825
BROW_ABOVE = 2295

num_of_heads = 1
path = r'C:\Users\xiaochen.wang\Projects\Dataset\FLORENCE'
obj_list = glob.glob(f'{path}/**/*.obj', recursive=True)
scene = trimesh.Scene()

PITCH = 0.01
bound_min = [0, 0, 0]
bound_max = [0, 0, 0]

for mesh_nr in range(0, num_of_heads):
    print('Calculating Bound')
    print(f'Mesh number:{mesh_nr}')
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

    voxelized_mesh = head_mesh.voxelized(PITCH)

    voxelized_mesh.fill()

    bound_min = np.minimum(bound_min, voxelized_mesh.bounds[0])
    bound_max = np.maximum(bound_max, voxelized_mesh.bounds[1])

    head_pcl = trimesh.PointCloud(vertices=np.array(voxelized_mesh.points), colors = generate_random_color())
    scene.add_geometry(head_pcl)

scene.show(smooth=False, flags={'wireframe': False})
