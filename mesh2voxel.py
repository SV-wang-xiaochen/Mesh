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

num_of_heads = 53
path = r'C:\Users\xiaochen.wang\Projects\Dataset\FLORENCE'
obj_list = glob.glob(f'{path}/**/*.obj', recursive=True)
scene = trimesh.Scene()

PITCH = 0.0025
bound_min = [0, 0, 0]
bound_max = [0, 0, 0]

for mesh_nr in range(0, num_of_heads):
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

    # head_pcl = trimesh.PointCloud(vertices=np.array(voxelized_mesh.points), colors = generate_random_color())
    # scene.add_geometry(head_pcl)

voxel_center_min = bound_min + np.array([PITCH/2, PITCH/2, PITCH/2])
voxel_center_max = bound_max - np.array([PITCH/2, PITCH/2, PITCH/2])
print(voxel_center_min, voxel_center_max)

# # scene.add_geometry(voxelized_mesh.as_boxes())
# origin = trimesh.PointCloud(vertices=[[0,0,0]], colors = [255,0,0])
# scene.add_geometry(origin)

# Define the pitch
pitch = np.array([PITCH, PITCH, PITCH])

num_voxels = np.around(((voxel_center_max - voxel_center_min) / pitch) + 1).astype(int)

# Create a mesh grid for the x, y, z coordinates
x = np.around(np.linspace(voxel_center_min[0], voxel_center_max[0], num_voxels[0]),4)
y = np.around(np.linspace(voxel_center_min[1], voxel_center_max[1], num_voxels[1]),4)
z = np.around(np.linspace(voxel_center_min[2], voxel_center_max[2], num_voxels[2]),4)
# print(x,y,z)
# Generate the voxel grid using meshgrid
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Stack the grids to create the voxel grid
voxel_grid = np.stack((X, Y, Z), axis=-1)
voxel_list = voxel_grid.reshape(-1, 3).tolist()
print('voxel_list')
print(len(voxel_list))

accumulation = [0 for _ in range(len(voxel_list))]

for mesh_nr in range(0, num_of_heads):
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

    ####################### remove unrelated mesh #######################
    x1 = (head_mesh.vertices[Head1][0] + head_mesh.vertices[Head2][0] + head_mesh.vertices[Head3][0]) / 3
    y1 = head_mesh.vertices[MOUTH_ABOVE][1]
    y2 = head_mesh.vertices[BROW_ABOVE][1]
    z1 = head_mesh.vertices[LeftEyeRear][2]

    x1_voxel_limit = round(x1 / PITCH) * PITCH
    y1_voxel_limit = round(y1 / PITCH) * PITCH
    y2_voxel_limit = round(y2 / PITCH) * PITCH
    z1_voxel_limit = round(z1 / PITCH) * PITCH
    print(x1_voxel_limit,y1_voxel_limit,y2_voxel_limit,z1_voxel_limit)

    voxelized_mesh = head_mesh.voxelized(PITCH)

    voxelized_mesh.fill()

    head_voxelization = np.array(voxelized_mesh.points)

    condition = ((head_voxelization[:, 0] > x1_voxel_limit)&(head_voxelization[:, 1] > y1_voxel_limit)
                 &(head_voxelization[:, 1] < y2_voxel_limit)&(head_voxelization[:, 2] > z1_voxel_limit))

    head_voxelization = head_voxelization[condition]

    # head_pcl = trimesh.PointCloud(vertices=np.array(head_voxelization), colors=generate_random_color())
    # scene.add_geometry(head_pcl)

    head_occupancy = np.around(head_voxelization,4).tolist()

    # print(voxel_list)
    for v in head_occupancy:
        index = voxel_list.index(v)
        accumulation[index] = accumulation[index]+1

voxel_list_remove_zero = [voxel_list[i] for i in range(len(accumulation)) if accumulation[i] != 0]
accumulation_remove_zero = [accumulation[i] for i in range(len(accumulation)) if accumulation[i] != 0]

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Normalize the values in the list to range between 0 and 1
norm = mcolors.Normalize(vmin=0, vmax=num_of_heads)

# Create a scalar mappable using the Reds colormap
alpha_value = 1
scalar_map = plt.cm.ScalarMappable(cmap='Reds', norm=norm)

# Convert each value in the list to a color using the Reds colormap
colors_list = [scalar_map.to_rgba(val, alpha_value) for val in accumulation_remove_zero]
# print(colors_list)

# colors_list = list(map(lambda x:[round(x/num_of_heads*255),0,0], accumulation_remove_zero))
# colors_list = list(map(lambda x: [255,255,255] if x==[0,0,0] else x, colors_list))
print('colors')
print(len(colors_list))
# V = trimesh.PointCloud(vertices=[[0,0,0.01],[0,0,0.02]], colors=[[255,0,0],[0,255,0]])
V = trimesh.PointCloud(vertices=voxel_list_remove_zero, colors=colors_list)
# print(voxel_list)
# print(colors_list)
scene.add_geometry(V)
scene.show(smooth=False, flags={'wireframe': False})

np.save(f"num_of_heads_{PITCH}", num_of_heads)
np.save(f"voxel_center_min_{PITCH}", voxel_center_min)
np.save(f"voxel_center_max_{PITCH}", voxel_center_max)
np.save(f"voxel_list_remove_zero_{PITCH}", voxel_list_remove_zero)
np.save(f"colors_list_{PITCH}", colors_list)