import trimesh
import numpy as np
from utils import bottom_vertex_index, mouth_vertex_index
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
path = './voxel_results/FLORENCE'
obj_list = glob.glob(f'{path}/**/*.obj', recursive=True)
scene = trimesh.Scene()

PITCH = 0.0004
voxel_center_min = np.array([-0.0520, -0.1500, -0.0132])
voxel_center_max = np.array([0.0520, 0.0520, 0.0720])

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
# print('voxel_list')
# print(voxel_list)

z_step = round((voxel_center_max[2]-voxel_center_min[2])/PITCH+1)
y_step = round((voxel_center_max[1]-voxel_center_min[1])/PITCH+1)

def voxel2index(v):
    # set out-of-head-range indices as 0
    if not (voxel_center_min[0] < v[0] < voxel_center_max[0] and
            voxel_center_min[1] < v[1] < voxel_center_max[1] and
            voxel_center_min[2] < v[2] < voxel_center_max[2]):
        index = 0
    else:
        index = round((y_step * z_step * (v[0] - voxel_center_min[0]) + z_step * (v[1] - voxel_center_min[1]) + (
                v[2] - voxel_center_min[2])) / PITCH)
    return index

accumulation = [0 for _ in range(len(voxel_list))]
print('len(voxel_list)')
print(len(voxel_list))
for mesh_nr in range(0, num_of_heads):
    print(f'Mesh number:{mesh_nr}')
    head_mesh = trimesh.load_mesh(obj_list[mesh_nr])

    # seal the bottom
    bottom_vertex = head_mesh.vertices[bottom_vertex_index]
    hole_boundary_vertices = bottom_vertex

    bottom_vertices_np = np.array(hole_boundary_vertices)
    bottom_vertices_np_swap = np.empty(bottom_vertices_np.shape)
    bottom_vertices_np_swap[:, [0, 1, 2]] = bottom_vertices_np[:, [0, 2, 1]]

    bottom_vertices_selected = bottom_vertices_np_swap[:,:2]

    from scipy.spatial import Delaunay
    tri = Delaunay(bottom_vertices_selected)

    bottom_faces = np.array(tri.simplices)
    bottom_converted_faces = np.array([bottom_vertex_index[val] for row in bottom_faces for val in row]).reshape(bottom_faces.shape)

    # Add the new faces to the mesh
    head_mesh.faces = np.append(head_mesh.faces, bottom_converted_faces, axis=0)

    # seal the mouth
    mouth_vertex = head_mesh.vertices[mouth_vertex_index]
    hole_boundary_vertices = mouth_vertex

    mouth_vertices_np = np.array(hole_boundary_vertices)
    mouth_vertices_selected = mouth_vertices_np[:,:2]

    tri = Delaunay(mouth_vertices_selected)

    mouth_faces = np.array(tri.simplices)
    mouth_converted_faces = np.array([mouth_vertex_index[val] for row in mouth_faces for val in row]).reshape(mouth_faces.shape)

    # Add the new faces to the mesh
    head_mesh.faces = np.append(head_mesh.faces, mouth_converted_faces, axis=0)

    ####################### remove unrelated voxel #######################
    x1_voxel_limit = voxel_center_min[0]
    x2_voxel_limit = voxel_center_max[0]
    y1_voxel_limit = voxel_center_min[1]
    y2_voxel_limit = voxel_center_max[1]
    z1_voxel_limit = voxel_center_min[2]
    z2_voxel_limit = voxel_center_max[2]

    voxelized_mesh = head_mesh.voxelized(PITCH)

    voxelized_mesh.fill()

    head_voxelization = np.array(voxelized_mesh.points)

    condition = ((head_voxelization[:, 0] > x1_voxel_limit)&(head_voxelization[:, 0] < x2_voxel_limit)&
                 (head_voxelization[:, 1] > y1_voxel_limit)&(head_voxelization[:, 1] < y2_voxel_limit)&
                 (head_voxelization[:, 2] > z1_voxel_limit)&(head_voxelization[:, 2] < z2_voxel_limit))

    head_voxelization = head_voxelization[condition]

    # head_pcl = trimesh.PointCloud(vertices=np.array(head_voxelization), colors=generate_random_color())
    # scene.add_geometry(head_pcl)

    head_occupancy = np.around(head_voxelization,4).tolist()

    head_occupancy_index_list = list(filter(lambda x: x != 0, list(map(voxel2index, head_occupancy))))
    print('head_occupancy_index_list')
    print(len(head_occupancy_index_list))

    accumulation_np = np.array(accumulation)
    head_occupancy_index_list_np = np.array(head_occupancy_index_list)

    accumulation_np[head_occupancy_index_list_np] = accumulation_np[head_occupancy_index_list_np]+1

    accumulation = list(accumulation_np)

accumulation_np = np.array(accumulation)
nonzero_accumulation = np.nonzero(accumulation_np)
voxel_list_remove_zero = list(np.array(voxel_list)[nonzero_accumulation])
accumulation_remove_zero = list(accumulation_np[nonzero_accumulation])

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Normalize the values in the list to range between 0 and 1
norm = mcolors.Normalize(vmin=0, vmax=num_of_heads)

# Create a scalar mappable using the Reds colormap
alpha_value = 1
scalar_map = plt.cm.ScalarMappable(cmap='Reds', norm=norm)

# Convert each value in the list to a color using the colormap
colors_list = scalar_map.to_rgba(np.array(accumulation_remove_zero), alpha_value)

print('colors')
print(len(colors_list))

np.save(f"num_of_heads_{PITCH}", num_of_heads)
np.save(f"voxel_center_min_{PITCH}", voxel_center_min)
np.save(f"voxel_center_max_{PITCH}", voxel_center_max)
np.save(f"voxel_list_remove_zero_{PITCH}", voxel_list_remove_zero)
np.save(f"colors_list_{PITCH}", colors_list)
np.save(f"accumulation_remove_zero_{PITCH}", accumulation_remove_zero)

V = trimesh.PointCloud(vertices=voxel_list_remove_zero, colors=colors_list)

head_voxel_list = voxel_list_remove_zero
head_voxel_indices = list(filter(lambda x: x != 0, list(map(voxel2index, head_voxel_list))))

np.save(f"head_voxel_indices_{PITCH}", head_voxel_indices)

scene.add_geometry(V)
scene.show(smooth=False, flags={'wireframe': False})
