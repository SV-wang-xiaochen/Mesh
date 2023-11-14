import open3d as o3d
import trimesh
import glob
import numpy as np
# import matplotlib.pyplot as plt
import copy
import random
import math


SHOW_WIREFRAME = False
ONLY_SHOW_INTERSECTION = False
NOT_SHOW_MESH = False # Do not show any mesh. Only show the intersection and lens
SHOW_FULL_MESH = True # Show full mesh or ONLY "mesh of interest" which is used to find the intersection with lens plane

MESH_NR = 0
LeftEyeFront = 4043
LeftEyeRear = 4463
Head1 = 1726
Head2 = 1335
Head3 = 1203
MOUTH_ABOVE = 825
BROW_ABOVE = 2295
CUT_LENS = False
INTERACTIVE_INPUT = False
CORNEA_IGNORE = 6 # size of cornea region to be ignored. size = CORNEA*PITH

eye_ball_shift = [0, 0, -1.30439425e-02] # Pre-calculated by averaging 53 EyeBallCentroid
lens_half_height_after_cut = 22

PITCH = float(input('Size of voxel, e.g. 0.001 means 1mm. Currently, only 0.0005, 0.001, 0.0025, 0.005 allowed:')) if INTERACTIVE_INPUT else 0.0005
working_distance = float(input('Default working distance of lens, e.g. 12 means 12mm. Range [0,50] mm:')) if INTERACTIVE_INPUT else 12

# When scale is 1, the diameter of the lens is around 54.8 mm
lens_diameter = float(input('Lens diameter, e.g. 50 means 50mm. Range [20, 80] mm:')) if INTERACTIVE_INPUT else 50

# Define the lens rotation by Spherical coordinate system: https://en.wikipedia.org/wiki/Spherical_coordinate_system
theta = float(input('Rotation angle, theta. Range[0,15] degrees:'))  if INTERACTIVE_INPUT else 0
phi = float(input('Rotation direction, phi. Range[0,90] degrees, 0 means pure left, 90 means pure down:')) if INTERACTIVE_INPUT else 0
print('\n')

def intersection_elements(a, b):
    mask = np.isin(b, a).all(axis=1)
    c = b[mask]
    indices = np.where(mask)[0]
    return c, indices


def generate_random_color():
    # Generate random values for red, green, and blue components
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)

    # Create the color list in the specified format [r, g, b, 100]
    color = [r, g, b, 100]
    return color


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if any(v):  # if not all zeros then
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    else:
        return np.eye(3)  # cross of all zeros only occurs on identical directions


def angle_between_two_vectors(vec1, vec2):
    """ Calculate the angle between vec1 to vec2
    :param vec1, vec2: two 3d vectors
    :return angle: angle in degrees
    """

    # Calculate the dot product
    dot_product = np.dot(vec1, vec2)

    # Calculate the magnitudes of both vectors
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)

    # Calculate the angle in radians
    cosine_theta = dot_product / (magnitude1 * magnitude2)
    theta = np.arccos(cosine_theta)

    # Convert the angle to degrees
    angle = np.degrees(theta)

    return angle


def ellipsoid(a, b, c):
    u = np.linspace(0, 2 * np.pi, 75)
    v = np.linspace(0, np.pi, 75)
    x = a * np.outer(np.cos(u), np.sin(v))
    y = b * np.outer(np.sin(u), np.sin(v))
    z = c * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z


def trimesh2open3d(mesh):
    vertices = mesh.vertices
    faces = mesh.faces

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

    return o3d_mesh


def o3dTriangleMesh2PointCloud(mesh):
    # Extract vertices from the TriangleMesh
    vertices = mesh.vertices

    # Create a new PointCloud using the extracted vertices
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(vertices)

    return point_cloud


path = './voxel_results/FLORENCE'
obj_list = glob.glob(f'{path}/**/*.obj', recursive=True)

mesh_original = trimesh.load_mesh(obj_list[MESH_NR])
mesh_original.visual.face_colors = [64, 64, 64, 100]

scene = trimesh.Scene()

# plot the LeftEyeFront/LeftEyeRear/centroid point
eye_ball_key_points = trimesh.points.PointCloud(vertices=[mesh_original.vertices[LeftEyeFront], mesh_original.vertices[LeftEyeRear],
                                                [0,0,0]], colors=(0, 255, 0))

scene.add_geometry(eye_ball_key_points)

mesh = copy.deepcopy(mesh_original)

####################### remove unrelated mesh #######################
x1 = (mesh.vertices[Head1][0] + mesh.vertices[Head2][0] + mesh.vertices[Head3][0])/3
y1 = mesh.vertices[MOUTH_ABOVE][1]
y2 = mesh.vertices[BROW_ABOVE][1]
z1 = mesh.vertices[LeftEyeRear][2]

vertices = mesh.vertices
vertices_mask = vertices[:,0] > x1
face_mask = vertices_mask[mesh.faces].all(axis=1)
mesh.update_faces(face_mask)

vertices = mesh.vertices
vertices_mask = vertices[:,1] < y2
face_mask = vertices_mask[mesh.faces].all(axis=1)
mesh.update_faces(face_mask)

vertices = mesh.vertices
vertices_mask = vertices[:,1] > y1
face_mask = vertices_mask[mesh.faces].all(axis=1)
mesh.update_faces(face_mask)

vertices = mesh.vertices
vertices_mask = vertices[:,2] > z1
face_mask = vertices_mask[mesh.faces].all(axis=1)
mesh.update_faces(face_mask)

if not NOT_SHOW_MESH:
    if SHOW_FULL_MESH:
        scene.add_geometry(mesh_original)
    else:
        scene.add_geometry(mesh)

# #######################  create cone lens  #######################

cone_lens = trimesh.creation.cone(lens_diameter/2000, -working_distance/1000)
cone_lens.apply_translation([0, 0, working_distance/1000-eye_ball_shift[2]])

cone_lens_center = trimesh.points.PointCloud(vertices=[[0, 0, working_distance/1000-eye_ball_shift[2]]], colors=(0, 255, 0))

x_angle = 8 / 180 * np.pi
x_Rotation = np.eye(4)
x_R = np.array([[1, 0, 0],
                [0, np.cos(x_angle), -np.sin(x_angle)],
                [0, np.sin(x_angle), np.cos(x_angle)]])
x_Rotation[:3, :3] = x_R

cone_lens.apply_transform(x_Rotation)
cone_lens_center.apply_transform(x_Rotation)

# Convert Spherical coordinates to Cartesian coordinates
x = math.sin(theta/180*np.pi)*math.cos(phi/180*np.pi)
y = -math.sin(theta/180*np.pi)*math.sin(phi/180*np.pi)
z = math.cos(theta/180*np.pi)

# Calculate the rotation matrix between initial direction vector [0,0,1) and (x,y,z)
R = rotation_matrix_from_vectors((0,0,1), (x,y,z))
Rotation = np.eye(4)
Rotation[:3, :3] = R

# Rotate the lens
cone_lens.apply_transform(Rotation)
cone_lens_center.apply_transform(Rotation)
scene.add_geometry(cone_lens)
scene.add_geometry(cone_lens_center)

voxelized_cone_lens = cone_lens.voxelized(PITCH).fill()
lens_voxelization = np.around(np.array(voxelized_cone_lens.points),4)
lens_pcl = trimesh.PointCloud(vertices=np.array(lens_voxelization), colors=[0, 0, 255, 100])

# scene.add_geometry(lens_pcl)

# Create 3d sphere, which is the region where centroid of the lens could be located
sphere_radius = working_distance/1000-eye_ball_shift[2]

# Generate sphere points
x, y, z = ellipsoid(sphere_radius, sphere_radius, sphere_radius)

# Create a trimesh object from vertices and faces
vertices = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)

cloud = trimesh.PointCloud(vertices)
sphere_mesh = cloud.convex_hull
sphere_mesh.visual.face_colors = [255, 255, 0, 100]

# scene.add_geometry(sphere_mesh)

for mesh_nr in range(0, len(obj_list)):
    mesh_original = trimesh.load_mesh(obj_list[mesh_nr])
    mesh_original.visual.face_colors = [64, 64, 64, 50]
    scene.add_geometry(mesh_original)

# #######################  create a cuboid around cornea center where the head hits should be ignored  #######################
cuboid = trimesh.creation.box(extents=[CORNEA_IGNORE*PITCH,CORNEA_IGNORE*PITCH,CORNEA_IGNORE*PITCH])
cuboid.apply_translation([0, 0, -eye_ball_shift[2]])
cuboid.apply_transform(x_Rotation)
cuboid.apply_transform(Rotation)
# scene.add_geometry(cuboid)

voxelized_cuboid = cuboid.voxelized(PITCH).fill()
cuboid_voxelization = np.around(np.array(voxelized_cuboid.points),4)
cuboid_pcl = trimesh.PointCloud(vertices=np.array(cuboid_voxelization), colors=[255, 0, 255, 100])
# scene.add_geometry(cuboid_pcl)

# Visualize the trimesh
scene.show(smooth=False, flags={'wireframe': SHOW_WIREFRAME})

scene_voxel = trimesh.Scene()

# load prepared voxel
num_of_heads = np.load(f"voxel_results/num_of_heads_{PITCH}.npy")
voxel_list_remove_zero = np.load(f"voxel_results/voxel_list_remove_zero_{PITCH}.npy")
colors_list = np.load(f"voxel_results/colors_list_{PITCH}.npy")
accumulation_remove_zero = np.load(f"voxel_results/accumulation_remove_zero_{PITCH}.npy")
voxel_center_min = np.load(f"voxel_results/voxel_center_min_{PITCH}.npy")
voxel_center_max = np.load(f"voxel_results/voxel_center_max_{PITCH}.npy")

multi_heads = trimesh.PointCloud(vertices=voxel_list_remove_zero, colors=colors_list)

scene_voxel.add_geometry(eye_ball_key_points)
scene_voxel.add_geometry(lens_pcl)
scene_voxel.add_geometry(multi_heads)

scene_voxel.show(smooth=False, flags={'wireframe': SHOW_WIREFRAME})

z_step = round((voxel_center_max[2]-voxel_center_min[2])/PITCH+1)
y_step = round((voxel_center_max[1]-voxel_center_min[1])/PITCH+1)

def voxel2index(v):
    return round((y_step*z_step*(v[0]-voxel_center_min[0])+z_step*(v[1]-voxel_center_min[1])+(v[2]-voxel_center_min[2]))/PITCH)

lens_list = lens_voxelization.tolist()
lens_indices = list(map(voxel2index, lens_list))

cuboid_list = cuboid_voxelization.tolist()
cuboid_voxel_indices = list(map(voxel2index, cuboid_list))

voxel_list = voxel_list_remove_zero.tolist()
head_voxel_indices = list(map(voxel2index, voxel_list))

valid_lens_indices = list(np.setdiff1d(np.array(lens_indices), np.array(cuboid_voxel_indices), True))
_, intersection_indices, _ = np.intersect1d(np.array(head_voxel_indices), np.array(valid_lens_indices), return_indices=True)

intersection_voxels = voxel_list_remove_zero[intersection_indices]
intersection_colors_list = colors_list[intersection_indices]
head_hits = accumulation_remove_zero[intersection_indices]

print(f'Size of voxel:{PITCH} mm')
print(f'Lens working distance:{working_distance} mm')
print(f'Lens diameter:{lens_diameter} mm')
print(f'Rotation angle, theta:{theta} degrees')
print(f'Rotation direction, phi (left-down):{phi} degrees')

if len(head_hits) > 0:
    print(f'max head_hits:{max(head_hits)}')
    intersection_multi_heads = trimesh.PointCloud(vertices=intersection_voxels, colors=intersection_colors_list)

    scene_voxel_intersection = trimesh.Scene()
    scene_voxel_intersection.add_geometry(intersection_multi_heads)
    scene_voxel_intersection.add_geometry(eye_ball_key_points)
    scene_voxel_intersection.add_geometry(cone_lens_center)

    path = r'C:\Users\xiaochen.wang\Projects\Dataset\FLORENCE'
    obj_list = glob.glob(f'{path}/**/*.obj', recursive=True)

    for mesh_nr in range(0, len(obj_list)):
        mesh_original = trimesh.load_mesh(obj_list[mesh_nr])
        mesh_original.visual.face_colors = [64, 64, 64, 50]
        scene_voxel_intersection.add_geometry(mesh_original)
    scene_voxel_intersection.show(smooth=False, flags={'wireframe': SHOW_WIREFRAME})
else:
    print('NO intersection')
