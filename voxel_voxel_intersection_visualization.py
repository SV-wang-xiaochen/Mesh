import open3d as o3d
import trimesh
import glob
import numpy as np
import matplotlib.pyplot as plt
import copy
import random


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

PITCH = 0.005

ref_vertex_index = 15
eye_ball_centroid = [0, 0, -1.30439425e-02] # Pre-calculated by averaging 53 EyeBallCentroid
lens_half_height_after_cut = 22
lens_init_centroid_z = 12   # currently max is 18 when scale is 1, because the lens may be outside the Voxel grid of head
lens_scale = 1 # When scale is 1, the diameter of the lens is around 54 mm

# # Define the semi-axes lengths of the ellipsoid lens
# lens_scale = 10
# lens_semi_x = 3 # eye left-right direction
# lens_semi_y = 3 # eye up-down direction
# lens_semi_z = 0.1 # eye front-rear direction
#
# # Define the rotation angles in degrees
# angle_x = 0  # Eye up-down Rotation around x-axis, + means down
# angle_y = 0  # Eye left-right Rotation around y-axis, + means left
#
# # Define lens centroid
# lens_init_centroid_x = 0
# lens_init_centroid_y = 0


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


path = r'C:\Users\xiaochen.wang\Projects\Dataset\FLORENCE'
obj_list = glob.glob(f'{path}/**/*.obj', recursive=True)

mesh_original = trimesh.load_mesh(obj_list[MESH_NR])
mesh_original.visual.face_colors = [64, 64, 64, 100]

scene = trimesh.Scene()

# plot the LeftEyeFront/LeftEyeRear/centroid point
eye_ball_key_points = trimesh.points.PointCloud(vertices=[mesh_original.vertices[LeftEyeFront], mesh_original.vertices[LeftEyeRear],
                                                eye_ball_centroid], colors=(255, 0, 0))

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

# #######################  Load lens mesh #######################

lens_path = r'C:\Users\xiaochen.wang\Projects\Dataset\lens_z12.obj'
lens_mesh_original = trimesh.load_mesh(lens_path)
lens_mesh_original.visual.face_colors = [64, 64, 64, 100]

if CUT_LENS:
    # cut the lens at the top and bottom
    lens_mesh = trimesh.intersections.slice_mesh_plane(trimesh.intersections.slice_mesh_plane(lens_mesh_original,
    [0,-1,0], [0,lens_half_height_after_cut/1000,0]),[0,1,0], [0,-lens_half_height_after_cut/1000,0])
else:
    lens_mesh = lens_mesh_original

# translate the lens between ref 12mm and lens_init_centroid_z
lens_mesh.apply_translation([0, 0, (lens_init_centroid_z-12)/1000])

# scale the lens
lens_mesh.apply_translation([0, 0, -lens_init_centroid_z/1000])
lens_mesh.apply_scale(lens_scale)
lens_mesh.apply_translation([0, 0, lens_init_centroid_z/1000])

# Translate the coordinates so that the centroid of eye ball becomes the origin
lens_mesh.apply_translation([-eye_ball_centroid[0], -eye_ball_centroid[1], -eye_ball_centroid[2]])

# Create 3d sphere, which is the region where centroid of the lens could be located
sphere_radius = lens_init_centroid_z/1000-eye_ball_centroid[2]

# Generate sphere points
x, y, z = ellipsoid(sphere_radius, sphere_radius, sphere_radius)

# Create a trimesh object from vertices and faces
vertices = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)
print(f"vertices:{len(vertices)}")
# Get the vertices which are within the 15 degree region
vertices_valid = []
angles = []
for sphere_point in vertices:
    # put only <= 15 degree here after debug
    if (angle_between_two_vectors([0,0,1], sphere_point) <= 15
            and angle_between_two_vectors([0,0,1], sphere_point) > 0
            and sphere_point[0]>=0 and sphere_point[2]>=0 and sphere_point[1]<=0):
        vertices_valid.append(sphere_point)
vertices_valid = np.array(vertices_valid)
print(vertices_valid.shape)

cloud = trimesh.PointCloud(vertices)
sphere_mesh = cloud.convex_hull
sphere_mesh.visual.face_colors = [255, 255, 0, 100]

# Rotate lens
ref_vertex = vertices[ref_vertex_index]
print(ref_vertex)
print(f"Before rotation, the angle in degree between ref vertex and lens centroid: {angle_between_two_vectors([0,0,1], ref_vertex)}\n")
R = rotation_matrix_from_vectors([0,0,1], ref_vertex)
Rotation = np.eye(4)
Rotation[:3, :3] = R

lens_mesh.apply_transform(Rotation)

# Translate the coordinates back so that the LeftEyeFront point becomes the origin
sphere_mesh.apply_translation([eye_ball_centroid[0], eye_ball_centroid[1], eye_ball_centroid[2]])
# scene.add_geometry(sphere_mesh)
lens_mesh.apply_translation([eye_ball_centroid[0], eye_ball_centroid[1], eye_ball_centroid[2]])

voxelized_lens = lens_mesh.voxelized(PITCH)

voxelized_lens.fill()

lens_voxelization = np.around(np.array(voxelized_lens.points),4)
lens_pcl = trimesh.PointCloud(vertices=np.array(lens_voxelization), colors=generate_random_color())

if not ONLY_SHOW_INTERSECTION:
    scene.add_geometry(lens_mesh)

# import time
# start = time.time()
# # check intersection between lens_mesh and mesh_original
# intersections = trimesh.boolean.intersection([lens_mesh, mesh_original], engine='blender')
# if hasattr(intersections, 'vertices'):
#     print("intersected")
#     intersections.visual.face_colors = [255, 0, 0, 100]
#     scene.add_geometry(intersections)
# else:
#     print("NOT intersected")
# end = time.time()
# print(end-start)

# plot the center of lens
lens_center = trimesh.points.PointCloud(vertices=[ref_vertex+eye_ball_centroid], colors=(255, 0, 0))

scene.add_geometry(lens_center)

# # Visualize the trimesh
# scene.show(smooth=False, flags={'wireframe': SHOW_WIREFRAME})

scene_voxel = trimesh.Scene()

# load prepared voxel
num_of_heads = np.load(f"voxel_results/num_of_heads_{PITCH}.npy")
voxel_list_remove_zero = np.load(f"voxel_results/voxel_list_remove_zero_{PITCH}.npy")
colors_list = np.load(f"voxel_results/colors_list_{PITCH}.npy")
accumulation_remove_zero = np.load(f"voxel_results/accumulation_remove_zero_{PITCH}.npy")

multi_heads = trimesh.PointCloud(vertices=voxel_list_remove_zero, colors=colors_list)

scene_voxel.add_geometry(eye_ball_key_points)
scene_voxel.add_geometry(lens_pcl)
scene_voxel.add_geometry(lens_center)
scene_voxel.add_geometry(mesh_original)

# scene_voxel.show(smooth=False, flags={'wireframe': SHOW_WIREFRAME})

voxel_list = voxel_list_remove_zero.tolist()
lens_list = lens_voxelization.tolist()

intersection_indices = []
for i, row in enumerate(voxel_list):
    if row in lens_list:
        intersection_indices.append(i)

intersection_voxels = voxel_list_remove_zero[intersection_indices]
intersection_colors_list = colors_list[intersection_indices]
head_hits = accumulation_remove_zero[intersection_indices]

print(f'max head_hits:{max(head_hits)}')

intersection_multi_heads = trimesh.PointCloud(vertices=intersection_voxels, colors=intersection_colors_list)

scene_voxel_intersection = trimesh.Scene()
scene_voxel_intersection.add_geometry(intersection_multi_heads)
scene_voxel_intersection.add_geometry(eye_ball_key_points)
scene_voxel_intersection.add_geometry(lens_center)
scene_voxel_intersection.add_geometry(mesh_original)

scene_voxel_intersection.show(smooth=False, flags={'wireframe': SHOW_WIREFRAME})




