import open3d as o3d
import trimesh
import glob
import numpy as np
import math


SHOW_WIREFRAME = False

MESH_NR = 0
LeftEyeFront = 4043
LeftEyeRear = 4463

eye_ball_centroid = [0, 0, -1.30439425e-02] # Pre-calculated by averaging 53 EyeBallCentroid

# Define the semi-axes lengths of the ellipsoid lens
lens_scale = 10
lens_semi_x = 3 # eye left-right direction
lens_semi_y = 3 # eye up-down direction
lens_semi_z = 0.1 # eye front-rear direction

# Define the rotation angles in degrees
angle_x = 0  # Eye up-down Rotation around x-axis, + means down
angle_y = 0  # Eye left-right Rotation around y-axis, + means left

# Define lens centroid
lens_init_centroid_x = 0
lens_init_centroid_y = 0
lens_init_centroid_z = 12

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
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
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

scene.add_geometry(mesh_original)

# plot the LeftEyeFront/LeftEyeRear/centroid point
eye_ball_key_points = trimesh.points.PointCloud(vertices=[mesh_original.vertices[LeftEyeFront], mesh_original.vertices[LeftEyeRear],
                                                eye_ball_centroid], colors=(255, 0, 0))

scene.add_geometry(eye_ball_key_points)

# #######################  Create ellipsoid lens (this part will be replaced by loading a pre-defined lens mesh) #######################

# Generate ellipsoid points
x, y, z = ellipsoid(lens_semi_x*lens_scale/1000, lens_semi_y*lens_scale/1000, lens_semi_z*lens_scale/1000)

# Create a trimesh object from vertices and faces
vertices = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)
cloud = trimesh.PointCloud(vertices)
ellipsoid_mesh = cloud.convex_hull
ellipsoid_mesh.visual.face_colors = [0, 0, 255, 100]

# Convert degrees to radians
theta = math.radians(angle_x)
phi = math.radians(angle_y)

# Define the rotation matrices in 3D
Rx = np.array([[1, 0, 0],
               [0, math.cos(theta), -math.sin(theta)],
               [0, math.sin(theta), math.cos(theta)]])

Ry = np.array([[math.cos(phi), 0, math.sin(phi)],
               [0, 1, 0],
               [-math.sin(phi), 0, math.cos(phi)]])

# Create 4x4 transformation matrices (homogeneous transformation matrices)
# For rotations around x and y axes
homogeneous_Rx = np.eye(4)
homogeneous_Rx[:3, :3] = Rx

homogeneous_Ry = np.eye(4)
homogeneous_Ry[:3, :3] = Ry

# Create the translation matrix for the initial lens centroid
translation = np.array([lens_init_centroid_x/1000, lens_init_centroid_y/1000,
                        lens_init_centroid_z/1000])
homogeneous_translation = np.eye(4)
homogeneous_translation[:3, 3] = translation

# Combine the transformations (order matters)
# Here, we perform rotation about y first, then about x
combined_transform = np.dot(homogeneous_translation, np.dot(homogeneous_Rx, homogeneous_Ry))

# Apply the rotation to the mesh vertices
ellipsoid_mesh.apply_transform(combined_transform)
# #######################  Create ellipsoid lens (this part will be replaced by loading a pre-defined lens mesh) #######################

#######################  Translate the coordinates so that the centroid of eye ball becomes the origin. #######################
ellipsoid_mesh.apply_translation([-eye_ball_centroid[0], -eye_ball_centroid[1], -eye_ball_centroid[2]])

############### do 15 degree rotation here ################

# #######################  Create 3d sphere, which is the region where centroid of the lens could be located #######################

sphere_radius = lens_init_centroid_z/1000-eye_ball_centroid[2]

# Generate sphere points
x, y, z = ellipsoid(sphere_radius, sphere_radius, sphere_radius)

# Create a trimesh object from vertices and faces
vertices = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)

# Get the vertices which are within the 15 degree region
vertices_valid = []
angles = []
for sphere_point in vertices:
    # put only <= 15 degree here after debug
    if (angle_between_two_vectors([0,0,1], sphere_point) <= 45
            and angle_between_two_vectors([0,0,1], sphere_point) >= 30
            and sphere_point[0]>=0 and sphere_point[2]>=0 and sphere_point[1]<=0):
        vertices_valid.append(sphere_point)
vertices_valid = np.array(vertices_valid)
print(vertices_valid.shape)

cloud = trimesh.PointCloud(vertices)
sphere_mesh = cloud.convex_hull
sphere_mesh.visual.face_colors = [255, 255, 0, 100]

# Rotate lens
ref_vertex = vertices_valid[0]
print(f"Before rotation, the angle in degree between ref vertex and lens centroid: {angle_between_two_vectors([0,0,1], ref_vertex)}\n")
R = rotation_matrix_from_vectors([0,0,1], vertices_valid[0])
Rotation = np.eye(4)
Rotation[:3, :3] = R

ellipsoid_mesh.apply_transform(Rotation)

#######################  Translate the coordinates back so that the LeftEyeFront point becomes the origin. #######################
sphere_mesh.apply_translation([eye_ball_centroid[0], eye_ball_centroid[1], eye_ball_centroid[2]])
scene.add_geometry(sphere_mesh)
ellipsoid_mesh.apply_translation([eye_ball_centroid[0], eye_ball_centroid[1], eye_ball_centroid[2]])
scene.add_geometry(ellipsoid_mesh)

# Visualize the trimesh
scene.show(smooth=False, flags={'wireframe': SHOW_WIREFRAME})

# #######################  Convert trimesh to open3d mesh #######################
# ellipsoid_mesh_o3d = trimesh2open3d(ellipsoid_mesh)
# mesh_original_o3d = trimesh2open3d(mesh_original)
# ellipsoid_pcl_o3d = o3dTriangleMesh2PointCloud(ellipsoid_mesh_o3d)
# mesh_pcl_o3d = o3dTriangleMesh2PointCloud(mesh_original_o3d)
#
# distances = np.array(ellipsoid_pcl_o3d.compute_point_cloud_distance(mesh_pcl_o3d))
# print(np.max(distances), np.min(distances))
#
# # Visualize the Open3D mesh
# o3d.visualization.draw_geometries([mesh_original_o3d, ellipsoid_mesh_o3d], mesh_show_wireframe=True)
