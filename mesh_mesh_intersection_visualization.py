import trimesh
import glob
import numpy as np
import copy

MESH_NR = 0
LeftEyeRear = 4463

def line_segment_with_circle(line_segment, circle_origin, circle_radius):
    """ Check if a line segment is within a circle. Here the line segment must be within the plane where the circle is within.
        Here we define that if any point of the line segment is within the circle, then True is returned.
    :param line_segment: line segment [start_point, end_point]
    :param circle_origin: origin of circle
    :param circle_radius: radius of circle
    :return isWithinCircle: Boolean
    """

    distance1 = np.linalg.norm(line_segment[0] - circle_origin)
    distance2 = np.linalg.norm(line_segment[1] - circle_origin)

    # Check if the point is within the circle
    if distance1 < circle_radius or distance2 < circle_radius:
        return True
    else:
        return False

path = r'C:\Users\xiaochen.wang\Projects\Dataset\FLORENCE'
obj_list = glob.glob(f'{path}/**/*.obj', recursive=True)

mesh_original = trimesh.load_mesh(obj_list[MESH_NR])

scene = trimesh.Scene()

scene.add_geometry(mesh_original)

# plot the LeftEyeFront point
origin = trimesh.points.PointCloud(vertices=[[0, 0, 0]], colors=(255, 0, 0))
scene.add_geometry(origin)

# #######################  Create ellipsoid lens #######################

def ellipsoid(a, b, c):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    x = a * np.outer(np.cos(u), np.sin(v))
    y = b * np.outer(np.sin(u), np.sin(v))
    z = c * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z

# Define the semi-axes lengths of the ellipsoid
lens_scale = 10
a = 5 # eye left-right direction
b = 3 # eye up-down direction
c = 1 # eye front-rear direction

# Generate ellipsoid points
x, y, z = ellipsoid(a*lens_scale/1000, b*lens_scale/1000, c*lens_scale/1000)

# Create a trimesh object from vertices and faces
vertices = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)
cloud = trimesh.PointCloud(vertices)
ellipsoid_mesh = cloud.convex_hull
ellipsoid_mesh.visual.face_colors = [0, 0, 255, 100]

import math

# Define the rotation angles in degrees
angle_x = 0  # Rotation around x-axis, eye up-down direction, + means down
angle_y = 15  # Rotation around y-axis, eye left-right direction, + means left

# Define translation values
trans_x = 0
trans_y = 0
trans_z = 45

# plot the lens center point
origin = trimesh.points.PointCloud(vertices=[[trans_x/1000, trans_y/1000, trans_z/1000]], colors=(0, 255, 0))
scene.add_geometry(origin)

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

# Create the translation matrix
translation = np.array([trans_x/1000, trans_y/1000, trans_z/1000])
homogeneous_translation = np.eye(4)
homogeneous_translation[:3, 3] = translation

# Combine the transformations (order matters)
# Here, we perform rotation about y first, then about x
combined_transform = np.dot(homogeneous_translation, np.dot(homogeneous_Rx, homogeneous_Ry))

# Apply the rotation to the mesh vertices
ellipsoid_mesh.apply_transform(combined_transform)

scene.add_geometry(ellipsoid_mesh)

scene.show(smooth=False)