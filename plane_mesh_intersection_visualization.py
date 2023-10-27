import trimesh
import glob
import numpy as np
import copy
import os

Z_LENS = 12 # Z distance from Lens plane to LeftEyeFront point. Range: 3-30 mm
RADIUS = 40 # Radius of lens
MESH_NR = 0 # Set which mesh to load

NOT_SHOW_MESH = False # Do not show any mesh. Only show the intersection and lens
SHOW_FULL_MESH = True # Show full mesh or ONLY "mesh of interest" which is used to find the intersection with lens plane

LeftEyeRear = 4463
Head1 = 1726
Head2 = 1335
Head3 = 1203
MOUTH_ABOVE = 825
BROW_ABOVE = 2295

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

scene = trimesh.Scene()

if not NOT_SHOW_MESH:
    if SHOW_FULL_MESH:
        scene.add_geometry(mesh_original)
    else:
        scene.add_geometry(mesh)

####################### draw the max lens (a circle with diameter 80mm) #######################

# Define the parameters
origin_lens = np.array([0, 0, Z_LENS/1000])
radius = RADIUS/1000
num_points = 100  # Number of points to create the circle

# Create an array of angles to parametrically define the circle
theta = np.linspace(0, 2 * np.pi, num_points)

# Parametric equations to generate points on the circle
x = radius * np.cos(theta) + origin_lens[0]
y = radius * np.sin(theta) + origin_lens[1]
z = np.full(num_points, origin_lens[2])  # Constant z-coordinate

# Create an array to store the line segments
circle_segments = []

# Connect the points to create line segments
for i in range(num_points - 1):
    circle_segments.append([[x[i], y[i], z[i]], [x[i + 1], y[i + 1], z[i + 1]]])

# Connect the last point to the first point to close the circle
circle_segments.append([[x[-1], y[-1], z[-1]], [x[0], y[0], z[0]]])

# Convert the line segments to a NumPy array
circle_segments = np.array(circle_segments)

circle = trimesh.load_path(circle_segments)
scene.add_geometry(circle)

# plot the LeftEyeFront point
origin = trimesh.points.PointCloud(vertices=[[0, 0, 0]], colors=(255, 0, 0))
scene.add_geometry(origin)

####################### find intersection between lens plane and mesh #######################

# Define a plane as "a point on the plane and its normal"
point_on_plane = [0, 0, Z_LENS/1000]
plane_normal = [0, 0, 1]

# Find the intersection between the mesh and the plane
intersection_segments = trimesh.intersections.mesh_plane(mesh, plane_normal, point_on_plane)

#######################  Only keep line segments within the lens #######################
intersection_segments_within_circle = []
for segment in intersection_segments:
    if line_segment_with_circle(segment, origin_lens, radius):
        intersection_segments_within_circle.append(segment)

if len(intersection_segments_within_circle)>0:
    intersection_segments_within_circle = trimesh.load_path(np.array(intersection_segments_within_circle))
    scene.add_geometry(intersection_segments_within_circle)

scene.show(smooth=False)