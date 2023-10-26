import trimesh
import glob
import numpy as np
import copy

Z_LENS = 12 # Z distance from Lens plane to LeftEyeFront point. Range: 3-30 mm

LeftEyeRear = 4463
Head1 = 1726
Head2 = 1335
Head3 = 1203
MOUTH_ABOVE = 825
BROW_ABOVE = 2295

path = r'C:\Users\xiaochen.wang\Projects\Dataset\FLORENCE'
obj_list = glob.glob(f'{path}/**/*.obj', recursive=True)

mesh_original = trimesh.load_mesh(obj_list[0])

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
scene.add_geometry(mesh_original)

####################### find intersection between lens plane and mesh #######################

# Define a plane as "a point on the plane and its normal"
point_on_plane = [0, 0, Z_LENS/1000]
plane_normal = [0, 0, 1]

# Find the intersection between the mesh and the plane
lines = trimesh.intersections.mesh_plane(mesh, plane_normal, point_on_plane)
print(len(lines))

intersection = trimesh.load_path(lines)
scene.add_geometry(intersection)

####################### draw the max lens (a circle with diameter 80mm) #######################

# Define the parameters
origin = np.array([0, 0, Z_LENS/1000])
radius = 0.04
num_points = 100  # Number of points to create the circle

# Create an array of angles to parametrically define the circle
theta = np.linspace(0, 2 * np.pi, num_points)

# Parametric equations to generate points on the circle
x = radius * np.cos(theta) + origin[0]
y = radius * np.sin(theta) + origin[1]
z = np.full(num_points, origin[2])  # Constant z-coordinate

# Create an array to store the line segments
line_segments = []

# Connect the points to create line segments
for i in range(num_points - 1):
    line_segments.append([[x[i], y[i], z[i]], [x[i + 1], y[i + 1], z[i + 1]]])

# Connect the last point to the first point to close the circle
line_segments.append([[x[-1], y[-1], z[-1]], [x[0], y[0], z[0]]])

# Convert the line segments to a NumPy array
line_segments = np.array(line_segments)

circle = trimesh.load_path(line_segments)
scene.add_geometry(circle)

# plot the LeftEyeFront point
origin = trimesh.points.PointCloud(vertices=[[0, 0, 0]], colors=(255, 0, 0))
scene.add_geometry(origin)

scene.show(smooth=False)

