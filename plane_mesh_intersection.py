import trimesh
import glob

Z_LENS = 12 # Z distance from Lens plane to LeftEyeFront point. Range: 3-30 mm

LeftEyeRear = 4463
Head1 = 1726
Head2 = 1335
Head3 = 1203
MOUTH_ABOVE = 825
BROW_ABOVE = 2295

path = r'C:\Users\xiaochen.wang\Projects\Dataset\FLORENCE'
obj_list = glob.glob(f'{path}/**/*.obj', recursive=True)

mesh = trimesh.load_mesh(obj_list[38])

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

####################################################################

scene = trimesh.Scene()
scene.add_geometry(mesh)

# Define a plane as "a point on the plane and its normal"
point_on_plane = [0, 0, Z_LENS/1000]
plane_normal = [0, 0, 1]

# Find the intersection between the mesh and the plane
lines = trimesh.intersections.mesh_plane(mesh, plane_normal, point_on_plane) # lines is np.ndarry
print(len(lines))

p = trimesh.load_path(lines)

scene.add_geometry(p)
scene.show(smooth=False)