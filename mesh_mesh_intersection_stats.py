import open3d as o3d
import trimesh
import glob
import numpy as np
import matplotlib.pyplot as plt
import copy

SHOW_WIREFRAME = False
ONLY_SHOW_INTERSECTION = False

MESH_NR = 0
LeftEyeFront = 4043
LeftEyeRear = 4463

ref_vertex_index = 0
eye_ball_centroid = [0, 0, -1.30439425e-02] # Pre-calculated by averaging 53 EyeBallCentroid
lens_half_height_after_cut = 22
lens_init_centroid_z = 12
lens_scale_list = [1] # When scale is 1, the diameter of the lens is around 54 mm

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

# #######################  Load lens mesh #######################
for lens_scale in lens_scale_list:

    lens_path = r'C:\Users\xiaochen.wang\Projects\Dataset\lens_z12.obj'
    lens_mesh_original = trimesh.load_mesh(lens_path)
    lens_mesh_original.visual.face_colors = [64, 64, 64, 100]

    # cut the lens at the top and bottom
    lens_mesh = trimesh.intersections.slice_mesh_plane(trimesh.intersections.slice_mesh_plane(lens_mesh_original,
    [0,-1,0], [0,lens_half_height_after_cut/1000,0]),[0,1,0], [0,-lens_half_height_after_cut/1000,0])

    # scale the lens
    lens_mesh.apply_translation([0, 0, -lens_init_centroid_z/1000])
    lens_mesh.apply_scale(lens_scale)
    lens_mesh.apply_translation([0, 0, lens_init_centroid_z/1000])

    # Translate the coordinates so that the centroid of eye ball becomes the origin
    lens_mesh.apply_translation([-eye_ball_centroid[0], -eye_ball_centroid[1], -eye_ball_centroid[2]])

    path = r'C:\Users\xiaochen.wang\Projects\Dataset\FLORENCE'
    obj_list = glob.glob(f'{path}/**/*.obj', recursive=True)

    head_hits = np.zeros(5000) # this value only works when len(vertices_valid) <= 5000

    for mesh_nr in range(0, len(obj_list)):

        mesh_original = trimesh.load_mesh(obj_list[mesh_nr])

        mesh_original.visual.face_colors = [64, 64, 64, 100]

        # Create 3d sphere, which is the region where centroid of the lens could be located
        sphere_radius = lens_init_centroid_z/1000-eye_ball_centroid[2]

        # Generate sphere points
        x, y, z = ellipsoid(sphere_radius, sphere_radius, sphere_radius)

        # Create a trimesh object from vertices and faces
        vertices = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)

        # Get the vertices which are within the 15 degree region
        if mesh_nr == 0:
            vertices_valid = []
        valid_point_index = 0
        print(f"all vertices: {len(vertices)} ")
        for vertices_index, sphere_point in enumerate(vertices):
            # put only <= 15 degree here after debug
            if (angle_between_two_vectors([0,0,1], sphere_point) <= 15
                    and angle_between_two_vectors([0,0,1], sphere_point) > 0
                    and sphere_point[0]>=0 and sphere_point[2]>=0 and sphere_point[1]<=0):
                print(f"lens scale {lens_scale}, vertices {vertices_index} is valid")
                if mesh_nr == 0:
                    vertices_valid.append(sphere_point)

                # Rotate lens
                ref_vertex = sphere_point
                print(ref_vertex)
                print(f"Before rotation, the angle in degree between ref vertex and lens centroid: {angle_between_two_vectors([0,0,1], ref_vertex)}")
                R = rotation_matrix_from_vectors([0,0,1], ref_vertex)
                Rotation = np.eye(4)
                Rotation[:3, :3] = R

                lens_mesh_ready = copy.deepcopy(lens_mesh)

                lens_mesh_ready.apply_transform(Rotation)

                lens_mesh_ready.apply_translation([eye_ball_centroid[0], eye_ball_centroid[1], eye_ball_centroid[2]])

                # check intersection between lens_mesh and mesh_original
                intersections = trimesh.boolean.intersection([lens_mesh_ready, mesh_original], engine='blender')
                if hasattr(intersections, 'vertices'):
                    print(f"Head {mesh_nr}, Vertices {vertices_index}/{len(vertices)}")
                    print("intersected\n")
                    head_hits[valid_point_index] += 1
                else:
                    print(f"Head {mesh_nr}, Vertices {vertices_index}/{len(vertices)}")
                    print("NOT intersected\n")
                valid_point_index += 1
        if mesh_nr == 0:
            vertices_valid = np.array(vertices_valid)
            print(vertices_valid.shape)

    print("head hit results:")
    print(head_hits[:len(vertices_valid)])

    # Plot 3d points from vertices_valid. Color with the value of head hit.
    x = vertices_valid[:, 0]*1000
    y = vertices_valid[:, 1]*1000
    z = vertices_valid[:, 2]*1000

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    colors = x # placeholder, later replace with the value of head hit.
    ax.scatter(x, y, z, c=head_hits[:len(vertices_valid)], cmap='inferno', marker='o')

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.savefig(f"{lens_scale}_3d.png")

    result_to_save =[x, y, z, head_hits[:len(vertices_valid)]]

    np.save(f"{lens_scale}", result_to_save)

