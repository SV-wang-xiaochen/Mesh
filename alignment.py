import open3d as o3d
import numpy as np
import copy
import glob
import random
import os

LeftEyeFront = 4051
LeftEyeRear = 4346

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if any(v): #if not all zeros then
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    else:
        return np.eye(3) #cross of all zeros only occurs on identical directions

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


# def visualize(mesh):
#     vis = o3d.visualization.Visualizer()
#     vis.create_window()
#     vis.add_geometry(mesh)
#     vis.run()
#     vis.destroy_window()

def main():
    path = r'C:\Users\xiaochen.wang\Projects\Dataset\FLORENCE'
    list = glob.glob(f'{path}/**/*.obj', recursive=True)

    meshNr1 = 52
    meshNr2 = random.randint(0, len(list))

    while (meshNr1 == meshNr2):
        meshNr2 = random.randint(0, len(list))

    print("Destination mesh:")
    print(meshNr1, os.path.basename(list[meshNr1]))
    print("Source mesh:")
    print(meshNr2, os.path.basename(list[meshNr2]))

    mesh1 = o3d.io.read_triangle_mesh(list[meshNr1])
    # print(mesh1.vertices[LeftEyeFront])
    # Convert the triangle mesh to a point cloud
    # point_cloud1 = mesh1.sample_points_poisson_disk(number_of_points=5000)

    mesh2 = o3d.io.read_triangle_mesh(list[meshNr2])
    # print(mesh2.vertices[LeftEyeFront])
    # Convert the triangle mesh to a point cloud
    # point_cloud2 = mesh2.sample_points_poisson_disk(number_of_points=10000)
    # Visualize the point cloud

    eyeAxis1 = mesh1.vertices[LeftEyeFront] - mesh1.vertices[LeftEyeRear]
    eyeAxis2 = mesh2.vertices[LeftEyeFront] - mesh2.vertices[LeftEyeRear]

    print(f"Before transformation, the angle in degree between two eye axes: {angle_between_two_vectors(eyeAxis1,eyeAxis2)}")

    R = rotation_matrix_from_vectors(eyeAxis2, eyeAxis1)
    Rotation = np.eye(4)
    Rotation[:3, :3] = R

    # T =  np.matmul(np.eye(4), Translation)
    mesh2 = copy.deepcopy(mesh2).transform(Rotation)

    trans = mesh1.vertices[LeftEyeFront] - mesh2.vertices[LeftEyeFront]
    Translation = np.eye(4)
    Translation[0, 3] = trans[0]
    Translation[1, 3] = trans[1]
    Translation[2, 3] = trans[2]

    mesh2 = copy.deepcopy(mesh2).transform(Translation)

    mesh1.paint_uniform_color([1, 0, 0])
    mesh2.paint_uniform_color([0, 1, 0])
    o3d.visualization.draw_geometries([mesh1, mesh2], mesh_show_wireframe=True)

    eyeAxis1 = mesh1.vertices[LeftEyeFront] - mesh1.vertices[LeftEyeRear]
    eyeAxis2 = mesh2.vertices[LeftEyeFront] - mesh2.vertices[LeftEyeRear]

    print(f"After transformation, the angle in degree between two eye axes: {angle_between_two_vectors(eyeAxis1,eyeAxis2)}")

    print("Check if coordinates of two LeftEyeFront points are identical:")
    print(f"Mesh1: {mesh1.vertices[LeftEyeFront]}")
    print(f"Mesh2: {mesh2.vertices[LeftEyeFront]}")

if __name__ == "__main__":
    main()
