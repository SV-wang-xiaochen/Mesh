# Align head meshes of FLORENCE dataset by rules:
# 1) mid plane of any head is parallel to y-z plane
# 2) coordinate of LeftEyeFront point of any head is identical
# 3) eye axis of any head is in a plane which is perpendicular to mid plane

import open3d as o3d
import numpy as np
import copy
import glob
import os

LeftEyeFront = 4043
LeftEyeRear = 4463
RightEyeFront = 4587
Head1 = 1726
Head2 = 1335
Head3 = 1203


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


def get_plane_normal_vector_from_points(p, q, r):
    """ Calculate the normal vector of a 3d plane from three points
    :param p, q, r: three 3d points
    :return a, b, c: normal vector
    """

    (x1, y1, z1) = p
    (x2, y2, z2) = q
    (x3, y3, z3) = r
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * x1 - b * y1 - c * z1)  # if plane equation is needed
    return a, b, c


def align(mesh):
    """ Align the mid plane, LeftEyeFront point and eye axis
    :param mesh: source mesh
    :return copy.deepcopy(mesh_mid).transform(Translation): aligned mesh
    """

    ref_plane = (1, 0, 0)
    mid_plane = get_plane_normal_vector_from_points(mesh.vertices[Head1], mesh.vertices[Head2], mesh.vertices[Head3])
    print(f"Before rotation, the angle in degree between ref plane and mid plane: {angle_between_two_vectors(ref_plane,mid_plane)}\n")

    # align mid plane
    R = rotation_matrix_from_vectors(mid_plane, ref_plane)
    Rotation = np.eye(4)
    Rotation[:3, :3] = R

    mesh_mid = copy.deepcopy(mesh).transform(Rotation)

    ref_plane = (1, 0, 0)
    mid_plane = get_plane_normal_vector_from_points(mesh_mid.vertices[Head1], mesh_mid.vertices[Head2], mesh_mid.vertices[Head3])

    print(f"After rotation, the angle in degree between ref plane and mid plane: {angle_between_two_vectors(ref_plane,mid_plane)}\n")

    # align LeftEyeFront point
    trans = (0, 0, 0) - mesh_mid.vertices[LeftEyeFront]
    Translation = np.eye(4)
    Translation[0, 3] = trans[0]
    Translation[1, 3] = trans[1]
    Translation[2, 3] = trans[2]

    mesh_mid_T = copy.deepcopy(mesh_mid).transform(Translation)

    # align eye axis in a plane which is perpendicular to mid plane (also parallel to x-z plane)
    ref_vec = (0, 0, -1)
    eye_axis_vec = mesh_mid_T.vertices[LeftEyeRear]
    eye_axis_vec[0] = 0
    print('eye_axis_vec')
    print(eye_axis_vec)

    R = rotation_matrix_from_vectors(eye_axis_vec, ref_vec)
    Rotation = np.eye(4)
    Rotation[:3, :3] = R

    return copy.deepcopy(mesh_mid_T).transform(Rotation)


def main():
    path = r'C:\Users\xiaochen.wang\Projects\Dataset\FLORENCE'
    obj_list = glob.glob(f'{path}/**/*.obj', recursive=True)

    mesh_plot_list = []

    for mesh_nr in range(0,len(obj_list)):

        print("Source mesh:")
        print(mesh_nr, os.path.basename(obj_list[mesh_nr]), '\n')

        mesh = o3d.io.read_triangle_mesh(obj_list[mesh_nr])

        mesh_aligned = align(mesh)

        mesh_plot_list.append(mesh_aligned)

        # # Uncomment to save processed mesh
        # o3d.io.write_triangle_mesh(obj_list[mesh_nr], mesh_aligned)

    o3d.visualization.draw_geometries(mesh_plot_list, mesh_show_wireframe=True)

    # # Uncomment for Debug
    # o3d.io.write_triangle_mesh('1.obj', mesh_plot_list[0])
    # o3d.io.write_triangle_mesh('2.obj', mesh_plot_list[1])

if __name__ == "__main__":
    main()
