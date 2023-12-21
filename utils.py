import open3d as o3d
import trimesh
import os
import numpy as np
import random
import math
import io
import PIL.Image as Image
import xlsxwriter


def xyz_from_alpha_beta(alpha, beta):
    """ Find the destination coordinate of lens (x,y,z) after rotation defined by alpha and beta
    :param alpha: Rotation angle, down-up direction.
    :param beta: Rotation angle, left-right direction.
    :return (x,y,z): destination coordinate of lens (x,y,z)
    """
    A = math.pow(math.tan(alpha / 180 * np.pi), 2)
    B = math.pow(math.tan(beta / 180 * np.pi), 2)

    if alpha == 90 and beta == 90:
        raise Exception("Sorry, the pair of rotation angles are invalid")
    elif alpha == 90:
        x = 0
        y = -1
        z = 0
    elif beta == 90:
        x = 1
        y = 0
        z = 0
    else:
        x = np.sign(beta) * math.sqrt(B / (A + B + 1))
        y = -np.sign(alpha) * math.sqrt(A / (A + B + 1))
        z = math.sqrt(1 / (A + B + 1))
    return x, y, z


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


def createCircle(center, radius):

    # Create an array of angles to parametrically define the circle
    num_points = 100  # Number of points to create the circle
    theta = np.linspace(0, 2 * np.pi, num_points)

    # Parametric equations to generate points on the circle
    x = radius * np.cos(theta) + center[0]
    y = radius * np.sin(theta) + center[1]
    z = np.full(num_points, center[2])  # Constant z-coordinate

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

    return circle


def saveSceneImage(scene, path):
    bytes_ = scene.save_image()
    image = Image.open(io.BytesIO(bytes_))
    image.save(path)


def loadVoxelizationResults(pitch):
    num_of_heads = np.load(f"voxel_results/num_of_heads_{pitch}.npy")
    voxel_list_remove_zero = np.load(f"voxel_results/voxel_list_remove_zero_{pitch}.npy")
    colors_list = np.load(f"voxel_results/colors_list_{pitch}.npy")
    accumulation_remove_zero = np.load(f"voxel_results/accumulation_remove_zero_{pitch}.npy")
    voxel_center_min = np.load(f"voxel_results/voxel_center_min_{pitch}.npy")
    voxel_center_max = np.load(f"voxel_results/voxel_center_max_{pitch}.npy")
    head_voxel_indices = np.load(f"voxel_results/head_voxel_indices_{pitch}.npy")

    return num_of_heads, voxel_list_remove_zero, colors_list, accumulation_remove_zero, voxel_center_min, voxel_center_max, head_voxel_indices

def createLensAndLightCone(lens_diameter, lens_thickness, cone_diameter, cone_angle, working_distance, eye_ball_shift):
    light_cone_radius = cone_diameter / 2000
    light_cone_height = light_cone_radius / math.tan(cone_angle / 2 / 180 * np.pi)
    light_cone = trimesh.creation.cone(light_cone_radius, -light_cone_height)
    light_cone.apply_translation([0, 0, working_distance / 1000 - eye_ball_shift[2]])

    lens = trimesh.creation.cylinder(lens_diameter / 2000, lens_thickness / 1000)

    return lens, light_cone, light_cone_height

def paraSweepTable(result_table, xlsx_path, summary, column_indices, row_indices, header):
    # Create a 2D NumPy array
    array_2d = np.array(result_table)

    # Create a new Excel file and add a worksheet
    workbook = xlsxwriter.Workbook(xlsx_path)
    worksheet = workbook.add_worksheet()

    # Write the 2D NumPy array to the worksheet
    for row_num, row_data in enumerate(array_2d):
        for col_num, value in enumerate(row_data):
            worksheet.write(row_num + 1, col_num + 1, value)

    # Write column and row indices

    for col_num, value in enumerate(np.array(column_indices)):
        worksheet.write(0, col_num + 1, value)

    for row_num, value in enumerate(np.array(row_indices)):
        worksheet.write(row_num + 1, 0, value)

    summary_start_row = len(row_indices) + 2

    for row_num, item in enumerate(summary):
        worksheet.write(summary_start_row + row_num, 0, item)

    worksheet.write(0, 0, header)

    # Close the workbook to save the changes
    workbook.close()
