import open3d as o3d
import trimesh
import glob
import numpy as np
# import matplotlib.pyplot as plt
import copy
import random
import math
import io
import PIL.Image as Image

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

