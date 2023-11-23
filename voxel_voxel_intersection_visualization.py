import open3d as o3d
import trimesh
import glob
import numpy as np
# import matplotlib.pyplot as plt
import copy
import random
import math
from utils import xyz_from_alpha_beta, intersection_elements, generate_random_color, rotation_matrix_from_vectors, angle_between_two_vectors, ellipsoid, trimesh2open3d, o3dTriangleMesh2PointCloud

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


SHOW_WIREFRAME = False
ONLY_SHOW_INTERSECTION = False
NOT_SHOW_MESH = False # Do not show any mesh. Only show the intersection and lens
SHOW_FULL_MESH = True # Show full mesh or ONLY "mesh of interest" which is used to find the intersection with lens plane

MESH_NR = 0
LeftEyeFront = 4043
LeftEyeRear = 4463
Head1 = 1726
Head2 = 1335
Head3 = 1203
MOUTH_ABOVE = 825
BROW_ABOVE = 2295
CUT_LENS = False
INTERACTIVE_INPUT = False

# Define a cylinder volumn where the hits should be ignored
# BOX1_HEIGHT_UPPER = 0.007
# BOX1_HEIGHT_LOWER = 0.015
# BOX1_WIDTH = 0.014
#
# BOX2_HEIGHT_UPPER = 0.006
# BOX2_HEIGHT_LOWER = 0.014
# BOX2_WIDTH = 0.025

CYLINDER_RADIUS = 0.012
CYLINDER_Y_SHIFT = -0.003

DEPTH_IGNORE = 0.03

eye_ball_shift = [0, 0, -1.30439425e-02] # Pre-calculated by averaging 53 EyeBallCentroid
lens_half_height_after_cut = 22

marker = trimesh.creation.axis(origin_size=0.0004, transform=None, origin_color=None, axis_radius=0.0002, axis_length=0.1)

while True:
    flag = input('输入y继续，输入n退出:') if INTERACTIVE_INPUT else 'y'
    if flag == 'y':
        # PITCH_TEMP = float(input('Size of voxel. Only [0.4, 0.5, 1] mm allowed. The smaller, the more accurate:')) if INTERACTIVE_INPUT else 0.4
        PITCH = 0.0004
        working_distance = float(input('工作距离[0,45]mm:')) if INTERACTIVE_INPUT else 10
        lens_diameter = float(input('镜片直径[20,80]mm:')) if INTERACTIVE_INPUT else 58

        # Define the lens rotation
        lens_alpha = float(input('镜片俯仰角[-90,90]度(+仰,-俯):')) if INTERACTIVE_INPUT else 12
        lens_beta = float(input('镜片内外旋角[-90,90]度(+内旋,-外旋):')) if INTERACTIVE_INPUT else 17

        # Define the eye rotation
        eye_alpha = float(input('眼睛俯仰角[-90,90]度(+俯,-仰):')) if INTERACTIVE_INPUT else 23
        eye_beta = float(input('眼睛内外旋角[-90,90]度(+外旋,-内旋):')) if INTERACTIVE_INPUT else 45

        print('\n')

        path = './voxel_results/FLORENCE'
        obj_list = glob.glob(f'{path}/**/*.obj', recursive=True)

        mesh_original = trimesh.load_mesh(obj_list[MESH_NR])
        mesh_original.visual.face_colors = [64, 64, 64, 100]

        scene = trimesh.Scene()

        # plot the LeftEyeFront/centroid point
        eye_ball_key_points = trimesh.points.PointCloud(vertices=[[0, 0, -eye_ball_shift[2]],[0,0,0]], colors=(0, 255, 0))

        scene.add_geometry(eye_ball_key_points)

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

        if not NOT_SHOW_MESH:
            if SHOW_FULL_MESH:
                scene.add_geometry(mesh_original)
            else:
                scene.add_geometry(mesh)

        # #######################  create cone lens  #######################

        cone_lens = trimesh.creation.cone(lens_diameter/2000, -working_distance/1000)
        cone_lens.apply_translation([0, 0, working_distance/1000-eye_ball_shift[2]])

        cone_lens_center = [0, 0, working_distance/1000-eye_ball_shift[2]]
        cone_top = [0, 0, -eye_ball_shift[2]]
        cone_lens_key_points = trimesh.points.PointCloud(vertices=[cone_top, cone_lens_center], colors=(255, 255, 0))
        cone_lens_top_point = trimesh.points.PointCloud(vertices=[cone_top], colors=(255, 255, 0))

        x,y,z = xyz_from_alpha_beta(lens_alpha, lens_beta)

        # Calculate the rotation matrix between initial direction vector [0,0,1) and (x,y,z)
        R = rotation_matrix_from_vectors((0,0,1), (x,y,z))
        Rotation_front = np.eye(4)
        Rotation_front[:3, :3] = R

        # Rotate the lens
        cone_lens.apply_transform(Rotation_front)
        cone_lens_key_points.apply_transform(Rotation_front)

        # Translate the lens to final position
        x_side,y_side,z_side = xyz_from_alpha_beta(eye_alpha, eye_beta)

        R = rotation_matrix_from_vectors((0,0,1), (x_side,y_side,z_side))
        Rotation_side = np.eye(4)
        Rotation_side[:3, :3] = R

        cone_lens_top_point.apply_transform(Rotation_side)

        cone_lens.apply_translation(cone_lens_top_point.vertices[0]-cone_lens_key_points.vertices[0])

        #######################  create a circle to show the rim the lens  #######################
        lens_rim = createCircle([0, 0, working_distance/1000-eye_ball_shift[2]], lens_diameter/2000)
        lens_rim.apply_transform(Rotation_front)
        lens_rim.apply_translation(cone_lens_top_point.vertices[0]-cone_lens_key_points.vertices[0])
        scene.add_geometry(lens_rim)

        cone_lens_key_points.apply_translation(cone_lens_top_point.vertices[0]-cone_lens_key_points.vertices[0])

        cone_lens.visual.face_colors = [0, 64, 64, 100]
        scene.add_geometry(cone_lens)
        scene.add_geometry(cone_lens_key_points)

        voxelized_cone_lens = cone_lens.voxelized(PITCH).fill()
        lens_voxelization = np.around(np.array(voxelized_cone_lens.points),4)
        lens_pcl = trimesh.PointCloud(vertices=np.array(lens_voxelization), colors=[0, 0, 255, 1])

        # scene.add_geometry(lens_pcl)

        # Create 3d sphere, which is the region where centroid of the lens could be located
        sphere_radius = working_distance/1000-eye_ball_shift[2]

        # Generate sphere points
        x, y, z = ellipsoid(sphere_radius, sphere_radius, sphere_radius)

        # Create a trimesh object from vertices and faces
        vertices = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)

        cloud = trimesh.PointCloud(vertices)
        sphere_mesh = cloud.convex_hull
        sphere_mesh.visual.face_colors = [255, 255, 0, 100]

        # scene.add_geometry(sphere_mesh)

        for mesh_nr in range(0, len(obj_list)):
            mesh_original = trimesh.load_mesh(obj_list[mesh_nr])
            mesh_original.visual.face_colors = [64, 64, 64, 50]
            scene.add_geometry(mesh_original)

        # #######################  create a volumn around cornea center where the head hits should be ignored  #######################
        # volumn1_ignore = trimesh.creation.box([BOX1_WIDTH, BOX1_HEIGHT_UPPER+BOX1_HEIGHT_LOWER, DEPTH_IGNORE])
        # volumn1_ignore.apply_translation([0, (BOX1_HEIGHT_UPPER+BOX1_HEIGHT_LOWER)/2-BOX1_HEIGHT_LOWER, DEPTH_IGNORE/2])
        # # scene.add_geometry(volumn1_ignore)
        #
        # voxelized_volumn1 = volumn1_ignore.voxelized(PITCH).fill()
        # volumn1_voxelization = np.around(np.array(voxelized_volumn1.points),4)
        # volumn1_pcl = trimesh.PointCloud(vertices=np.array(volumn1_voxelization), colors=[255, 0, 255, 100])
        # # scene.add_geometry(cylinder_pcl)
        #
        # volumn2_ignore = trimesh.creation.box([BOX2_WIDTH, BOX2_HEIGHT_UPPER+BOX2_HEIGHT_LOWER, DEPTH_IGNORE])
        # volumn2_ignore.apply_translation([0, (BOX2_HEIGHT_UPPER+BOX2_HEIGHT_LOWER)/2-BOX2_HEIGHT_LOWER, DEPTH_IGNORE/2])
        # # scene.add_geometry(volumn2_ignore)
        #
        # voxelized_volumn2= volumn2_ignore.voxelized(PITCH).fill()
        # volumn2_voxelization = np.around(np.array(voxelized_volumn2.points),4)
        # volumn2_pcl = trimesh.PointCloud(vertices=np.array(volumn2_voxelization), colors=[255, 0, 255, 100])
        # # scene.add_geometry(cylinder_pcl)

        cylinder_ignore = trimesh.creation.cylinder(radius=CYLINDER_RADIUS,height=DEPTH_IGNORE)
        cylinder_ignore.apply_translation([0, CYLINDER_Y_SHIFT, DEPTH_IGNORE/2])
        # scene.add_geometry(cylinder_ignore)

        voxelized_cylinder = cylinder_ignore.voxelized(PITCH).fill()
        cylinder_voxelization = np.around(np.array(voxelized_cylinder.points),4)
        cylinder_pcl = trimesh.PointCloud(vertices=np.array(cylinder_voxelization), colors=[255, 0, 255, 100])
        # scene.add_geometry(cylinder_pcl)

        # Visualize the trimesh
        scene.add_geometry(marker)
        scene.show(smooth=False, flags={'wireframe': SHOW_WIREFRAME})

        scene_voxel = trimesh.Scene()

        # load prepared voxel
        num_of_heads = np.load(f"voxel_results/num_of_heads_{PITCH}.npy")
        voxel_list_remove_zero = np.load(f"voxel_results/voxel_list_remove_zero_{PITCH}.npy")
        colors_list = np.load(f"voxel_results/colors_list_{PITCH}.npy")
        accumulation_remove_zero = np.load(f"voxel_results/accumulation_remove_zero_{PITCH}.npy")
        voxel_center_min = np.load(f"voxel_results/voxel_center_min_{PITCH}.npy")
        voxel_center_max = np.load(f"voxel_results/voxel_center_max_{PITCH}.npy")

        colors_np = np.array(colors_list)
        colors_np[:, 3] = 0.3

        multi_heads = trimesh.PointCloud(vertices=voxel_list_remove_zero, colors=list(colors_np))

        scene_voxel.add_geometry(eye_ball_key_points)
        scene_voxel.add_geometry(lens_pcl)
        scene_voxel.add_geometry(multi_heads)
        scene_voxel.add_geometry(lens_rim)

        scene_voxel.add_geometry(marker)
        scene_voxel.show(smooth=False, flags={'wireframe': SHOW_WIREFRAME})

        z_step = round((voxel_center_max[2]-voxel_center_min[2])/PITCH+1)
        y_step = round((voxel_center_max[1]-voxel_center_min[1])/PITCH+1)

        def voxel2index(v):
            return round((y_step*z_step*(v[0]-voxel_center_min[0])+z_step*(v[1]-voxel_center_min[1])+(v[2]-voxel_center_min[2]))/PITCH)

        lens_list = lens_voxelization.tolist()
        lens_indices = list(map(voxel2index, lens_list))

        # volumn1_list = volumn1_voxelization.tolist()
        # volumn1_voxel_indices = list(map(voxel2index, volumn1_list))
        #
        # volumn2_list = volumn2_voxelization.tolist()
        # volumn2_voxel_indices = list(map(voxel2index, volumn2_list))

        cylinder_list = cylinder_voxelization.tolist()
        cylinder_voxel_indices = list(map(voxel2index, cylinder_list))

        voxel_list = voxel_list_remove_zero.tolist()
        head_voxel_indices = list(map(voxel2index, voxel_list))

        # valid_lens_indices_temp = list(np.setdiff1d(np.array(lens_indices), np.array(volumn1_voxel_indices), True))
        # valid_lens_indices = list(np.setdiff1d(np.array(valid_lens_indices_temp), np.array(volumn2_voxel_indices), True))

        valid_lens_indices = list(np.setdiff1d(np.array(lens_indices), np.array(cylinder_voxel_indices), True))
        _, intersection_indices, _ = np.intersect1d(np.array(head_voxel_indices), np.array(valid_lens_indices), return_indices=True)

        intersection_voxels = voxel_list_remove_zero[intersection_indices]
        intersection_colors_list = colors_list[intersection_indices]
        head_hits = accumulation_remove_zero[intersection_indices]

        print(f'Voxel尺寸:{PITCH*1000} mm')
        print(f'工作距离:{working_distance} mm')
        print(f'镜片直径:{lens_diameter} mm')
        print(f'镜片俯仰角(+仰,-俯):{lens_alpha} 度')
        print(f'镜片内外旋角(+内旋,-外旋):{lens_beta} 度')
        print(f'眼睛俯仰角(+俯,-仰):{eye_alpha} 度')
        print(f'眼睛内外旋角(+外旋,-内旋):{eye_beta} 度')
        print('\n')

        if len(head_hits) > 0:
            print(f'碰撞人头数/总人头数:{max(head_hits)}/{len(obj_list)}')
            print(f'碰撞几率:{np.around(float(max(head_hits))/53,4)*100}%')
            print('\n')

            intersection_colors_np = np.array(intersection_colors_list)
            intersection_colors_np[:, 3] = 0.9
            intersection_multi_heads = trimesh.PointCloud(vertices=intersection_voxels, colors=list(intersection_colors_np))

            scene_voxel_intersection = trimesh.Scene()
            scene_voxel_intersection.add_geometry(intersection_multi_heads)
            scene_voxel_intersection.add_geometry(eye_ball_key_points)
            scene_voxel_intersection.add_geometry(cone_lens_center)

            path = r'C:\Users\xiaochen.wang\Projects\Dataset\FLORENCE'
            obj_list = glob.glob(f'{path}/**/*.obj', recursive=True)

            for mesh_nr in range(0, len(obj_list)):
                mesh_original = trimesh.load_mesh(obj_list[mesh_nr])
                mesh_original.visual.face_colors = [64, 64, 64, 50]
                scene_voxel_intersection.add_geometry(mesh_original)
            scene_voxel_intersection.add_geometry(lens_rim)
            scene_voxel_intersection.add_geometry(cone_lens_key_points)
            scene_voxel_intersection.add_geometry(marker)
            scene_voxel_intersection.show(smooth=False, flags={'wireframe': SHOW_WIREFRAME})
        else:
            print('NO intersection')

    elif flag == 'n':
        break
    else:
        continue