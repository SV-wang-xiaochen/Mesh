import trimesh
import glob
import numpy as np
from utils import xyz_from_alpha_beta, intersection_elements, generate_random_color, rotation_matrix_from_vectors, angle_between_two_vectors, ellipsoid, trimesh2open3d, o3dTriangleMesh2PointCloud, createCircle, saveSceneImage
import math

SHOW_WIREFRAME = False
ONLY_SHOW_INTERSECTION = False
NOT_SHOW_MESH = False # Do not show any mesh. Only show the intersection and lens
SHOW_FULL_MESH = True # Show full mesh or ONLY "mesh of interest" which is used to find the intersection with lens plane

LeftEyeFront = 4043
LeftEyeRear = 4463
Head1 = 1726
Head2 = 1335
Head3 = 1203
MOUTH_ABOVE = 825
BROW_ABOVE = 2295
CUT_LENS = False
INTERACTIVE_INPUT = False
SHOW_LIGHT_BLOCK = True

CYLINDER_RADIUS = 0.015
CYLINDER_Y_SHIFT = -0.003
DEPTH_IGNORE = 0.03

LENS_THICKNESS = 0.01

VOXEL_TRANSPARENCY = 0.1

eye_ball_shift = [0, 0, -1.30439425e-02] # Pre-calculated by averaging 53 EyeBallCentroid
lens_half_height_after_cut = 22

marker = trimesh.creation.axis(origin_size=0.0004, transform=None, origin_color=None, axis_radius=0.0002, axis_length=0.1)

PITCH = 0.001

#######################  load a predefined elliptical cylinder where the light blocks should be ignored  #######################
elliptical_cylinder_path = f'voxel_results/EllipticalCylinder.obj'
elliptical_cylinder = trimesh.load_mesh(elliptical_cylinder_path)
elliptical_cylinder.apply_scale(0.001)

voxelized_elliptical_cylinder = elliptical_cylinder.voxelized(PITCH).fill()
elliptical_cylinder_voxelization = np.around(np.array(voxelized_elliptical_cylinder.points), 4)
elliptical_cylinder_pcl = trimesh.PointCloud(vertices=np.array(elliptical_cylinder_voxelization), colors=[255, 0, 255, 100])

# load prepared voxel
num_of_heads = np.load(f"voxel_results/num_of_heads_{PITCH}.npy")
voxel_list_remove_zero = np.load(f"voxel_results/voxel_list_remove_zero_{PITCH}.npy")
colors_list = np.load(f"voxel_results/colors_list_{PITCH}.npy")
accumulation_remove_zero = np.load(f"voxel_results/accumulation_remove_zero_{PITCH}.npy")
voxel_center_min = np.load(f"voxel_results/voxel_center_min_{PITCH}.npy")
voxel_center_max = np.load(f"voxel_results/voxel_center_max_{PITCH}.npy")

####################### select Colormap for heat map #######################
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Normalize the values in the list to range between 0 and 1
norm = mcolors.Normalize(vmin=0, vmax=num_of_heads)

# Create a scalar mappable using the colormap
alpha_value = 1
scalar_map = plt.cm.ScalarMappable(cmap='Greys', norm=norm)

# adjust transparency
colors_np = np.array(colors_list)
colors_np[:, 3] = VOXEL_TRANSPARENCY

multi_heads = trimesh.PointCloud(vertices=voxel_list_remove_zero, colors=list(colors_np))

# Convert each value in the list to a color using the colormap
accumulation_np = np.array(accumulation_remove_zero)
colors_list = scalar_map.to_rgba(accumulation_np, alpha_value)

common_colors = [
    [0.745, 0.788, 0.886, 1],
    [0.353, 0.702, 0.863, 1],
    [0.141, 0.906, 0.22, 1],
    [1, 0.8, 0, 1],
    [0.91, 0.039, 0.039, 1],
    [0.682, 0.137, 0.137, 1],
    [0.427, 0.102, 0.102, 1],
    [0.322, 0.047, 0.047, 1],
    [0.165, 0, 0, 1],
    [0, 0, 0, 1]
]

for i, color in enumerate(common_colors):
    colors_list[accumulation_np==(i+1)] = color
colors_list[accumulation_np>10] = [0, 0, 0, 1]

z_step = round((voxel_center_max[2] - voxel_center_min[2]) / PITCH + 1)
y_step = round((voxel_center_max[1] - voxel_center_min[1]) / PITCH + 1)


def voxel2index(v):
    return round((y_step * z_step * (v[0] - voxel_center_min[0]) + z_step * (v[1] - voxel_center_min[1]) + (
                v[2] - voxel_center_min[2])) / PITCH)

head_voxel_list = voxel_list_remove_zero.tolist()
head_voxel_indices = list(map(voxel2index, head_voxel_list))

while True:
    flag = input('输入y继续，输入n退出:') if INTERACTIVE_INPUT else 'y'
    if flag == 'y':
        # PITCH_TEMP = float(input('Size of voxel. Only [0.4, 0.5, 1] mm allowed. The smaller, the more accurate:')) if INTERACTIVE_INPUT else 0.4

        working_distance = float(input('工作距离(周边)[0,45]mm:')) if INTERACTIVE_INPUT else 10
        lens_diameter = float(input('目镜外框直径[20,80]mm:')) if INTERACTIVE_INPUT else 58

        # Define the light cone
        cone_diameter = float(input('通光孔径[20,80]mm:')) if INTERACTIVE_INPUT else 43
        cone_angle = float(input('单张范围（眼外角）[70,140]度:')) if INTERACTIVE_INPUT else 110

        # Define the lens rotation
        lens_alpha = float(input('机器俯仰角[-90,90]度(+仰,-俯):')) if INTERACTIVE_INPUT else 12
        lens_beta = float(input('机器内外旋角[-90,90]度(+内旋,-外旋):')) if INTERACTIVE_INPUT else 25

        # Define the eye rotation
        eye_alpha = float(input('眼睛俯仰角[-30,45]度(+俯,-仰):')) if INTERACTIVE_INPUT else 12
        eye_beta = float(input('眼睛内外旋角[-45,45]度(+外旋,-内旋):')) if INTERACTIVE_INPUT else 10

        print('\n')

        path = './voxel_results/FLORENCE'
        obj_list = glob.glob(f'{path}/**/*.obj', recursive=True)

        scene = trimesh.Scene()

        # plot the LeftEyeFront/centroid point
        eye_ball_key_points = trimesh.points.PointCloud(vertices=[[0, 0, -eye_ball_shift[2]],[0,0,0]], colors=(0, 255, 0))

        scene.add_geometry(eye_ball_key_points)

        # #######################  create light cone and lens #######################
        light_cone_radius = cone_diameter / 2000
        light_cone_height = light_cone_radius / math.tan(cone_angle / 2 / 180*np.pi)
        light_cone = trimesh.creation.cone(light_cone_radius, -light_cone_height)
        light_cone.apply_translation([0, 0, working_distance / 1000 - eye_ball_shift[2]])

        lens = trimesh.creation.cylinder(lens_diameter/2000, LENS_THICKNESS/1000)
        lens.apply_translation([0, 0, working_distance/1000-eye_ball_shift[2]])

        lens_center = [0, 0, working_distance/1000-eye_ball_shift[2]]
        lens_cone_top = [0, 0, -eye_ball_shift[2]]
        lens_cone_key_points = trimesh.points.PointCloud(vertices=[lens_cone_top, lens_center], colors=(255, 255, 0))
        lens_cone_top_point = trimesh.points.PointCloud(vertices=[lens_cone_top], colors=(255, 255, 0))

        light_cone_center = [0, 0, working_distance/1000-eye_ball_shift[2]]
        light_cone_top = [0, 0, working_distance/1000-eye_ball_shift[2]-light_cone_height]
        light_cone_key_points = trimesh.points.PointCloud(vertices=[light_cone_top, light_cone_center], colors=(255, 0, 0))

        x,y,z = xyz_from_alpha_beta(lens_alpha, lens_beta)

        # Calculate the rotation matrix between initial direction vector [0,0,1) and (x,y,z)
        R = rotation_matrix_from_vectors((0,0,1), (x,y,z))
        Rotation_front = np.eye(4)
        Rotation_front[:3, :3] = R

        # Rotate the lens
        light_cone.apply_transform(Rotation_front)
        lens.apply_transform(Rotation_front)
        lens_cone_key_points.apply_transform(Rotation_front)
        light_cone_key_points.apply_transform(Rotation_front)

        # Translate the lens to final position
        x_side,y_side,z_side = xyz_from_alpha_beta(eye_alpha, eye_beta)

        R = rotation_matrix_from_vectors((0,0,1), (x_side,y_side,z_side))
        Rotation_side = np.eye(4)
        Rotation_side[:3, :3] = R

        lens_cone_top_point.apply_transform(Rotation_side)

        light_cone.apply_translation(lens_cone_top_point.vertices[0]-lens_cone_key_points.vertices[0])
        light_cone_key_points.apply_translation(lens_cone_top_point.vertices[0] - lens_cone_key_points.vertices[0])
        lens.apply_translation(lens_cone_top_point.vertices[0] - lens_cone_key_points.vertices[0])

        #######################  create a circle to show the rim the lens  #######################
        lens_rim = createCircle([0, 0, working_distance/1000-eye_ball_shift[2]], lens_diameter/2000)
        lens_rim.apply_transform(Rotation_front)
        lens_rim.apply_translation(lens_cone_top_point.vertices[0]-lens_cone_key_points.vertices[0])
        scene.add_geometry(lens_rim)

        lens_cone_key_points.apply_translation(lens_cone_top_point.vertices[0]-lens_cone_key_points.vertices[0])

        light_cone.visual.face_colors = [0, 64, 64, 100]
        lens.visual.face_colors = [64, 64, 64, 100]

        if SHOW_LIGHT_BLOCK:
            scene.add_geometry(light_cone)
            scene.add_geometry(light_cone_key_points)

        scene.add_geometry(lens)
        scene.add_geometry(lens_cone_key_points)

        # ####################### Debug Only: Create 3d sphere, which is the region where centroid of the lens could be located #######################
        # sphere_radius = working_distance/1000-eye_ball_shift[2]
        #
        # # Generate sphere points
        # x, y, z = ellipsoid(sphere_radius, sphere_radius, sphere_radius)
        #
        # # Create a trimesh object from vertices and faces
        # vertices = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)
        #
        # cloud = trimesh.PointCloud(vertices)
        # sphere_mesh = cloud.convex_hull
        # sphere_mesh.visual.face_colors = [255, 255, 0, 100]
        #
        # # scene.add_geometry(sphere_mesh)

        for mesh_nr in range(0, len(obj_list)):
            mesh_original = trimesh.load_mesh(obj_list[mesh_nr])
            mesh_original.visual.face_colors = generate_random_color()
            scene.add_geometry(mesh_original)

        # Visualize the trimesh
        scene.add_geometry(marker)
        # saveSceneImage(scene, '1.png')

        # Access the camera in the scene
        camera = scene.camera

        # # set camera orientation
        # scene.set_camera((90 / 180 * np.pi, 90 / 180 * np.pi, 0 / 180 * np.pi))

        scene.show(smooth=False, flags={'wireframe': SHOW_WIREFRAME})

        #######################  Show Overlapping Voxelization  #######################
        if SHOW_LIGHT_BLOCK:
            voxelized_light_cone = light_cone.voxelized(PITCH).fill()
            light_cone_voxelization = np.around(np.array(voxelized_light_cone.points),4)
            light_cone_pcl = trimesh.PointCloud(vertices=np.array(light_cone_voxelization), colors=[0, 255, 0, 100])

            # scene.add_geometry(light_cone_pcl)

        voxelized_lens = lens.voxelized(PITCH).fill()
        lens_voxelization = np.around(np.array(voxelized_lens.points),4)
        lens_pcl = trimesh.PointCloud(vertices=np.array(lens_voxelization), colors=[0, 0, 255, 100])

        # scene.add_geometry(lens_pcl)

        # show voxel of multi heads and lens
        scene_voxel_lens = trimesh.Scene()
        scene_voxel_lens.add_geometry(eye_ball_key_points)
        scene_voxel_lens.add_geometry(lens_pcl)
        scene_voxel_lens.add_geometry(multi_heads)

        scene_voxel_lens.add_geometry(marker)
        scene_voxel_lens.show(smooth=False, flags={'wireframe': SHOW_WIREFRAME}, line_settings={'point_size':5})

        if SHOW_LIGHT_BLOCK:
            scene_voxel_light_cone = trimesh.Scene()
            scene_voxel_light_cone.add_geometry(eye_ball_key_points)
            scene_voxel_light_cone.add_geometry(light_cone_pcl)
            scene_voxel_light_cone.add_geometry(multi_heads)

            scene_voxel_light_cone.add_geometry(marker)
            scene_voxel_light_cone.show(smooth=False, flags={'wireframe': SHOW_WIREFRAME}, line_settings={'point_size':5})

        #######################  Show Intersection Heat Map  #######################

        if SHOW_LIGHT_BLOCK:
            # calculate light blocks
            light_cone_list = light_cone_voxelization.tolist()
            light_cone_indices = list(map(voxel2index, light_cone_list))

            elliptical_cylinder_list = elliptical_cylinder_voxelization.tolist()
            elliptical_cylinder_voxel_indices = list(map(voxel2index, elliptical_cylinder_list))

            valid_light_cone_indices = list(np.setdiff1d(np.array(light_cone_indices), np.array(elliptical_cylinder_voxel_indices), True))
            _, light_block_indices, _ = np.intersect1d(np.array(head_voxel_indices), np.array(valid_light_cone_indices), return_indices=True)

            block_voxels = voxel_list_remove_zero[light_block_indices]
            block_colors_list = colors_list[light_block_indices]
            blocks = accumulation_remove_zero[light_block_indices]

        # calculate lens hits
        lens_list = lens_voxelization.tolist()
        lens_indices = list(map(voxel2index, lens_list))

        _, hit_indices, _ = np.intersect1d(np.array(head_voxel_indices), np.array(lens_indices),return_indices=True)

        hit_voxels = voxel_list_remove_zero[hit_indices]
        hit_colors_list = colors_list[hit_indices]
        hits = accumulation_remove_zero[hit_indices]

        print(f'工作距离（周边）:{working_distance} mm')
        print(f'目镜外框直径:{lens_diameter} mm')
        print(f'通光孔径:{cone_diameter} mm')
        print(f'单张范围（眼外角）:{cone_angle} mm')
        print(f'机器俯仰角(+仰,-俯):{lens_alpha} 度')
        print(f'机器内外旋角(+内旋,-外旋):{lens_beta} 度')
        print(f'眼睛俯仰角(+俯,-仰):{eye_alpha} 度')
        print(f'眼睛内外旋角(+外旋,-内旋):{eye_beta} 度')
        print(f'俯仰眼位 夹角:{eye_alpha-lens_alpha} 度')
        print(f'鼻颞眼位 夹角:{eye_beta-lens_beta} 度')
        print('\n')

        if len(hits)>0:
            print(f'镜片碰撞人头数/总人头数:{max(hits)}/{len(obj_list)}')
            print(f'镜片碰撞几率:{np.around(float(max(hits))/53,4)*100}%')
            print('\n')

            intersection_multi_heads = trimesh.PointCloud(vertices=hit_voxels, colors=hit_colors_list)

            scene_voxel_intersection = trimesh.Scene()
            scene_voxel_intersection.add_geometry(intersection_multi_heads)
            scene_voxel_intersection.add_geometry(eye_ball_key_points)
            scene_voxel_intersection.add_geometry(lens_center)

            path = f'voxel_results/FLORENCE'
            obj_list = glob.glob(f'{path}/**/*.obj', recursive=True)

            for mesh_nr in range(0, len(obj_list)):
                mesh_original = trimesh.load_mesh(obj_list[mesh_nr])
                mesh_original.visual.face_colors = [64, 64, 64, 100]
                scene_voxel_intersection.add_geometry(mesh_original)
            scene_voxel_intersection.add_geometry(lens_rim)
            scene_voxel_intersection.add_geometry(lens_cone_key_points)
            scene_voxel_intersection.add_geometry(marker)
            # saveSceneImage(scene_voxel_intersection, '3.png')
            scene_voxel_intersection.show(smooth=False, flags={'wireframe': SHOW_WIREFRAME}, line_settings={'point_size':10})
        else:
            print('无碰撞')

        if SHOW_LIGHT_BLOCK:
            if len(blocks)>0:
                print(f'光路遮挡人头数/总人头数:{max(blocks)}/{len(obj_list)}')
                print(f'光路遮挡几率:{np.around(float(max(blocks))/53,4)*100}%')
                print('\n')

                intersection_multi_heads = trimesh.PointCloud(vertices=block_voxels, colors=block_colors_list)

                scene_voxel_intersection = trimesh.Scene()
                scene_voxel_intersection.add_geometry(intersection_multi_heads)
                scene_voxel_intersection.add_geometry(eye_ball_key_points)
                scene_voxel_intersection.add_geometry(lens_center)

                path = f'voxel_results/FLORENCE'
                obj_list = glob.glob(f'{path}/**/*.obj', recursive=True)

                for mesh_nr in range(0, len(obj_list)):
                    mesh_original = trimesh.load_mesh(obj_list[mesh_nr])
                    mesh_original.visual.face_colors = [64, 64, 64, 100]
                    scene_voxel_intersection.add_geometry(mesh_original)
                scene_voxel_intersection.add_geometry(lens_rim)
                scene_voxel_intersection.add_geometry(light_cone_key_points)
                scene_voxel_intersection.add_geometry(marker)
                # saveSceneImage(scene_voxel_intersection, '3.png')
                scene_voxel_intersection.show(smooth=False, flags={'wireframe': SHOW_WIREFRAME}, line_settings={'point_size':10})
            else:
                print('无遮挡')

    elif flag == 'n':
        break
    else:
        continue