import open3d as o3d
import matplotlib.pyplot as plt
import glob
import os
import trimesh
import numpy as np


# def main():
#     path = r'C:\Users\xiaochen.wang\Projects\Dataset\FLORENCE'
#     obj_list = glob.glob(f'{path}/**/*.obj', recursive=True)
#
#     for mesh_nr in range(0,1):
#
#         print("Source mesh:")
#         print(mesh_nr, os.path.basename(obj_list[mesh_nr]), '\n')
#
#         # mesh = o3d.io.read_triangle_mesh(obj_list[mesh_nr])
#         mesh = trimesh.load_mesh(obj_list[mesh_nr])
#
#         # Define the ray as an origin and a direction
#         ray_origin = np.array([0, 0, 0])  # Replace with your ray origin
#         ray_direction = np.array([1, 0, 0])  # Replace with your ray direction
#
#         # Perform ray-mesh intersection check
#         intersections = mesh.ray.intersects_location(
#             ray_origins=[ray_origin],
#             ray_directions=[ray_direction]
#         )
#         #
#         # # Check if there is an intersection
#         # if len(intersections) > 0:
#         #     print("Ray intersects the mesh at points:", intersections)
#         # else:
#         #     print("Ray does not intersect the mesh.")
#
# if __name__ == "__main__":
#     main()


# # test on a sphere primitive
# mesh = trimesh.creation.icosphere()
#
# # create some rays
# ray_origins = np.array([[0, 0, -3],
#                         [2, 2, -3]])
# ray_directions = np.array([[0, 0, 1],
#                            [0, 0, 1]])
#
# # # check out the docstring for intersects_location queries
# # mesh.ray.intersects_location.__doc__
#
# # run the mesh-ray query
# locations, index_ray, index_tri = mesh.ray.intersects_location(
#         ray_origins=ray_origins,
#         ray_directions=ray_directions)
#
# # stack rays into line segments for visualization as Path3D
# ray_visualize = trimesh.load_path(np.hstack((ray_origins,
#                                              ray_origins + ray_directions*5.0)).reshape(-1, 2, 3))
#
# # unmerge so viewer doesn't smooth
# mesh.unmerge_vertices()
# # make mesh white- ish
# mesh.visual.face_colors = [255,255,255,255]
# mesh.visual.face_colors[index_tri] = [255, 0, 0, 255]
#
# # create a visualization scene with rays, hits, and mesh
# scene = trimesh.Scene([mesh, ray_visualize])
#
# # show the visualization
# scene.show()

# import open3d as o3d
# import matplotlib.pyplot as plt
#
# cube = o3d.t.geometry.TriangleMesh.from_legacy(
#                                     o3d.geometry.TriangleMesh.create_box())
#
# # Create scene and add the cube mesh
# scene = o3d.t.geometry.RaycastingScene()
# scene.add_triangles(cube)
#
# # Rays are 6D vectors with origin and ray direction.
# # Here we use a helper function to create rays for a pinhole camera.
# rays = scene.create_rays_pinhole(fov_deg=60,
#                                  center=[0.5,0.5,0.5],
#                                  eye=[-1,-1,-1],
#                                  up=[0,0,1],
#                                  width_px=320,
#                                  height_px=240)
#
# # Compute the ray intersections.
# ans = scene.cast_rays(rays)
#
# # Visualize the hit distance (depth)
# plt.imshow(ans['t_hit'].numpy())

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# # Create a function to generate the points on the surface of an ellipsoid
# def ellipsoid(a, b, c):
#     u = np.linspace(0, 2 * np.pi, 100)
#     v = np.linspace(0, np.pi, 50)
#     x = a * np.outer(np.cos(u), np.sin(v))
#     y = b * np.outer(np.sin(u), np.sin(v))
#     z = c * np.outer(np.ones(np.size(u)), np.cos(v))
#     return x, y, z
#
# # Define the semi-axes lengths of the ellipsoid
# a, b, c = 3, 5, 2
#
# # Generate ellipsoid points
# x, y, z = ellipsoid(a, b, c)
#
# # Plot the ellipsoid
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x, y, z, color='b')
#
# # Set labels
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
#
# plt.title('Ellipsoid')
# plt.show()

# import trimesh
# import numpy as np
#
# def ellipsoid(a, b, c):
#     u = np.linspace(0, 2 * np.pi, 100)
#     v = np.linspace(0, np.pi, 50)
#     x = a * np.outer(np.cos(u), np.sin(v))
#     y = b * np.outer(np.sin(u), np.sin(v))
#     z = c * np.outer(np.ones(np.size(u)), np.cos(v))
#     return x, y, z
#
# # Define the semi-axes lengths of the ellipsoid
# a, b, c = 3, 5, 2
#
# # Generate ellipsoid points
# x, y, z = ellipsoid(a, b, c)
#
# # Create a trimesh object from vertices and faces
# vertices = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)
# # faces = np.array(trimesh.creation.triangulate_quads(x, y, z))  # Generating faces for the ellipsoid
# cloud = trimesh.PointCloud(vertices)
# mesh = cloud.convex_hull
#
# # mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
# #
# scene = trimesh.scene.Scene([mesh])
# scene.show(smooth=False)

import open3d as o3d
import numpy as np

def ellipsoid(a, b, c):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    x = a * np.outer(np.cos(u), np.sin(v))
    y = b * np.outer(np.sin(u), np.sin(v))
    z = c * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z

# Define the semi-axes lengths of the ellipsoid
a, b, c = 3, 5, 2

# Generate ellipsoid points
x, y, z = ellipsoid(a, b, c)

# Combine the points into a single point cloud
points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)
# Create an Open3D point cloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
# point_cloud = point_cloud.compute_convex_hull()

o3d.io.write_point_cloud("ellipsoid.ply", point_cloud)

# # print(point_cloud)
# # # print(a)
# # # print(point_cloud.points)
# # # # Create a mesh from the point cloud
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud)
# # #
#
#
# # mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
# # # Visualize the mesh
# o3d.visualization.draw_geometries([mesh])