# Rotate the head mesh of aligned v1

import open3d as o3d
import glob
import numpy as np

eye_ball_centroid = [0, 0, -1.30439425e-02] # Pre-calculated by averaging 53 EyeBallCentroid

def main():
    path = r'C:\Users\xiaochen.wang\Projects\Dataset\FLORENCE'
    obj_list = glob.glob(f'{path}/**/*.obj', recursive=True)

    mesh = o3d.io.read_triangle_mesh(path)

    Translation = np.eye(4)
    Translation[0, 3] = 0
    Translation[1, 3] = 0
    Translation[2, 3] = -eye_ball_centroid[2]

    x_angle = 8 / 180 * np.pi
    x_Rotation = np.eye(4)
    x_R = np.array([[1, 0, 0],
                    [0, np.cos(x_angle), -np.sin(x_angle)],
                    [0, np.sin(x_angle), np.cos(x_angle)]])
    x_Rotation[:3, :3] = x_R

    for mesh_nr in range(0, len(obj_list)):
        mesh = o3d.io.read_triangle_mesh(obj_list[mesh_nr])
        mesh1 = mesh.transform(Translation)
        mesh2 = mesh1.transform(x_Rotation)

        o3d.io.write_triangle_mesh(obj_list[mesh_nr], mesh2)

if __name__ == "__main__":
    main()
