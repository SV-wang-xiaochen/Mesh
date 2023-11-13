# Translate the lens

import open3d as o3d
import numpy as np

eye_ball_centroid = [0, 0, -1.30439425e-02] # Pre-calculated by averaging 53 EyeBallCentroid
def main():
    path = r'C:\Users\xiaochen.wang\Projects\Dataset\lens_z12.obj'
    save_path = r'C:\Users\xiaochen.wang\Projects\Dataset\lens_new.obj'

    mesh = o3d.io.read_triangle_mesh(path)

    Translation = np.eye(4)
    Translation[0, 3] = 0
    Translation[1, 3] = 0
    Translation[2, 3] = -eye_ball_centroid[2]

    mesh_new = mesh.transform(Translation)

    o3d.io.write_triangle_mesh(save_path, mesh_new)

if __name__ == "__main__":
    main()
