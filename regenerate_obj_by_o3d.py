# The .obj file in our dataset stores the mesh by the order of vertices. We can check the index of vertices in Meshlab.
# o3d.io.read_triangle_mesh can load .obj file, however it loads by the order of faces rather than vertices.
# Therefore, the index of vertices seen from Meshlab is not correct anymore.

# Solution: Load the .obj file by o3d.io.read_triangle_mesh and then save it by o3d.io.write_triangle_mesh

import open3d as o3d
import glob

def main():
    path = r'C:\Users\xiaochen.wang\Projects\Dataset\FLORENCE'
    obj_list = glob.glob(f'{path}/**/*.obj', recursive=True)

    for item in obj_list:
        mesh = o3d.io.read_triangle_mesh(item)

        o3d.io.write_triangle_mesh(item, mesh)

if __name__ == "__main__":
    main()
