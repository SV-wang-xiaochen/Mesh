# Assume that there is a plane which is parallel to x-y plane. It moves along z-axis.
# This code calculates how many heads are touched by the plane at different z.
# "Head is touched" is defined that z value of the plane is equal or smaller than that of the check point.

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import glob
import os

NUM_BIN = 50

NASAL_TIP = ['Nasal Tip', 2825]
NASAL_ALA = ['Nasal Ala', 2881]
BROW_BONE = ['Brow Bone', 2111]

# set CHECK_POINT here
CHECK_POINT = NASAL_TIP


def main():
    path = r'C:\Users\xiaochen.wang\Projects\Dataset\FLORENCE'
    obj_list = glob.glob(f'{path}/**/*.obj', recursive=True)

    mesh_plot_list = []

    check_point_values = []

    for mesh_nr in range(0,len(obj_list)):

        print("Source mesh:")
        print(mesh_nr, os.path.basename(obj_list[mesh_nr]), '\n')

        mesh = o3d.io.read_triangle_mesh(obj_list[mesh_nr])

        mesh_plot_list.append(mesh)

        if CHECK_POINT == NASAL_TIP:
            check_point_value = np.max(np.asarray(mesh.vertices)[:,2])*1000
        else:
            check_point_value = np.asarray(mesh.vertices)[CHECK_POINT[1], 2]*1000

        check_point_values.append(check_point_value)

    # o3d.visualization.draw_geometries(mesh_plot_list, mesh_show_wireframe=True)
    print(sorted(check_point_values))

    # # Create histogram and cumulative frequency histogram
    res = stats.cumfreq(check_point_values, numbins=NUM_BIN)

    x = res.lowerlimit + np.linspace(res.binsize*res.cumcount.size, 0, res.cumcount.size)

    fig = plt.figure(figsize=(10, 4))
    fig.suptitle(CHECK_POINT[0])
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.hist(check_point_values, bins=NUM_BIN)
    ax1.set_title('Histogram')
    ax2.bar(x, res.cumcount, width=res.binsize)
    ax2.set_title('Cumulative histogram')
    ax2.set_xlim([x.min(), x.max()])

    plt.show()

if __name__ == "__main__":
    main()
