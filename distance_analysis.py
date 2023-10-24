import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import glob
import os

NUM_BIN = 50

def main():
    path = r'C:\Users\xiaochen.wang\Projects\Dataset\FLORENCE'
    obj_list = glob.glob(f'{path}/**/*.obj', recursive=True)

    mesh_plot_list = []

    max_z_list = []

    for mesh_nr in range(0,len(obj_list)):

        print("Source mesh:")
        print(mesh_nr, os.path.basename(obj_list[mesh_nr]), '\n')

        mesh = o3d.io.read_triangle_mesh(obj_list[mesh_nr])

        mesh_plot_list.append(mesh)

        max_z = np.max(np.asarray(mesh.vertices)[:,2])

        max_z_list.append(max_z)

    # o3d.visualization.draw_geometries(mesh_plot_list, mesh_show_wireframe=True)
    print(sorted(max_z_list))

    # # Create a histogram
    # plt.hist(max_z_list, bins=50, color='skyblue', edgecolor='black')
    #
    # # Add labels and a title
    # plt.xlabel('Max Z Value')
    # plt.ylabel('Frequency')
    # plt.title('Histogram')
    #
    # # Show the plot
    # plt.show()

    # # Create a cumulative frequency histogram
    res = stats.cumfreq(max_z_list, numbins=NUM_BIN)

    x = res.lowerlimit + np.linspace(res.binsize*res.cumcount.size, 0, res.cumcount.size)

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.hist(max_z_list, bins=NUM_BIN)
    ax1.set_title('Histogram')
    ax2.bar(x, res.cumcount, width=res.binsize)
    ax2.set_title('Cumulative histogram')
    ax2.set_xlim([x.min(), x.max()])

    plt.show()

if __name__ == "__main__":
    main()
