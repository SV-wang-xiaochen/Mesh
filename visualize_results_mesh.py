import matplotlib.pyplot as plt
import numpy as np

a = np.load('mesh_results/0.8.npy')
x = a[0]
y = a[1]
z = a[2]-13.0439425
colors = a[3]
colors_new = np.where(colors < 3, 255, colors)

print(np.sum(colors))
print(colors)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the points
sc = ax.scatter(x, y, z, c=colors, cmap='Reds', marker='o', s=80)

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Create a colorbar and set label
cbar = plt.colorbar(sc)
cbar.set_label('Color Value')

plt.show()