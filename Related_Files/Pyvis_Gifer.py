import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import re
import os
import glob

# === CONFIGURATION ===
directory = r"C:/Users/hanst/Desktop/DTU/4 semester/Softwareprojekt/Expi"
output_path = r"C:/Users/hanst/Desktop/ocean_animation.gif"

# === PARSING FUNCTION ===
def parse_file(filename):
    data_list = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            row = []
            for part in parts:
                try:
                    value = float(part)
                except ValueError:
                    match = re.match(r'(\d+\.\d+)([+-]\d+)', part)
                    if match:
                        value = float(f"{match.group(1)}E{match.group(2)}")
                    else:
                        value = 0.0
                row.append(value)
            if len(row) == 4:
                data_list.append(row)
    return np.array(data_list)

# === FILE COLLECTION ===
file_pattern = os.path.join(directory, "fort.1*")
files = sorted(glob.glob(file_pattern), key=lambda x: int(x.split('.')[-1]))
files = [f for f in files if '.123' not in f]
print(f"Found {len(files)} files to process.")

first_data = parse_file(files[0])
x_values = first_data[:, 0]
y_values = first_data[:, 1]
unique_x = np.unique(x_values)
unique_y = np.unique(y_values)
nx = len(unique_x)
ny = len(unique_y)
rows = first_data.shape[0]
can_reshape = (nx * ny == rows)

if can_reshape:
    X, Y = np.meshgrid(unique_x, unique_y)
    grid_shape = (ny, nx)
else:
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
    try:
        Z_test = first_data[:, 2].reshape(ny, nx)
        grid_shape = (ny, nx)
    except ValueError:
        print("Can't reshape to grid. Using raw data points instead.")
        X = x_values
        Y = y_values
        grid_shape = None

# === INITIAL PLOT SETUP ===
plt.rcParams['figure.figsize'] = [12, 8]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=30, azim=-60)

if grid_shape:
    Z_initial = first_data[:, 2].reshape(grid_shape)
    surf = ax.plot_surface(X, Y, Z_initial, cmap='viridis', linewidth=0, antialiased=True, alpha=0.8)
else:
    surf = ax.scatter(X, Y, first_data[:, 2], c=first_data[:, 2], cmap='viridis')

fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Surface Elevation η')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('η')
ax.set_title(f'Ocean Surface Elevation - Frame 0')

# === GLOBAL Z LIMITS ===
z_mins, z_maxs = [], []
for file in files[:min(10, len(files))]:
    try:
        data = parse_file(file)
        z_mins.append(np.min(data[:, 2]))
        z_maxs.append(np.max(data[:, 2]))
    except Exception as e:
        print(f"Error sampling {file}: {e}")

z_min = min(z_mins) if z_mins else np.min(first_data[:, 2])
z_max = max(z_maxs) if z_maxs else np.max(first_data[:, 2])
z_range = z_max - z_min
z_min -= z_range * 0.1
z_max += z_range * 0.1
ax.set_zlim(z_min, z_max)

# === ANIMATION FUNCTION ===
def update(frame):
    ax.clear()
    try:
        data = parse_file(files[frame])
        if grid_shape:
            Z = data[:, 2].reshape(grid_shape)
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=True, alpha=0.8)
        else:
            surf = ax.scatter(X, Y, data[:, 2], c=data[:, 2], cmap='viridis')

        ax.view_init(elev=30, azim=-60)
        ax.set_zlim(z_min, z_max)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('η')
        ax.set_title(f'Ocean Surface Elevation - Frame {frame}\nFile: {os.path.basename(files[frame])}')
        ax.set_box_aspect([1, 1, 0.3])
        return [surf]
    except Exception as e:
        print(f"Error on frame {frame}, file {files[frame]}: {e}")
        return []

# === CREATE ANIMATION ===
ani = FuncAnimation(fig, update, frames=len(files), interval=200, blit=False)

# === SAVE TO GIF ===
print("Saving animation...")
ani.save(output_path, writer=PillowWriter(fps=5))
print(f"Saved to {output_path}")

# === OPEN THE FILE AFTERWARD ===
#os.startfile(output_path)  # Windows only
