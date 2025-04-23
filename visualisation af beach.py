import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import re
import os
import glob

# Directory path
directory = r"C:/Users/alans/OneDrive - Danmarks Tekniske Universitet/4. Semester\Software Projekt/OceanWave3D-Fortran90-botp/docker/Whalin"

# Function to parse a single file with our custom scientific notation handler
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

# Get all fort.* files in order
file_pattern = os.path.join(directory, "fort.1*")
files = sorted(glob.glob(file_pattern), 
               key=lambda x: int(x.split('.')[-1]))
files = [f for f in files if '.123' not in f]
print(f"Found {len(files)} files to process.")

# Read the first file to understand the data structure
first_data = parse_file(files[0])
rows = first_data.shape[0]

# Extract x and y values from the first two columns
x_values = first_data[:, 0]
y_values = first_data[:, 1]

# Find unique values to determine the grid structure
unique_x = np.unique(x_values)
unique_y = np.unique(y_values)
nx = len(unique_x)
ny = len(unique_y)

# Check if we can reshape correctly
print(f"Data rows: {rows}, Unique X: {nx}, Unique Y: {ny}, Product: {nx*ny}")
can_reshape = (nx * ny == rows)

if can_reshape:
    # Create meshgrid for proper surface plotting
    X, Y = np.meshgrid(unique_x, unique_y)
    grid_shape = (ny, nx)
else:
    # Fallback if we can't determine the proper grid
    print("Warning: Can't determine proper grid shape. Using alternative approach.")
    # Create a simple grid based on data points
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
    # This reshape might not work - depends on how data is organized
    try:
        Z_test = first_data[:, 2].reshape(ny, nx)
        grid_shape = (ny, nx)
    except ValueError:
        print("Can't reshape to grid. Using raw data points instead.")
        X = x_values
        Y = y_values
        grid_shape = None

# Create figure with better controls
plt.rcParams['figure.figsize'] = [12, 8]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set a good default view angle
ax.view_init(elev=30, azim=-60)

# Initialize the plot object
if grid_shape:
    # Try to use surface plot if we have a grid
    Z_initial = first_data[:, 2].reshape(grid_shape)
    surf = ax.plot_surface(X, Y, Z_initial, cmap='viridis', 
                          linewidth=0, antialiased=True, alpha=0.8)
else:
    # Fallback to scatter plot for unstructured data
    surf = ax.scatter(X, Y, first_data[:, 2], c=first_data[:, 2], cmap='viridis')

# Add a color bar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Surface Elevation η')

# Set labels and adjust display
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('η')
ax.set_title(f'Ocean Surface Elevation - Frame 0')

# Find global min/max for consistent z scaling
z_mins = []
z_maxs = []
for file in files[:min(10, len(files))]:  # Sample files for min/max
    try:
        data = parse_file(file)
        z_mins.append(np.min(data[:, 2]))
        z_maxs.append(np.max(data[:, 2]))
    except Exception as e:
        print(f"Error sampling {file}: {e}")

z_min = min(z_mins) if z_mins else np.min(first_data[:, 2])
z_max = max(z_maxs) if z_maxs else np.max(first_data[:, 2])
z_range = z_max - z_min
z_min -= z_range * 0.1  # Add 10% margin
z_max += z_range * 0.1
ax.set_zlim(z_min, z_max)

# Animation update function
def update(frame):
    # Clear previous surface
    ax.clear()
    
    try:
        # Load data for this frame
        data = parse_file(files[frame])
        
        # Create the plot based on earlier determined approach
        if grid_shape:
            Z = data[:, 2].reshape(grid_shape)
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', 
                                  linewidth=0, antialiased=True, alpha=0.8)
        else:
            surf = ax.scatter(X, Y, data[:, 2], c=data[:, 2], cmap='viridis')
        
        # Reset view settings
        ax.view_init(elev=30, azim=-60)
        ax.set_zlim(z_min, z_max)
        
        # Add labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('η')
        ax.set_title(f'Ocean Surface Elevation - Frame {frame}\nFile: {os.path.basename(files[frame])}')
        
        # Keep a reasonable aspect ratio
        ax.set_box_aspect([1, 1, 0.3])
        
        return [surf]
    except Exception as e:
        print(f"Error on frame {frame}, file {files[frame]}: {e}")
        return []

# Create animation
ani = FuncAnimation(fig, update, frames=len(files), 
                    interval=200, blit=False)

plt.tight_layout()
plt.show()

# Uncomment to save the animation
# ani.save('ocean_animation.mp4', writer='ffmpeg', fps=5, dpi=300)