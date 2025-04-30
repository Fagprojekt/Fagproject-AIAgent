import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
# Axes3D potentially imported later if needed
# from mpl_toolkits.mplot3d import Axes3D
import re
import os
import glob
# Needed for joining paths and getting CWD if saving relative to script location
import os

# Directory path (ensure this is correct for your system)
# Placeholder - Update this path
directory = r"Beach2"


# Function to parse a single file with custom scientific notation handler
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
                    match = re.match(r'(-?\d+\.\d+)([+-]\d+)', part)
                    if match:
                        try:
                            value = float(f"{match.group(1)}E{match.group(2)}")
                        except ValueError:
                             print(f"Warning: Could not parse custom notation '{part}' in {filename}. Using 0.0.")
                             value = 0.0
                    else:
                         print(f"Warning: Could not parse value '{part}' in {filename}. Using 0.0.")
                         value = 0.0
                row.append(value)

            if len(row) == 4:
                data_list.append(row)
            elif len(row) > 0:
                 print(f"Warning: Row in {filename} has {len(row)} columns, expected 4. Skipping row: {line.strip()}")

    if not data_list:
         print(f"Warning: No valid data found in {filename}.")
         return np.empty((0, 4))
    return np.array(data_list)

# Get all fort.* files in order
file_pattern = os.path.join(directory, "fort.1*")
if not os.path.isdir(directory):
    print(f"Error: Directory not found: {directory}")
    # Consider exiting or handling this error appropriately
    exit() # Exit if data directory is crucial

files = sorted(glob.glob(file_pattern),
               key=lambda x: int(x.split('.')[-1]))
files = [f for f in files if '.123' not in os.path.basename(f)]

if not files:
    print(f"Error: No files matching pattern '{file_pattern}' found in {directory}")
    # Consider exiting or handling this error appropriately
    exit() # Exit if no data files found

print(f"Found {len(files)} files to process.")

# Read the first file to understand the data structure
try:
    first_data = parse_file(files[0])
    if first_data.shape[0] == 0:
        print(f"Error: First file {files[0]} contains no valid data.")
        exit()
except FileNotFoundError:
    print(f"Error: First file {files[0]} not found.")
    exit()
except Exception as e:
    print(f"Error parsing first file {files[0]}: {e}")
    exit()

rows = first_data.shape[0]

if first_data.shape[1] < 3:
    print(f"Error: Data in {files[0]} has fewer than 3 columns (x, y, z required). Shape: {first_data.shape}")
    exit()

x_values = first_data[:, 0]
y_values = first_data[:, 1]
z_values_initial = first_data[:, 2]

unique_x = np.unique(x_values)
unique_y = np.unique(y_values)
nx = len(unique_x)
ny = len(unique_y)

is_2d = (ny == 1)

plt.rcParams['figure.figsize'] = [12, 8]
fig = plt.figure()
grid_shape = None # Initialize grid_shape
X, Y = None, None  # Initialize X, Y

if is_2d:
    ax = fig.add_subplot(111)
    print(f"Detected 2D data (nx={nx}, ny={ny}). Creating 2D line plot.")
    sort_indices = np.argsort(x_values)
    x_values_sorted = x_values[sort_indices]
    z_values_initial_sorted = z_values_initial[sort_indices]
else:
    from mpl_toolkits.mplot3d import Axes3D
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=-60)
    print(f"Detected 3D data (nx={nx}, ny={ny}). Creating 3D plot.")
    can_reshape = (nx * ny == rows)
    if can_reshape:
        print("Data appears grid-like. Using plot_surface.")
        try:
            Z_test_reshape = z_values_initial.reshape((ny, nx))
            X, Y = np.meshgrid(unique_x, unique_y)
            grid_shape = (ny, nx)
        except ValueError:
             print("Warning: Automatic reshape failed. Data might not be perfectly ordered for meshgrid.")
             can_reshape = False
             grid_shape = None
             X = x_values
             Y = y_values
    else:
        print("Warning: Data points don't match unique X/Y product (nx*ny != rows). Using 3D scatter plot.")
        grid_shape = None
        X = x_values
        Y = y_values


# --- Find global min/max Z (η) for consistent scaling ---
z_mins = []
z_maxs = []
print("Sampling files to determine consistent Z-axis limits...")
sample_size = min(20, len(files))
indices_to_sample = np.linspace(0, len(files) - 1, sample_size, dtype=int)

for i in indices_to_sample:
    file = files[i]
    try:
        data = parse_file(file)
        if data.shape[0] > 0 and data.shape[1] >= 3:
            z_mins.append(np.min(data[:, 2]))
            z_maxs.append(np.max(data[:, 2]))
        else:
             print(f"Warning: Skipping file {file} for Z-limit calculation due to invalid data.")
    except Exception as e:
        print(f"Error sampling {file} for Z-limits: {e}")

if not z_mins or not z_maxs:
     print("Warning: Could not determine Z-limits from sampled files. Using first frame only.")
     z_min_val = np.min(z_values_initial)
     z_max_val = np.max(z_values_initial)
else:
    z_min_val = min(z_mins)
    z_max_val = max(z_maxs)

z_range = z_max_val - z_min_val
if z_range < 1e-6:
    z_margin = 0.5
else:
    z_margin = z_range * 0.1

z_min_limit = z_min_val - z_margin
z_max_limit = z_max_val + z_margin
print(f"Determined Z limits: [{z_min_limit:.2f}, {z_max_limit:.2f}]")


# --- Initialize Plot ---
plot_object = None

if is_2d:
    line, = ax.plot(x_values_sorted, z_values_initial_sorted)
    plot_object = line
    ax.set_xlabel('X')
    ax.set_ylabel('Surface Elevation η')
    ax.set_title(f'Ocean Surface Elevation (2D) - Frame 0\nFile: {os.path.basename(files[0])}')
    ax.set_ylim(z_min_limit, z_max_limit)
    ax.grid(True)
else:
    if grid_shape:
        Z_initial_grid = z_values_initial.reshape(grid_shape)
        surf = ax.plot_surface(X, Y, Z_initial_grid, cmap='viridis', linewidth=0, antialiased=True, alpha=0.8)
        plot_object = surf
    else:
        scatter = ax.scatter(X, Y, z_values_initial, c=z_values_initial, cmap='viridis')
        plot_object = scatter

    try:
        mappable = plot_object
        fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label='Surface Elevation η')
    except Exception as e:
        print(f"Could not add colorbar: {e}")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('η')
    ax.set_title(f'Ocean Surface Elevation (3D) - Frame 0\nFile: {os.path.basename(files[0])}')
    ax.set_zlim(z_min_limit, z_max_limit)
    # Calculate aspect ratio based on data ranges for better visualization
    aspect_ratio = [1, 1, 1] # Default
    if X is not None and Y is not None and nx > 1 and ny > 1 :
        x_range = np.ptp(unique_x) if unique_x.size > 1 else 1
        y_range = np.ptp(unique_y) if unique_y.size > 1 else 1
        z_data_range = z_max_limit - z_min_limit
        # Make z-axis visually smaller relative to x/y if its range is large
        aspect_ratio = [x_range, y_range, z_data_range * 0.4] # Adjust 0.4 factor as needed

    # Normalize aspect ratio while keeping relative proportions
    # max_dim = max(aspect_ratio)
    # normalized_aspect = [dim / max_dim for dim in aspect_ratio]
    # ax.set_box_aspect(normalized_aspect)
    # Or simpler fixed aspect:
    ax.set_box_aspect([1, (unique_y[-1]-unique_y[0])/(unique_x[-1]-unique_x[0]) if nx>1 and ny > 1 else 1, 0.3])


# --- Animation Update Function ---
def update(frame):
    # Removed 'global plot_object' as we return artists or redraw anyway
    current_file = files[frame]
    print(f"\rProcessing frame {frame+1}/{len(files)}: {os.path.basename(current_file)}", end='')

    try:
        data = parse_file(current_file)
        if data.shape[0] == 0 or data.shape[1] < 3:
             print(f"\nWarning: Skipping frame {frame} due to invalid data in {current_file}.")
             return [] # Return empty list

        x_frame = data[:, 0]
        z_frame = data[:, 2]

        if is_2d:
            sort_indices_frame = np.argsort(x_frame)
            x_frame_sorted = x_frame[sort_indices_frame]
            z_frame_sorted = z_frame[sort_indices_frame]

            # Update the line data directly
            plot_object.set_data(x_frame_sorted, z_frame_sorted)
            # Update the title - THIS WILL NOW WORK because blit=False forces redraw
            ax.set_title(f'Ocean Surface Elevation (2D) - Frame {frame}\nFile: {os.path.basename(current_file)}')
            # Need to ensure axes limits stay fixed if desired, set_data doesn't auto-scale
            ax.set_ylim(z_min_limit, z_max_limit) # Re-apply limits just in case
            # Return empty list when blit=False
            return [] # Changed from [plot_object]

        else: # Update 3D plot
            ax.clear()
            current_plot_object = None

            if grid_shape:
                try:
                    Z_frame_grid = z_frame.reshape(grid_shape)
                    current_plot_object = ax.plot_surface(X, Y, Z_frame_grid, cmap='viridis', linewidth=0, antialiased=True, alpha=0.8)
                except ValueError:
                    print(f"\nWarning: Reshape failed for frame {frame}. Check data consistency.")
                    pass
            else:
                y_frame = data[:, 1]
                current_plot_object = ax.scatter(x_frame, y_frame, z_frame, c=z_frame, cmap='viridis')

            # Reset all static plot elements after clearing
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('η')
            ax.set_zlim(z_min_limit, z_max_limit)
            ax.view_init(elev=30, azim=-60)
            ax.set_title(f'Ocean Surface Elevation (3D) - Frame {frame}\nFile: {os.path.basename(current_file)}')
            ax.set_box_aspect([1, (unique_y[-1]-unique_y[0])/(unique_x[-1]-unique_x[0]) if nx>1 and ny > 1 else 1, 0.3])

            # Return empty list when blit=False
            return [] # Changed from potential artist list


    except Exception as e:
        import traceback
        print(f"\nError processing frame {frame}, file {current_file}:")
        traceback.print_exc()
        return []

# --- Create Animation ---
# Disable blitting to ensure title updates correctly in both 2D and 3D
use_blit = False # <<< THE FIX IS HERE

ani = FuncAnimation(fig, update, frames=len(files),
                    interval=100,
                    blit=use_blit, # Now always False
                    repeat=True)

print("\nAnimation object created. Displaying plot window...")
plt.tight_layout()
plt.show()


# --- Save Animation (Optional) ---
save_animation = True # <<< Set to True to save GIF

if save_animation:
    output_filename = 'ocean_animation.gif'
    output_filepath = output_filename # Saves to CWD

    print(f"\nAttempting to save animation as GIF to {output_filepath}...")
    print("This requires the 'Pillow' library. Install with: pip install Pillow")
    try:
        ani.save(output_filepath, writer='pillow', fps=10, dpi=150,
                 progress_callback=lambda i, n: print(f'\rSaving frame {i+1}/{n}...', end=''))
        print(f"\nAnimation saved successfully to {output_filepath}")
    except Exception as e:
        print(f"\nError saving animation: {e}")
        print("Please ensure 'Pillow' is installed (`pip install Pillow`) and system dependencies (if any for Pillow) are met.")
        import traceback
        traceback.print_exc()