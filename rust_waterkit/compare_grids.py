import sys
import os

from gridData import Grid
import numpy as np
import matplotlib.pyplot as plt



def load_grids(path_to_grid_files):
    grid_gO = Grid(os.path.join(path_to_grid_files, 'gist-gO.dx'))
    grid_esw = Grid(os.path.join(path_to_grid_files, 'gist-Esw-dens.dx'))
    grid_eww = Grid(os.path.join(path_to_grid_files, 'gist-Eww-dens.dx'))
    grid_tst = Grid(os.path.join(path_to_grid_files, 'gist-dTStrans-dens.dx'))
    grid_tso = Grid(os.path.join(path_to_grid_files, 'gist-dTSorient-dens.dx'))
    return [grid_gO, grid_esw, grid_eww, grid_tst, grid_tso]

def select_voxels_within_distance(grid, max_distance=10.0):
    """
    Select voxels in a GIST grid within a specified distance from the origin.
    
    Parameters:
    grid_path (str): Path to the .dx file.
    max_distance (float): Maximum distance from origin in Ångströms (default: 5.0).
    
    Returns:
    tuple: (selected_values, selected_indices, selected_coords)
        - selected_values: Array of voxel values within max_distance.
        - selected_indices: Array of voxel indices (i, j, k).
        - selected_coords: Array of voxel coordinates (x, y, z).
    """
    # Get grid properties
    origin = grid.origin  # (x0, y0, z0)
    delta = grid.delta  # Spacing (dx, dy, dz); assuming uniform grid for simplicity
    grid_shape = grid.grid.shape  # (nx, ny, nz)
    
    # Generate voxel indices
    i, j, k = np.indices(grid_shape)
    
    # Compute voxel coordinates
    # For a uniform grid, delta is a scalar (or same for all axes)
    dx = delta[0]  # Assuming cubic grid (dx = dy = dz)
    coords_x = origin[0] + i * dx
    coords_y = origin[1] + j * dx
    coords_z = origin[2] + k * dx
    
    # Stack coordinates into (nx, ny, nz, 3) array
    coords = np.stack([coords_x, coords_y, coords_z], axis=-1)
    
    # Compute Euclidean distances from origin
    distances = np.sqrt(np.sum((coords - origin)**2, axis=-1))
    
    # Select voxels within max_distance
    # max_distance = None
    if max_distance is not None:
        mask = distances <= max_distance
    else:
        mask = np.ones_like(distances, dtype=bool)
    selected_values = grid.grid[mask]
    
    return selected_values

def compare_grids(grid_1, grid_2, grid_type, x_axis, y_axis, plot_name, path):
    densities_1 = select_voxels_within_distance(grid_1).flatten()
    densities_2 = select_voxels_within_distance(grid_2).flatten()
    # densities_1 = grid_1.grid.flatten()
    # densities_2 = grid_2.grid.flatten()
    # Create scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(densities_1, densities_2, alpha=0.5, s=10)  # s=10 for smaller points
    plt.xlabel(f"Density {x_axis}")
    plt.ylabel(f"Density {y_axis}")
    plt.title("Voxel-by-Voxel Density Comparison")

    # Add y=x reference line
    max_val = max(np.max(densities_1), np.max(densities_2))
    min_val = min(np.min(densities_1), np.min(densities_2))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="y=x")
    plt.legend()

    # Optional: Set equal aspect ratio for better comparison
    plt.axis('equal')
    plt.savefig(f"{path}/plot_{grid_type}_{plot_name}.png")
    # plt.show()
    plt.clf()
    return densities_1, densities_2

def compute_tanimoto(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1_sq = np.dot(vec1, vec1)
    norm2_sq = np.dot(vec2, vec2)
    
    # Check for zero denominator
    denominator = norm1_sq + norm2_sq - dot_product
    if denominator == 0:
        raise ValueError("Denominator is zero (grids may be zero vectors or identical)")
    
    # Compute Tanimoto similarity
    similarity = dot_product / denominator
    
    # Compute Tanimoto distance
    distance = 1 - similarity
    return similarity, distance

if __name__ == "__main__":
    path_to_grid_files_1 = sys.argv[1]
    path_to_grid_files_2 = sys.argv[2]
    x_axis = sys.argv[3]
    y_axis = sys.argv[4]
    plot_name = sys.argv[5]
    grid_types = ['gO', 'Esw', 'Eww', 'TSt', 'TSo']
    # grid_types = ['gO']
    grids_1 = load_grids(path_to_grid_files_1)
    grids_2 = load_grids(path_to_grid_files_2)
    
    for idx, grid_type in enumerate(grid_types):
        print(f"Analyzing grid: {grid_type}")
        d1, d2 = compare_grids(grids_1[idx], grids_2[idx], grid_type, x_axis=x_axis, y_axis=y_axis, plot_name=plot_name, path=path_to_grid_files_2)
        t_similarity, t_distance = compute_tanimoto(d1, d2)
        print(f"Tanimoto Similarity: {t_similarity}\nTanimoto Distance: {t_distance}")





