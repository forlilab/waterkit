import os
import shutil
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from gridData import Grid

# frames = [100, 200, 300, 400, 500, 800, 1000, 2000, 3000, 4000, 5000, 8000, 10000, 12000, 15000, 20000]
n_steps = [10000, 50000, 100000, 200000, 400000, 500000, 800000]
# n_steps = [1000, 5000, 10000, 20000, 40000, 50000, 70000, 80000, 90000, 100000, 200000]

def load_grids(path_to_grid_files):
    grid_gO = Grid(os.path.join(path_to_grid_files, 'gist-gO.dx'))
    grid_esw = Grid(os.path.join(path_to_grid_files, 'gist-Esw-dens.dx'))
    grid_eww = Grid(os.path.join(path_to_grid_files, 'gist-Eww-dens.dx'))
    grid_tst = Grid(os.path.join(path_to_grid_files, 'gist-dTStrans-dens.dx'))
    grid_tso = Grid(os.path.join(path_to_grid_files, 'gist-dTSorient-dens.dx'))
    return [grid_gO, grid_esw, grid_eww, grid_tst, grid_tso]

def select_voxels_within_distance(grid, max_distance=5.0):
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
    path_to_frames = sys.argv[1]

    grid_types = ['gO', 'Esw', 'Eww', 'TSt', 'TSo']
    # tanimotos_for_plot = {"gO": [],
    #                       "Esw": [],
    #                       "Eww": [],
    #                       "TSt": [],
    #                       "TSo": []}
    
    # grid_types = ['gO', 'Esw', 'Eww']
    # tanimotos_for_plot = {"gO": [], 
    #                       "Esw": [], 
    #                       "Eww": []}
    
    # for n_frames in frames:
    for rep in range(0, 3):
        tanimotos_for_plot = {"gO": [],
                          "Esw": [],
                          "Eww": [],
                          "TSt": [],
                          "TSo": []}
        for n_step in n_steps:
            path_to_grids_MC = os.path.join(path_to_frames, "TIP3P", "GCMC_STEPS", f"{n_step}_steps")
            path_to_grids_MD = os.path.join(path_to_frames, "GIST_MD_NO_HMR", f"gist_rep{rep+1}")
            grids_1 = load_grids(path_to_grids_MC)
            grids_2 = load_grids(path_to_grids_MD)
            for idx, grid_type in enumerate(grid_types):
                print(f"Analyzing grid: {grid_type}")
                densities_1 = select_voxels_within_distance(grids_1[idx]).flatten()
                densities_2 = select_voxels_within_distance(grids_2[idx]).flatten()
                smoothed_d1 = gaussian_filter(densities_1, sigma=3)
                smoothed_d2 = gaussian_filter(densities_2, sigma=3)
                t_similarity, t_distance = compute_tanimoto(smoothed_d1, smoothed_d2)
                # print(f"Tanimoto Similarity: {t_similarity}\nTanimoto Distance: {t_distance}")
                tanimotos_for_plot[grid_type].append(t_similarity)
        
        for grid_type in tanimotos_for_plot:
            print(n_step, tanimotos_for_plot[grid_type])
            plot_name = f"{grid_type}_tanimoto_{rep+1}.png"
            plt.figure(figsize=(8, 8))
            plt.plot(n_steps, 
                    tanimotos_for_plot[grid_type], 
                    alpha=0.5)  # s=10 for smaller points
            plt.xlabel(f"# of Steps")
            plt.ylabel(f"Non-Binary Tanimoto Similarity")
            plt.title("Distribution of Tanimoto Similarity amongst the frames")
            plt.legend()

            # Optional: Set equal aspect ratio for better comparison
            # plt.axis('equal')
            plt.savefig(f"{plot_name}")
            plt.clf()





