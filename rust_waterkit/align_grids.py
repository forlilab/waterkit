import sys
import os
import numpy as np
from scipy.linalg import svd
from scipy.ndimage import map_coordinates
from gridData import Grid

def compute_centroid(grid_data, origin, delta):
    """
    Compute the intensity-weighted centroid of a grid.
    
    Args:
        grid_data (np.ndarray): 3D grid data.
        origin (np.ndarray): Grid origin (x, y, z).
        delta (np.ndarray): Grid spacing (dx, dy, dz).
    
    Returns:
        np.ndarray: Centroid coordinates (x, y, z) in world coordinates.
    """
    # Get voxel coordinates in grid indices
    coords = np.indices(grid_data.shape).reshape(3, -1).T  # Shape: (nx*ny*nz, 3)
    weights = grid_data.flatten()
    
    # Compute weighted centroid in grid index space
    if np.sum(weights) > 0:
        centroid_idx = np.average(coords, axis=0, weights=weights)
    else:
        # Fallback to geometric center if weights are all zero
        centroid_idx = np.mean(coords, axis=0)
    
    # Convert to world coordinates
    if delta.ndim == 1:
        delta = np.diag(delta)
    centroid = centroid_idx @ delta + origin
    return centroid

def align_grids(grid1_path, grid2_path, output_path):
    """
    Align Grid 1 to Grid 2 using computed centroids for translation and SVD for rotation.
    Save the aligned Grid 1 to a new .dx file with Grid 2's metadata.

    Args:
        grid1_path (str): Path to the first grid (.dx file) to be aligned.
        grid2_path (str): Path to the second grid (.dx file) to align to.
        output_path (str): Path to save the aligned Grid 1 (.dx file).

    Returns:
        tuple: Rotation matrix (3x3 numpy array) and translation vector (3x1 numpy array).

    Raises:
        ValueError: If grids cannot be loaded or have mismatched shapes.
        OSError: If input files are inaccessible or output path is not writable.
    """
    # Step 1: Validate input and output paths
    for path in [grid1_path, grid2_path]:
        if not os.path.isfile(path):
            raise OSError(f"Input file does not exist: {path}")
    output_dir = os.path.dirname(output_path) or '.'
    if not os.path.exists(output_dir):
        raise OSError(f"Output directory does not exist: {output_dir}")
    if not os.access(output_dir, os.W_OK):
        raise OSError(f"Output directory is not writable: {output_dir}")

    # Step 2: Load grids using gridData.Grid
    try:
        grid1 = Grid(grid1_path)
        grid2 = Grid(grid2_path)
    except Exception as e:
        raise ValueError(f"Failed to load grid files. Grid1: {grid1_path}, Grid2: {grid2_path}. Error: {e}")

    # Step 3: Extract grid data and metadata
    try:
        grid1_data = grid1.grid
        grid2_data = grid2.grid
        origin1 = grid1.origin
        delta1 = grid1.delta
        origin2 = grid2.origin
        delta2 = grid2.delta
    except AttributeError as e:
        raise ValueError(f"Failed to access grid data or metadata: {e}")

    # Step 4: Verify grids have the same shape
    if grid1_data.shape != grid2_data.shape:
        raise ValueError(f"Grids must have the same dimensions. Grid1 shape: {grid1_data.shape}, Grid2 shape: {grid2_data.shape}")

    # Debugging: Print grid metadata
    print(f"Grid1 shape: {grid1_data.shape}, Origin: {grid1.origin}, Delta: {grid1.delta}")
    print(f"Grid2 shape: {grid2_data.shape}, Origin: {grid2.origin}, Delta: {grid2.delta}")

    # Step 5: Compute centroids
    c1 = compute_centroid(grid1_data, origin1, delta1)
    c2 = compute_centroid(grid2_data, origin2, delta2)
    print(f"Grid1 centroid: {c1}")
    print(f"Grid2 centroid: {c2}")

    # Step 6: Handle delta format for both grids
    if delta1.ndim == 1:
        delta1 = np.diag(delta1)
    elif delta1.shape != (3, 3):
        raise ValueError(f"Unexpected delta1 format: {delta1}. Expected 3x3 matrix or 3-element array.")
    if delta2.ndim == 1:
        delta2 = np.diag(delta2)
    elif delta2.shape != (3, 3):
        raise ValueError(f"Unexpected delta2 format: {delta2}. Expected 3x3 matrix or 3-element array.")

    # Step 7: Get grid coordinates for both grids
    nx, ny, nz = grid2_data.shape
    x1 = origin1[0] + np.arange(nx) * delta1[0, 0]
    y1 = origin1[1] + np.arange(ny) * delta1[1, 1]
    z1 = origin1[2] + np.arange(nz) * delta1[2, 2]
    X1, Y1, Z1 = np.meshgrid(x1, y1, z1, indexing='ij')
    coords1 = np.stack([X1, Y1, Z1], axis=-1).reshape(-1, 3)  # Grid 1 coordinates

    x2 = origin2[0] + np.arange(nx) * delta2[0, 0]
    y2 = origin2[1] + np.arange(ny) * delta2[1, 1]
    z2 = origin2[2] + np.arange(nz) * delta2[2, 2]
    X2, Y2, Z2 = np.meshgrid(x2, y2, z2, indexing='ij')
    coords2 = np.stack([X2, Y2, Z2], axis=-1).reshape(-1, 3)  # Grid 2 coordinates

    # Step 8: Compute translation (move Grid 1's centroid to Grid 2's centroid)
    translation = c2 - c1
    print(f"Translation vector: {translation}")

    # Step 9: Center point clouds for rotation calculation
    weights = grid2_data.flatten()  # Use Grid 2 intensities for weighting
    coords1_centered = coords1 - c1
    coords2_centered = coords2 - c2

    # Step 10: Compute rotation using SVD
    H = np.zeros((3, 3))
    for i in range(len(coords1)):
        H += weights[i] * np.outer(coords1_centered[i], coords2_centered[i])
    H /= np.sum(weights) if np.sum(weights) > 0 else 1.0

    U, _, Vt = svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:  # Correct for reflection
        Vt[:, -1] *= -1
        R = Vt.T @ U.T
    print(f"Rotation matrix:\n{R}")

    # Step 11: Apply transformation to Grid 1 coordinates
    transformed_coords = (R @ coords1_centered.T).T + c2

    # Step 12: Interpolate Grid 1 data onto Grid 2's grid
    inv_delta2 = np.linalg.inv(delta2)
    grid_coords = ((transformed_coords - origin2) @ inv_delta2).reshape(nx, ny, nz, 3)
    transformed_grid_data = map_coordinates(grid1_data, grid_coords.transpose(3, 0, 1, 2), order=1, mode='nearest')

    # Step 13: Save the transformed grid with Grid 2's metadata
    try:
        transformed_grid = Grid(transformed_grid_data, origin=grid2.origin, delta=grid2.delta)
        transformed_grid.export(output_path, file_format='DX')
    except Exception as e:
        raise OSError(f"Failed to save transformed grid to {output_path}: {e}")

    print(f"Success! Transformed grid saved to: {output_path}")
    return R, translation

# Example usage
if __name__ == "__main__":
    path_to_grid1 = sys.argv[1]
    path_to_grid2 = sys.argv[2]
    name_modified_grid = path_to_grid1.split("/")[-1].split(".dx")[0] + "_aligned.dx"
    path_to_modified_grid = "/".join(path_to_grid1.split("/")[0:-1])
    print(path_to_modified_grid)
    if len(path_to_grid1.split("/")[0:-1]) > 0:
        output_path = f"{path_to_modified_grid}/{name_modified_grid}"
    else:
        output_path = f"{name_modified_grid}"
    rotation, translation = align_grids(path_to_grid1, path_to_grid2, output_path)
    print(f"Rotation matrix:\n{rotation}")
    print(f"Translation vector: {translation}")
    
    