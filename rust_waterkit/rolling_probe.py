import numpy as np
from scipy.spatial import KDTree

def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt(np.sum((p1 - p2) ** 2))

def is_accessible(probe_center, surface_points, probe_radius):
    """
    Check if a probe sphere is accessible at a given position.
    A position is accessible if the sphere does not overlap with any surface atom.
    """
    for point in surface_points:
        if distance(probe_center, point) < probe_radius:
            return False
    return True

def find_valid_start(point, probe_radius, other_points):
    directions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], 
                           [-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    for direction in directions:
        probe_center = point + direction * probe_radius
        if is_accessible(probe_center, probe_radius, other_points):
            return probe_center
    # As a fallback, use random sampling
    for _ in range(100):
        offset = np.random.uniform(-probe_radius, probe_radius, size=3)
        probe_center = point + offset
        if is_accessible(probe_center, probe_radius, other_points):
            return probe_center
    raise ValueError("No valid starting position found.")

def roll_sphere(surface_points, probe_radius, step_size):
    """
    Roll a sphere on a surface defined by 3D points.
    
    Parameters:
    - surface_points: np.ndarray of shape (N, 3), representing the 3D coordinates of the surface atoms.
    - probe_radius: Radius of the probe sphere.
    - step_size: Distance to move the probe at each step.
    
    Returns:
    - sampled_points: List of 3D points representing the probe's positions.
    """
    # Build a KDTree for efficient nearest-neighbor queries
    tree = KDTree(surface_points)

    # Initialize the list of sampled points
    sampled_points = []

    # Iterate over all surface points
    for i, point in enumerate(surface_points):
        # Start the probe at the surface point
        probe_center = point + np.array([0, 0, 0])
        # Check if the probe is accessible at the initial position
        if is_accessible(probe_center, surface_points, probe_radius):
            sampled_points.append(probe_center)

        # Roll the sphere by moving it in a grid-like manner around the initial point
        for dx in np.arange(-step_size, step_size, 0.5):
            for dy in np.arange(-step_size, step_size, 0.5):
                for dz in np.arange(-step_size, step_size, 0.5):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    candidate_position = probe_center + np.array([dx, dy, dz])
                    # print(candidate_position)
                    # Check if the new position is accessible
                    if is_accessible(candidate_position, surface_points, probe_radius):
                        sampled_points.append(candidate_position)

    return np.array(sampled_points)

def to_xyz(traj):
    with open("traj_python.xyz", 'w') as fo:
        fo.write(f"{len(traj)}\n")
        fo.write("\n")
        for t in traj:
            fo.write(f"He {t[0]} {t[1]} {t[2]}\n")
    return 

if __name__ == "__main__":
    import time
    # select as, i. 111+107+103+162+150+98+97+184+96+93+55+52+51+138+139+136+135
    probe_radius = 1.4
    step_size = 1.2

    with open("/data/phd/waterkit/example/pocket.txt") as fi:
        lines = fi.readlines()
    
    surface_points = []
    for line in lines:
        line = line.strip().split(",")
        point = [float(line[3]), float(line[4]), float(line[5])]
        surface_points.append(point)
    
    start = time.time()
    trajectories = roll_sphere(surface_points, probe_radius, step_size)
    print(f"Time to grid python: {time.time() - start}")
    to_xyz(trajectories)