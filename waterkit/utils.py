#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WaterKit
#
# Utils functions
#

import tempfile
import contextlib
import shutil
import copy
import errno
import os
import subprocess
import sys

if sys.version_info >= (3, ):
    import importlib
else:
    import imp

import numpy as np


def vector(a, b):
    """
    Return the vector between a and b
    """
    return b - a


def normalize(a):
    """
    Return a normalized vector
    """
    return a / np.sqrt(np.sum(np.power(a, 2)))


def get_perpendicular_vector(u):
    """
    Return the perpendicular vector to u
    """
    v = normalize(np.random.rand(3))
    v = np.cross(u, v)

    return v


def get_euclidean_distance(a, b):
    """
    Return euclidean distance a (can be multiple coordinates) and b
    """
    return np.sqrt(np.sum(np.power(a - b, 2), axis=1))


def get_angle(a, b, c, degree=True):
    """
    Returm angle between a (can be multiple coordinates), b and c
    """
    ba = np.atleast_2d(vector(b, a))
    bc = vector(b, c)

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc))
    # Make sure values fit between -1 and 1 for arccos
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.arccos(cos_angle)

    if degree:
        return np.degrees(angle)

    return angle


def dihedral(p, degree=False):
    """Dihedral angle.

    Source:
    https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python"""
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]

    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that is fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)

    angle = np.arctan2(y, x)

    if degree:
        return np.degrees(angle)
    else:
        return angle


def get_rotation_matrix(a, b):
    """
    Return 3D rotation matrix between vectors a and b
    Sources:
    https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/897677
    """
    v = np.cross(b, a)
    c = np.dot(b, a)
    s = np.linalg.norm(v)
    I = np.identity(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    r = I + k + np.matmul(k, k) * ((1 - c) / (s**2))

    return r


def rotation_axis(p0, p1, p2, origin=None):
    """
    Compute rotation axis centered at the origin if not None
    """
    r = normalize(np.cross(vector(p1, p0), vector(p2, p0)))

    if origin is not None:
        return origin + r

    return p0 + r


def atom_to_move(o, p):
    """
    Return the coordinates xyz of an atom just above acceptor/donor atom o
    """
    # It will not work if there is just one dimension
    p = np.atleast_2d(p)
    return o + normalize(-1. * vector(o, np.mean(p, axis=0)))


def rotate_point(p, p1, p2, angle):
    """ Rotate the point p around the axis p1-p2
    Source: http://paulbourke.net/geometry/rotate/PointRotate.py"""
    # Translate the point we want to rotate to the origin
    pn = p - p1

    # Get the unit vector from the axis p1-p2
    n = p2 - p1
    n = normalize(n)

    # Setup the rotation matrix
    c = np.cos(angle)
    t = 1. - np.cos(angle)
    s = np.sin(angle)
    x, y, z = n[0], n[1], n[2]

    R = np.array([[t*x**2 + c, t*x*y - s*z, t*x*z + s*y],
                 [t*x*y + s*z, t*y**2 + c, t*y*z - s*x],
                 [t*x*z - s*y, t*y*z + s*x, t*z**2 + c]])

    # ... and apply it
    ptr = np.dot(pn, R)

    # And to finish, we put it back
    p = ptr + p1

    return p


def resize_vector(v, length, origin=None):
    """ Resize a vector v to a new length in regard to a origin """
    if origin is not None:
        return (normalize(v - origin) * length) + origin
    else:
        return normalize(v) * length


def generate_random_sphere(center, radius=1, size=100):
    """
    Generate a sphere with random point
    Source: https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
    """
    z = np.random.uniform(-radius, radius, size)
    p = np.random.uniform(0, np.pi * 2, size)

    x = np.sqrt(radius**2 - z**2) * np.cos(p)
    y = np.sqrt(radius**2 - z**2) * np.sin(p)

    coordinates = np.stack((x, y, z), axis=-1)
    coordinates += center

    return coordinates


def generate_sphere(center, radius=1, size=100):
    a = 4 * np.pi * radius**2 / size
    d = np.sqrt(a)

    M_v = np.int(np.round(np.pi / d))
    d_v = np.pi / M_v
    d_p = a / d_v

    coordinates = []

    for m in range(0, M_v):
        v = np.pi * (m + 0.5) / M_v
        M_p = np.int(np.round(2 * np.pi * np.sin(v) / d_p))

        for n in range(0, M_p):
            p = 2 * np.pi * n / M_p

            x = radius * np.sin(v) * np.cos(p)
            y = radius * np.sin(v) * np.sin(p)
            z = radius * np.cos(v)

            coordinates.append([x, y, z])

    coordinates = np.array(coordinates)
    coordinates += center

    return coordinates


def sphere_grid_points(center, spacing, radius, min_radius=0):
    """Generate grid sphere."""
    # Number of grid points based on the grid spacing
    n = np.int(np.rint(radius / spacing)) * 2
    # Transform even numbers to the nearest odd integer
    n = n // 2 * 2 + 1

    x = np.linspace(center[0] - radius, center[0] + radius, n)
    y = np.linspace(center[1] - radius, center[1] + radius, n)
    z = np.linspace(center[2] - radius, center[2] + radius, n)
    # Generate grid
    X, Y, Z = np.meshgrid(x, y, z)
    data = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
    # Compute distance and keep only the ones in the sphere
    distance = spatial.distance.cdist(data, center.reshape(1, -1)).ravel()
    points_in_sphere = data[(distance >= min_radius) & (distance <= radius)]
    return points_in_sphere


def makeW(r1, r2, r3, r4=0):
    """matrix involved in quaternion rotation
    
    source: https://github.com/charnley/rmsd
    """
    W = np.asarray([
        [r4, r3, -r2, r1],
        [-r3, r4, r1, r2],
        [r2, -r1, r4, r3],
        [-r1, -r2, -r3, r4]])
    return W


def makeQ(r1, r2, r3, r4=0):
    """matrix involved in quaternion rotation
    
    source: https://github.com/charnley/rmsd
    """
    Q = np.asarray([
        [r4, -r3, r2, r1],
        [r3, r4, -r1, r2],
        [-r2, r1, r4, r3],
        [-r1, -r2, -r3, r4]])
    return Q


def quaternion_rotate(Y, X):
    """Calculate the rotation between two set of coordinates

    source: https://github.com/charnley/rmsd

    Args:
        X (ndarray): (N,D) matrix, where N is points and D is dimension.
        Y: (ndarray): (N,D) matrix, where N is points and D is dimension.

    Returns:
        ndarray : quaternion
    """
    N = X.shape[0]
    W = np.asarray([makeW(*Y[k]) for k in range(N)])
    Q = np.asarray([makeQ(*X[k]) for k in range(N)])
    Qt_dot_W = np.asarray([np.dot(Q[k].T, W[k]) for k in range(N)])
    W_minus_Q = np.asarray([W[k] - Q[k] for k in range(N)])
    A = np.sum(Qt_dot_W, axis=0)
    eigen = np.linalg.eigh(A)
    r = eigen[1][:, eigen[0].argmax()]
    return r


def rotate_vector_by_quaternion(v, q):
    """Rotate point using a quaternion."""
    # https://gamedev.stackexchange.com/questions/28395/rotating-vector3-by-a-quaternion
    u = q[1:]
    s = q[0]
    v_prime = 2. * np.dot(u, v) * u + (s * s - np.dot(u, u)) * v + 2. * s * np.cross(u, v)
    return v_prime


def shoemake(coordinates):
    """Shoemake transformation."""
    # http://planning.cs.uiuc.edu/node198.html
    coordinates = np.atleast_2d(coordinates)
    t1 = np.sqrt(1. - coordinates[:,0])
    t2 = np.sqrt(coordinates[:,0])
    s1 = 2. * np.pi * coordinates[:,1]
    s2 = 2. * np.pi * coordinates[:,2]
    return np.dstack((t1 * np.sin(s1), t1 * np.cos(s1), 
                      t2 * np.sin(s2), t2 * np.cos(s2)))[0]


def random_quaternion(n=1):
    """Create n random quaternions."""
    u = np.random.random(size=(n, 3))
    return shoemake(u)


@contextlib.contextmanager
def temporary_directory(suffix=None, prefix=None, dir=None, clean=True):
    """Create and enter a temporary directory; used as context manager."""
    temp_dir = tempfile.mkdtemp(suffix, prefix, dir)
    cwd = os.getcwd()
    os.chdir(temp_dir)
    try:
        yield temp_dir
    finally:
        os.chdir(cwd)
        if clean:
            shutil.rmtree(temp_dir)


def is_writable(pathname):
    try:
        testfile = tempfile.NamedTemporaryFile(dir=pathname)
        testfile.close()
    except (PermissionError, FileNotFoundError) as e:
        raise RuntimeError('Can write in directory %s.' % pathname) from e


def execute_command(cmd_line):
    """Simple function to execute bash command."""
    args = cmd_line.split()
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    output, errors = p.communicate()
    return output, errors


def path_module(module_name):
    try:
        specs = importlib.machinery.PathFinder().find_spec(module_name)

        if specs is not None:
            return specs.submodule_search_locations[0]
    except:
        try:
            _, path, _ = imp.find_module(module_name)
            abspath = os.path.abspath(path)
            return abspath
        except ImportError:
            return None

    return None


def split_list_in_chunks(size, n):
    return [(l[0], l[-1]) for l in np.array_split(range(size), n)]


def boltzmann_probabilities(energies, temperature):
    # Boltzmann constant (kcal/mol)
    kb = 0.0019872041
    energies = np.array(energies)

    d = np.exp(-energies / (kb * temperature))
    d_sum = np.sum(d)

    if d_sum > 0:
        p = d / d_sum
    else:
        # It means that energies are too high
        return np.zeros(energies.shape[0])

    return p


def boltzmann_choices(energies, temperature, size=None):
    """Choose state i based on boltzmann probability."""
    p = boltzmann_probabilities(energies, temperature)

    if np.sum(p) == 0.:
        return np.array([])

    if size is None:
        size = 1

    if size > 1:
        # If some prob. in p are zero, ValueError: size of nonzero p is lower than size
        non_zero = np.count_nonzero(p)
        size = non_zero if non_zero < size else size

    i = np.random.choice(len(energies), size, False, p)

    return i


def boltzmann_acceptance_rejection(new_energies, old_energies, temperature=300):
    kb = 0.0019872041

    new_energies = np.ravel(new_energies)
    old_energies = np.ravel(old_energies)

    decisions = new_energies < old_energies

    if all(decisions):
        return decisions
    else:
        unfavorable_indices = np.where(decisions == False)

        if old_energies.size == 1:
            unfavorable_old_energies = old_energies
        else:
            unfavorable_old_energies = old_energies[unfavorable_indices]

        delta_e = new_energies[unfavorable_indices] - unfavorable_old_energies
        p_acc = np.minimum(1., np.exp(-delta_e / (kb * temperature)))
        r = np.random.rand(p_acc.shape[0])

        decisions[unfavorable_indices] = r <= p_acc

        return decisions


def convert_amber_to_autodock_types(molecule):
    """Convert amber atom types to AutoDock atom types
    
    Args:
        molecule (Molecule): input molecule

    Returns
        Molecule: copy of the original molecule with AutoDock atom types

    """
    molecule = copy.deepcopy(molecule)

    amber_autodock_dict = {
        'N3': 'N',
        'H': 'HD',
        'CX': 'C',
        'HP': 'H',
        'CT': 'C',
        'HC': 'H',
        'C': 'C',
        'O': 'OA',
        'N': 'N',
        'H1': 'H',
        'C3': 'C',
        '3C': 'C',
        'C2': 'C',
        '2C': 'C',
        'CO': 'C',
        'O2': 'OA',
        'OH': 'OA',
        'HO': 'HD',
        'SH': 'SA',
        'HS': 'HD',
        'CA': 'A',
        'HA': 'H',
        'S': 'SA',
        'C8': 'C',
        'N2': 'N',
        'CC': 'A',
        'NB': 'NA',
        'CR': 'A',
        'CV': 'A',
        'H5': 'H',
        'NA': 'N',
        'CW': 'A',
        'H4': 'H',
        'C*': 'A',
        'CN': 'A',
        'CB': 'A',
        'Zn2+': 'Zn',
        'Zn': 'Zn',
        'Mn2+': 'Mn',
        'Mn': 'Mn',
        'XC': 'C',
        'br': 'Br',
        'c' : 'C',
        'c1': 'C',
        'c2': 'C',
        'c3': 'C',
        'ca': 'A',
        'cc': 'A',
        'cd': 'A',
        'ce': 'C',
        'cf': 'C',
        'cl': 'Cl',
        'cp': 'A',
        'cq': 'A',
        'cu': 'C',
        'cv': 'C',
        'cx': 'C',
        'cy': 'C',
        'cz': 'C',
        'cs': 'C',
        'cg': 'C',
        'ch': 'C',
        'f' : 'F',
        'h1': 'H',
        'h2': 'H',
        'h3': 'H',
        'h4': 'H',
        'h5': 'H',
        'ha': 'H',
        'hc': 'H',
        'hn': 'HD',
        'ho': 'HD',
        'hp': 'HD',
        'hs': 'HD',
        'hx': 'H',
        'i' : 'I',
        'n' : 'N',
        'n1': 'NA',
        'n2': 'N',
        'n3': 'N',
        'n4': 'N',
        'n5': 'N',
        'n6': 'N',
        'n7': 'N',
        'n8': 'N',
        'n9': 'N',
        'na': 'N',
        'nb': 'N',
        'nc': 'N',
        'nd': 'N',
        'nh': 'N',
        'ne': 'N',
        'nf': 'N',
        'no': 'N',
        'n+': 'N',
        'nx': 'N',
        'ny': 'N',
        'nz': 'N',
        'ns': 'N',
        'nt': 'N',
        'nu': 'N',
        'nv': 'N',
        'ni': 'N',
        'nj': 'N',
        'nk': 'N',
        'nl': 'N',
        'nm': 'N',
        'nn': 'N',
        'np': 'N',
        'nq': 'N',
        'o' : 'OA',
        'oh': 'OA',
        'os': 'OA',
        'op': 'OA',
        'oq': 'OA',
        'p2': 'P',
        'p3': 'P',
        'p4': 'P',
        'p5': 'P',
        'pb': 'P',
        'pc': 'P',
        'pd': 'P',
        'pe': 'P',
        'pf': 'P',
        'px': 'P',
        'py': 'P',
        's' : 'S',
        's2': 'SA',
        's4': 'S',
        's6': 'S',
        'sh': 'SA',
        'ss': 'SA',
        'sx': 'S',
        'sy': 'S',
        'sp': 'S',
        'sq': 'S'
        #'Cu': 
    }

    for atom in molecule.atoms:
        if atom['resname'] == 'TYR' and atom['name'] == 'CZ' and atom['t'] == 'C':
            atom['t'] = 'A'
        elif atom['resname'] == 'ARG' and atom['name'] == 'CZ' and atom['t'] == 'CA':
            atom['t'] = 'C'
        else:
            atom['t'] = amber_autodock_dict[atom['t']]

    return molecule


def prepare_water_map(ad_map, water_model="tip3p", dielectric=1.):
    e_type = "Electrostatics"
    dielectric = np.float(dielectric)
    map_info = ad_map.info()

    """In TIP3P and TIP5P models, hydrogen atoms and lone-pairs does not
    have VdW radius, so their interactions with the receptor are purely
    based on electrostatics. So the HD and Lp maps are just the electrostatic
    map. Each map is multiplied by the partial charge. So it is just a
    look-up table to get the energy for each water molecule.
    """
    if water_model == "tip3p":
        ow_type = "OW"
        hw_type = "HW"
        ow_q = -0.834
        hw_q = 0.417

        missing = set([ow_type, e_type]).difference(map_info['maps'])
    elif water_model == "tip5p":
        ot_type = "OT"
        hw_type = "HT"
        lw_type = "LP"
        hw_q = 0.241
        lw_q = -0.241

        missing = set([ot_type, lw_type, e_type]).difference(map_info['maps'])
    else:
        raise RuntimeError("Water model %s unknown." % water_model)

    if missing:
        raise RuntimeError('Atom type(s) %s is(are) missing for %s water model.' % (list(missing), water_model))

    # For the TIP3P and TIP5P models
    ad_map.apply_operation_on_maps(hw_type, e_type, "x * %f / %f" % (hw_q, dielectric))

    if water_model == "tip3p":
        ad_map.apply_operation_on_maps(e_type, e_type, "x * %f / %f" % (ow_q, dielectric))
        ad_map.combine(ow_type, [ow_type, e_type], how="add")
    elif water_model == "tip5p":
        ad_map.apply_operation_on_maps(lw_type, e_type, "x * %f / %f" % (lw_q, dielectric))

    # Delete useless maps
    ad_map.delete_map('Electrostatics')
    ad_map.delete_map('Desolvation')
