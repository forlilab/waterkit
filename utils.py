#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Kits
#
# Utils functions
#


import numpy as np


def vector(a, b):
    return b - a

def normalize(a):
    return a/np.sqrt(np.sum(a**2))

def euclidean_distance(a, b):
    """ Euclidean distance function """
    return np.sqrt(np.sum(np.power(a - b, 2), axis=1))

def get_angle(a, b, c, degree=True):

    ba = vector(a, b)
    bc = vector(b, c)

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arcos(cos_angle)

    if degree:
        return np.degrees(angle)

    return angle

def rotation_axis(o, p1, p2):
    """
    Compute rotation axis centered at the origin o
    """
    return o + normalize(np.cross(vector(p1, o), vector(p2, o)))

def atom_to_move(o, p):
    """
    Compute coordinates of atom just above acceptor/donor atom o
    """
    # It won't work if there is just one dimension
    p = np.atleast_2d(p)
    return o + normalize(-1. * vector(o, np.mean(p, axis=0)))


def rotate_3d_point(p, p1, p2, angle):
    """http://paulbourke.net/geometry/rotate/PointRotate.py"""
    
    # Get the unit vector from the axis p1-p2
    n = p2 - p1
    nm = np.sqrt(np.sum(n ** 2))
    n /= nm
    
    # Setup the rotation matrix
    c = np.cos(angle)
    t = 1. - np.cos(angle)
    s = np.sin(angle)
    x, y, z = n[0], n[1], n[2]
    
    R = np.array([[t*x**2 + c, t*x*y - s*z, t*x*z + s*y],
                 [t*x*y + s*z, t*y**2 + c, t*y*z - s*x],
                 [t*x*z - s*y, t*y*z + s*x, t*z**2 + c]])
    
    # ... and apply it
    ptr = np.dot(p, R)
    
    return ptr

def rotate_atom(p, p1, p2, angle=0, length=None):
    # Translate the point we want to rotate to the origin
    pn = p - p1
    
    # Rotate the point if we have to
    if angle != 0:
        pn = rotate_3d_point(pn, p1, p2, angle)
    
    # Change the distance of the point from the origin
    if length is not None:
        pn = normalize(pn) * length
    
    return pn + p1

def write_pdb(fname, coor_atoms):
    i = 0
    line = "ATOM  %5d  D   DUM Z%4d    %8.3f%8.3f%8.3f  1.00  1.00     0.000 D\n"
    
    with open(fname, 'w') as w:
        for j in range(0, coor_atoms.shape[0]):
            try:
                x, y, z = coor_atoms[i, 0], coor_atoms[i, 1], coor_atoms[i, 2]
                w.write(line % (i, i, x, y, z))
                i += 1
            except:
                continue

def write_pdb_opt_water(fname, old, new):
    i = 0
    line = "ATOM  %5d%3s   DUM Z%4d    %8.3f%8.3f%8.3f  1.00  1.00     0.000%2s\n"
    connect_lines = ""

    with open(fname, 'w') as w:
        for j in range(0, old.shape[0]):
            try:
                x1, y1, z1 = old[i, 0], old[i, 1], old[i, 2]
                x2, y2, z2 = new[i, 0], new[i, 1], new[i, 2]

                w.write(line % (i, 'O', i, x1, y1, z1, 'O'))
                w.write(line % (i+1, 'N', i, x2, y2, z2, 'N'))

                connect_lines += "CONECT%5d%5d\n" % (i, i+1)

                i += 2
            except:
                continue

        w.write(connect_lines)
