#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Kits
#
# Utils functions
#


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
    angle = np.arccos(cos_angle)

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
    Return the coordinates xyz of an atom just above acceptor/donor atom o
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

def write_water(fname, waters, anchor=False, previous=False):

    if not isinstance(waters, (list, tuple)):
        waters = [waters]

    i = 0
    line = "ATOM  %5d%3s   DUM A%4d    %8.3f%8.3f%8.3f  1.00  1.00     0.000%2s\n"
    connect_lines = ""

    with open(fname, 'w') as w:
        for water in waters:
            coord = water.get_coordinates(atom_id=0)

            w.write(line % (i, 'O', i, coord[0][0], coord[0][1], coord[0][2], 'O'))

            if previous and water._previous is not None:
                coord = water._previous

                w.write(line % (i+1, 'D', i, coord[0][0], coord[0][1], coord[0][2], 'D'))
                connect_lines += "CONECT%5d%5d\n" % (i, i+1)

            if anchor:
                coord = water._anchor

                for i in range(coord.shape[0]):
                    w.write(line % (i, 'D', i, coord[i][0], coord[i][1], coord[i][2], 'D'))

            i += 2

        if previous:
            w.write(connect_lines)
