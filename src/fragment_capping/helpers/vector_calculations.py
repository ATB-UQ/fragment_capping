"""
Using GROMOS method for virtual hydrogen atom placement.
Source: Biomolecular Simulation: The GROMOS96 Manual and User Guide chII, p67-72
"""

from math import sqrt, cos, sin, pi, acos
from numpy import array as vector, cross, dot
from numpy.linalg import norm

C_H_BOND_LENGTH = 1.0
TETRAHEDRAL_BOND_ANGLE = 109.5 * (pi / 180)
TRIGONAL_PLANAR_BOND_ANGLE = 120 * (pi / 180)

""" Used to calculate position of the first hydrogen when more than one are to
    be placed, and only one vector is known (i.e. C-C)

    Returns new coordinates for a hydrogen atom.
"""
def place_first_hydrogen(points: list, theta: float):

    # define base vector (e.g. C-C bond)
    base_vector = points[0] - points[1]

    # define unit vector in x-direction
    u = base_vector / norm(base_vector)

    # create arbitrary vector
    arb_vector = vector([0, 0, 1])

    # create unit vector in y-direction
    v = cross(u, arb_vector)
    v = v / norm(v)

    # calculate x, y of right angled triangle
    x = C_H_BOND_LENGTH * cos(pi - theta)
    y = C_H_BOND_LENGTH * sin(pi - theta)

    new_coordinates = points[0] + (x * u) + (y * v)

    return new_coordinates

def get_angle_between_vectors(v1: vector, v2: vector):
    return acos(dot(v1, v2) / (norm(v1) * norm(v2)))

def gromos_trigonal_planar_1H(points: list):

    # define point vectors
    ri = points[0]
    rj = points[1]
    rk = points[2]

    # define sum vector and norm of sum
    s = 2 * ri - rj - rk
    s_norm = norm(s)

    # define hydrogen bond distance
    d = C_H_BOND_LENGTH

    # calculate resulting hydrogen vector
    rn = ri + d * (s / s_norm)
    new_coordinates = tuple(rn.tolist())

    return new_coordinates

def gromos_trigonal_planar_2H(points: list) -> list:

    theta = TRIGONAL_PLANAR_BOND_ANGLE
    new_coordinates = []

    # calculate first hydrogen vector and add to list of points/new coordinates
    rn = place_first_hydrogen(points, theta)
    points.append(rn)
    new_coordinates.append(tuple(rn.tolist()))

    # calculate other hydrogen position using 1H method
    new_coordinates.append(gromos_trigonal_planar_1H(points))

    return new_coordinates

def gromos_tetrahedral_1H(points: list):

    # define point vectors
    ri = points[0]
    rj = points[1]
    rk = points[2]
    rl = points[3]

    # SPECIAL CASE FOR PRIOR TRIGONAL-PLANAR BOND ANGLE
    v1 = ri - rj
    v2 = ri - rk
    # 2 radians = approx. 115 degrees
    if get_angle_between_vectors(v1, v2) > 2:
        new_coordinates = ri + cross(v1, v2)
        return tuple(new_coordinates.tolist())

    # define sum vector and norm of sum
    s = 3 * ri - rj - rk - rl
    s_norm = norm(s)

    # define hydrogen bond distance
    d = C_H_BOND_LENGTH

    # calculate resulting hydrogen vector
    rn = ri + d * (s / s_norm)
    new_coordinates = tuple(rn.tolist())

    return new_coordinates


def gromos_tetrahedral_from_2_vectors(points: list, h_num: int) -> list:

    theta = TETRAHEDRAL_BOND_ANGLE
    new_coordinates = []

    while h_num > 0:

        # define point vectors differently for each hydrogen
        ri = points[0]
        if h_num == 1:
            rj = points[1]
            rk = points[2]
        else:
            rk = points[1]
            rj = points[2]

        # define sum vector and norm of sum
        s = 2 * ri - rj - rk
        s_norm = norm(s)

        # define q vector and norm of q
        q = cross((ri - rj), (ri - rk))
        q_norm = norm(q)

        # define hydrogen bond distance
        d = C_H_BOND_LENGTH

        # define alpha and beta parameters
        alpha = d * cos(theta / 2)
        beta = d * sin(theta / 2)

        # calculate resulting hydrogen vector
        rn = ri + alpha * (s / s_norm) + beta * (q / q_norm)
        new_coordinates.append(tuple(rn.tolist()))

        h_num -= 1

    return new_coordinates

def gromos_tetrahedral_3H(points: list) -> list:

    theta = TETRAHEDRAL_BOND_ANGLE

    new_coordinates = []

    # calculate first hydrogen vector and add to list of points/new coordinates
    rn = place_first_hydrogen(points, theta)
    points.append(rn)
    new_coordinates.append(tuple(rn.tolist()))

    # calculate other two hydrogen positions using 2H method
    new_coordinates.extend(gromos_tetrahedral_from_2_vectors(points, 2))

    return new_coordinates