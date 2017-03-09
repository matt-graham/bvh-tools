# -*- coding: utf-8 -*-
"""NumPy based renderer functions for Biovision Hierarchy (bvh) skeletons."""

import numpy as np
from math import sin, cos


def rot_x(ang):
    return np.array([
        [1., 0., 0.], [0., cos(ang), -sin(ang)], [0., sin(ang), cos(ang)]
    ])


def rot_y(ang):
    return np.array([
        [cos(ang), 0., sin(ang)], [0., 1., 0.], [-sin(ang), 0., cos(ang)]
    ])


def rot_z(ang):
    return np.array([
        [cos(ang), -sin(ang), 0.], [sin(ang), cos(ang), 0.], [0., 0., 1.]
    ])


rotation_map = {
    'Xrotation': rot_x,
    'Yrotation': rot_y,
    'Zrotation': rot_z
}


def get_bones(node, angles_rad, parent_trans=np.eye(4)):
    bones = []
    rot = np.eye(3)
    for ch in node.channels:
        if ch in rotation_map:
            rot = rot.dot(rotation_map[ch](angles_rad.pop(0)))
    local_trans = np.zeros((4, 4))
    local_trans[:3, :3] = rot
    local_trans[:, 3] = node.offset + (1,)
    node_trans = parent_trans.dot(local_trans)
    point_1 = node_trans.dot(np.array([0., 0., 0., 1.]))
    for child in node.children:
        point_2 = node_trans.dot(np.r_[child.offset, 1.])
        bones.append((point_1[:3] / point_1[3], point_2[:3] / point_2[3]))
        bones += get_bones(child, angles_rad, node_trans)
    return bones
