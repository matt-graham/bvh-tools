# -*- coding: utf-8 -*-
"""Theano based renderer functions for Biovision Hierarchy (bvh) skeletons."""

import numpy as np
import theano as th
import theano.tensor as tt


def rot_x(ang):
    """Returns matrix which defines rotation by `ang` about x-axis (0-axis)."""
    s = tt.sin(ang)
    c = tt.cos(ang)
    return tt.stack([
        1, 0, 0,
        0, c, -s,
        0, s, c,
    ]).reshape((3, 3))


def rot_y(ang):
    """Returns matrix which defines rotation by `ang` about y-axis (1-axis)."""
    s = tt.sin(ang)
    c = tt.cos(ang)
    return tt.stack([
        c, 0, s,
        0, 1, 0,
        -s, 0, c,
    ]).reshape((3, 3))


def rot_z(ang):
    """Returns matrix which defines rotation by `ang` about z-axis (2-axis)."""
    s = tt.sin(ang)
    c = tt.cos(ang)
    return tt.stack([
        c, -s, 0,
        s, c, 0,
        0, 0, 1,
    ]).reshape((3, 3))


def rot_x_batch(ang):
    """Returns batch of matrices which defines rotations about x-axis (0)."""
    s = tt.sin(ang)
    c = tt.cos(ang)
    rot = tt.repeat(tt.eye(3)[None, :, :], ang.shape[0], 0)
    rot = tt.set_subtensor(rot[:, 1, 1], c)
    rot = tt.set_subtensor(rot[:, 1, 2], -s)
    rot = tt.set_subtensor(rot[:, 2, 1], s)
    rot = tt.set_subtensor(rot[:, 2, 2], c)
    return rot


def rot_y_batch(ang):
    """Returns batch of matrices which defines rotations about y-axis (1)."""
    s = tt.sin(ang)
    c = tt.cos(ang)
    rot = tt.repeat(tt.eye(3)[None, :, :], ang.shape[0], 0)
    rot = tt.set_subtensor(rot[:, 0, 0], c)
    rot = tt.set_subtensor(rot[:, 0, 2], s)
    rot = tt.set_subtensor(rot[:, 2, 0], -s)
    rot = tt.set_subtensor(rot[:, 2, 2], c)
    return rot


def rot_z_batch(ang):
    """Returns batch of matrices which defines rotations about z-axis (2)."""
    s = tt.sin(ang)
    c = tt.cos(ang)
    rot = tt.repeat(tt.eye(3)[None, :, :], ang.shape[0], 0)
    rot = tt.set_subtensor(rot[:, 0, 0], c)
    rot = tt.set_subtensor(rot[:, 0, 1], -s)
    rot = tt.set_subtensor(rot[:, 1, 0], s)
    rot = tt.set_subtensor(rot[:, 1, 1], c)
    return rot


rotation_map = {
    'Xrotation': rot_x,
    'Yrotation': rot_y,
    'Zrotation': rot_z
}

rotation_map_batch = {
    'Xrotation': rot_x_batch,
    'Yrotation': rot_y_batch,
    'Zrotation': rot_z_batch
}


def get_bones(node, angles_rad, i=[0], parent_trans=None):
    """Get list of bones given skeletong tree and joint angles."""
    if parent_trans is None:
        parent_trans = tt.constant(np.eye(4))
    bones = []
    rot = tt.constant(np.eye(3))
    for ch in node.channels:
        if ch in rotation_map:
            rot = rot.dot(rotation_map[ch](angles_rad[i[0]]))
            i[0] += 1
    local_trans = tt.constant(np.zeros((4, 4)))
    local_trans = tt.set_subtensor(local_trans[:3, :3], rot)
    local_trans = tt.set_subtensor(
        local_trans[:3, 3], np.array(node.offset).astype(th.config.floatX))
    local_trans = tt.set_subtensor(local_trans[3, 3], 1)
    node_trans = parent_trans.dot(local_trans)
    point_1 = node_trans[:, 3]
    for child in node.children:
        point_2 = node_trans.dot(np.r_[child.offset, 1.])
        bones.append((point_1[:3] / point_1[3], point_2[:3] / point_2[3]))
        bones += get_bones(child, angles_rad, i, node_trans)
    return bones


def joint_positions(node, angles, fixed_angles=None, lengths=None,
                    lengths_map=None, skip=[], i=None, parent_trans=None):
    """Get list of joint pos. given skeleton tree and joint angles."""
    if i is None:
        i = [0]
    if parent_trans is None:
        parent_trans = tt.eye(4)
    joints = []
    rot = tt.eye(3)
    for ch in node.channels:
        ch_key = node.name.lower() + '_' + ch[0].lower()
        if ch in rotation_map:
            if fixed_angles is not None and ch_key in fixed_angles:
                rot = rot.dot(rotation_map[ch](fixed_angles[ch_key]))
            else:
                rot = rot.dot(rotation_map[ch](angles[i[0]]))
                i[0] += 1
    local_trans = tt.eye(4)
    local_trans = tt.set_subtensor(local_trans[:3, :3], rot)
    if not (node.name.lower() in skip and node.is_end_site):
        if lengths is None or node.length == 0.:
            node_offset = np.array(node.offset).astype(th.config.floatX)
        else:
            length = lengths[lengths_map[node.name.lower()]]
            node_offset = (node.offset_unit * length).astype(th.config.floatX)
        local_trans = tt.set_subtensor(local_trans[:3, 3], node_offset)
        node_trans = parent_trans.dot(local_trans)
        if not node.name.lower() in skip:
            joints.append(node_trans[:, 3])
        for child in node.children:
            joints += joint_positions(child, angles, fixed_angles, lengths,
                                      lengths_map, skip, i, node_trans)
    return joints


def joint_positions_batch(
        node, angles, fixed_angles=None, lengths=None,
        lengths_map=None, skip=[], i=None, parent_trans=None):
    """Get list of joint pos. given skeleton tree and joint angles (batch)."""
    # check whether single vector of angles or mini-batch matrix provided
    if angles.ndim == 2:
        n_batch = angles.shape[0]
    elif angles.ndim == 1:
        n_batch = 1
        angles = angles[None, :]
        if lengths is not None and lengths.ndim == 1:
            lengths = lengths[None, :]
    else:
        raise Exception('angles should be one or two dimensional.')
    if i is None:
        i = [0]
    if parent_trans is None:
        parent_trans = tt.repeat(tt.eye(4)[None, :, :], n_batch, 0)
    joints = []
    rot = tt.repeat(tt.eye(3)[None, :, :], n_batch, 0)
    for ch in node.channels:
        ch_key = node.name.lower() + '_' + ch[0].lower()
        if ch in rotation_map:
            if fixed_angles is not None and ch_key in fixed_angles:
                rot = rot.dot(rotation_map[ch](fixed_angles[ch_key]))
            else:
                rot = tt.batched_dot(
                    rot, rotation_map_batch[ch](angles[:, i[0]]))
                i[0] += 1
    local_trans = tt.repeat(tt.eye(4)[None, :, :], n_batch, 0)
    local_trans = tt.set_subtensor(local_trans[:, :3, :3], rot)
    if not (node.name.lower() in skip and node.is_end_site):
        if lengths is None or node.length == 0.:
            node_offset = np.array(node.offset,
                                   dtype=th.config.floatX)[None, :]
        else:
            length = lengths[:, lengths_map[node.name.lower()]]
            node_offset = (node.offset_unit[None, :] *
                           length[:, None]).astype(th.config.floatX)
        local_trans = tt.set_subtensor(local_trans[:, :3, 3], node_offset)
        node_trans = tt.batched_dot(parent_trans, local_trans)
        if not node.name.lower() in skip:
            joints.append(node_trans[:, :, 3])
        for child in node.children:
            joints += joint_positions_batch(
                child, angles, fixed_angles, lengths,
                lengths_map, skip, i, node_trans)
    return joints


def camera_matrix(focal_length, position, yaw_pitch_roll):
    """Get projective camera matrix given pose and focal length."""
    cam_mtx = tt.constant(np.zeros((3, 4)))
    rot_mtx = rot_z(yaw_pitch_roll[2])
    rot_mtx = rot_mtx.dot(rot_x(yaw_pitch_roll[1]))
    rot_mtx = rot_mtx.dot(rot_y(yaw_pitch_roll[0]))
    cam_mtx = tt.set_subtensor(cam_mtx[:, :3], rot_mtx)
    cam_mtx = tt.set_subtensor(cam_mtx[:, 3], position)
    int_mtx = tt.stack([
        focal_length, 0, 0,
        0, focal_length, 0,
        0, 0, 1
    ]).reshape((3, 3))
    cam_mtx = int_mtx.dot(cam_mtx)
    return cam_mtx


def camera_matrix_batch(focal_length, position, yaw_pitch_roll):
    """Get projective camera matrix given pose and focal length (batch)."""
    if focal_length.ndim == 1:
        n_batch = focal_length.shape[0]
    else:
        n_batch = 1
        focal_length = focal_length.reshape((1,))
        position = position[None, :]
        yaw_pitch_roll = yaw_pitch_roll[None, :]
    cam_mtx = tt.zeros((n_batch, 3, 4))
    rot_mtx = rot_z_batch(yaw_pitch_roll[:, 2])
    rot_mtx = tt.batched_dot(
        rot_mtx, rot_x_batch(yaw_pitch_roll[:, 1]))
    rot_mtx = tt.batched_dot(
        rot_mtx, rot_y_batch(yaw_pitch_roll[:, 0]))
    cam_mtx = tt.set_subtensor(cam_mtx[:, :, :3], rot_mtx)
    cam_mtx = tt.set_subtensor(cam_mtx[:, :, 3], position)
    int_mtx = tt.stack([focal_length, focal_length,
                        tt.ones_like(focal_length)])
    cam_mtx = (int_mtx * cam_mtx.T).T
    return cam_mtx
