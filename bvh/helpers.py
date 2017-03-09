# -*- coding: utf-8 -*-
"""Helper functions for reading and rendering BVH files."""

import numpy as np
import os
from .numpy_reader import NumpyBvhReader


standard_joint_order = {
    joint: (i, channel_order) for i, (joint, channel_order) in enumerate(
        [
            ('hips', 'xyz'),
            ('spine', 'xyz'),
            ('head', 'xyz'),
            ('leftshoulder', 'xyz'),
            ('leftarm', 'xyz'),
            ('leftforearm', 'yzx'),
            ('lefthand', 'xyz'),
            ('lefthandthumb1', 'xzy'),
            ('rightshoulder', 'xyz'),
            ('rightarm', 'xyz'),
            ('rightforearm', 'yzx'),
            ('righthand', 'xyz'),
            ('righthandthumb1', 'xzy'),
            ('leftupleg', 'xyz'),
            ('leftleg', 'xyz'),
            ('leftfoot', 'xyz'),
            ('lefttoebase', 'xyz'),
            ('rightupleg', 'xyz'),
            ('rightleg', 'xyz'),
            ('rightfoot', 'xyz'),
            ('righttoebase', 'xyz')
        ]
    )
}


rom_joint_order = {
    joint: (i, channel_order) for i, (joint, channel_order) in enumerate(
        [
            ('hips', 'xyz'),
            ('spine', 'xyz'),
            ('head', 'xyz'),
            ('leftshoulder', 'xyz'),
            ('leftarm', 'xyz'),
            ('leftforearm', 'zyx'),
            ('lefthand', 'xyz'),
            ('lefthandthumb1', 'xyz'),
            ('leftupleg', 'xyz'),
            ('leftleg', 'xyz'),
            ('leftfoot', 'xyz'),
            ('lefttoebase', 'xyz'),
            ('rightshoulder', 'xyz'),
            ('rightarm', 'xyz'),
            ('rightforearm', 'zyx'),
            ('righthand', 'xyz'),
            ('righthandthumb1', 'xyz'),
            ('rightupleg', 'xyz'),
            ('rightleg', 'xyz'),
            ('rightfoot', 'xyz'),
            ('righttoebase', 'xyz')
        ]
    )
}


bone_lengths_map = {
    'spine': 'waist',
    'head': 'spine',
    'headendsite': 'head',
    'leftshoulder': 'mid-spine',
    'leftarm': 'half-shoulder',
    'leftforearm': 'upper-arm',
    'lefthand': 'lower-arm',
    'lefthandendsite': 'hand',
    'lefthandthumb1endsite': 'thumb',
    'rightshoulder': 'mid-spine',
    'rightarm': 'half-shoulder',
    'rightforearm': 'upper-arm',
    'righthand': 'lower-arm',
    'righthandendsite': 'hand',
    'righthandthumb1endsite': 'thumb',
    'leftupleg': 'half-hip',
    'leftleg': 'upper-leg',
    'leftfoot': 'lower-leg',
    'lefttoebase': 'back-foot',
    'lefttoebaseendsite': 'fore-foot',
    'rightupleg': 'half-hip',
    'rightleg': 'upper-leg',
    'rightfoot': 'lower-leg',
    'righttoebase': 'back-foot',
    'righttoebaseendsite': 'fore-foot',
}


def get_all_channels(joint_order):
    all_channels = [0] * (3 * len(joint_order))
    for name in joint_order:
        index_and_channels = joint_order[name]
        i = index_and_channels[0] * 3
        for channel in index_and_channels[1]:
            all_channels[i] = name + '_' + channel
            i += 1
    return all_channels


def populate_channel_order_and_offsets(node, channel_order, offsets, index,
                                       joint_order=standard_joint_order):
    if not node.is_end_site:
        file_order_string = ''
        for ch in node.channels:
            if ch[1:].lower() == 'rotation':
                file_order_string += ch[0].lower()
        joint_index, std_order_string = joint_order[node.name.lower()]
        assert file_order_string == std_order_string, (
            'Expected channel order {0} but file specifies {1} for node {2}'
            .format(std_order_string, file_order_string, node.name)
        )
        for i in range(joint_index * 3, (joint_index + 1) * 3):
            channel_order[i] = index[0]
            index[0] += 1
        offsets[joint_index * 3:(joint_index + 1) * 3] = node.offset
        for child in node.children:
            populate_channel_order_and_offsets(
                child, channel_order, offsets, index, joint_order)


def load_all(base_dir, joint_order=standard_joint_order):
    angles_list = []
    offsets_list = []
    for dir_name, subdir_list, file_list in os.walk(base_dir):
        for file_name in file_list:
            file_ext = os.path.splitext(file_name)[1]
            if file_ext.lower() == '.bvh':
                file_path = os.path.join(dir_name, file_name)
                reader = NumpyBvhReader(file_path)
                offsets = [0] * len(joint_order) * 3
                channel_order = [0] * len(joint_order) * 3
                try:
                    populate_channel_order_and_offsets(
                        reader.root, channel_order, offsets, [0], joint_order)
                    offsets_list.append(offsets)
                    angles_list.append(reader.angles_rad[:, channel_order])
                    print('Loaded ' + file_path)
                except AssertionError as e:
                    print('Skipping file {0} due to unexpected channel order: '
                          '{1}'.format(file_path, e.message))
    return np.vstack(angles_list), np.array(offsets_list)


def populate_lengths(node, lengths, lengths_map):
    if node.name.lower() in lengths_map:
        lengths[lengths_map[node.name.lower()]] = (
            np.array(node.offset)**2).sum()**0.5
    for child in node.children:
        populate_lengths(child, lengths, lengths_map)


def load_all_lengths(base_dir, lengths_map):
    lengths_list = []
    for dir_name, subdir_list, file_list in os.walk(base_dir):
        if file_list:
            file_name = file_list[0]
            file_ext = os.path.splitext(file_name)[1]
            if file_ext.lower() == '.bvh':
                file_path = os.path.join(dir_name, file_name)
                reader = NumpyBvhReader(file_path)
                process_skeleton(reader.root)
                lengths = np.zeros(len(np.unique(lengths_map.values())))
                populate_lengths(reader.root, lengths, lengths_map)
                lengths_list.append(lengths)
    return lengths_list


def process_skeleton(node):
    offset = np.array(node.offset)
    node.length = offset.dot(offset) ** 0.5
    node.offset_unit = offset / node.length if node.length != 0. else offset
    for child in node.children:
        if child.is_end_site and child.name == 'End Site':
            child.name = node.name + 'EndSite'
        process_skeleton(child)


def string_repr(node, prefix=''):
    rep = prefix + node.name + '(' + ', '.join(node.channels) + ')\n'
    for child in node.children:
        rep += string_repr(child, prefix + '--')
    return rep
