# -*- coding: utf-8 -*-
"""NumPy based reader of Biovision Hierarchy (bvh) motion-capture files."""

import numpy as np
from . import reader


class NumpyBvhReader(reader.BvhReader):

    def __init__(self, filename):
        super(NumpyBvhReader, self).__init__(filename)
        self.read()

    def on_hierarchy(self, root):
        ch_x = root.channels.index('Xposition')
        ch_y = root.channels.index('Yposition')
        ch_z = root.channels.index('Zposition')
        if ch_y == ch_x + 1 and ch_z == ch_y + 1:
            self._pos_idx = slice(ch_x, ch_z + 1)
        else:
            self._pos_idx = [ch_x, ch_y, ch_z]
        self._ang_idx = [i for i in range(self.num_channels)
                         if i not in [ch_x, ch_y, ch_z]]

    def sorted_angle_channels(self, node, ch_offset=[0]):
        ch_rot_x = node.channels.index('Xrotation') + ch_offset[0]
        ch_rot_y = node.channels.index('Yrotation') + ch_offset[0]
        ch_rot_z = node.channels.index('Zrotation') + ch_offset[0]
        angle_idx = [ch_rot_x, ch_rot_y, ch_rot_z]
        ch_offset[0] += len(node.channels)
        for child in node.children:
            if not child.is_end_site:
                angle_idx += self.sorted_angle_channels(child, ch_offset)
        return angle_idx

    def on_motion(self, frames, dt):
        self.num_frames = frames
        self.dt = dt
        self.positions = np.empty((self.num_frames, 3))
        self.angles_deg = np.empty((self.num_frames, len(self._ang_idx)))
        self.angles_rad = np.empty((self.num_frames, len(self._ang_idx)))
        self._frame_num = 0

    def on_frame(self, values):
        values_arr = np.array(values)
        self.positions[self._frame_num] = values_arr[self._pos_idx]
        self.angles_deg[self._frame_num] = values_arr[self._ang_idx]
        self.angles_rad[self._frame_num] = (
            self.angles_deg[self._frame_num] * np.pi / 180.)
        self._frame_num += 1
