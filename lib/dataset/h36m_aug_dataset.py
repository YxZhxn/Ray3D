# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import numpy as np
from lib.skeleton.skeleton import Skeleton
from lib.dataset.mocap_dataset import MocapDataset
from lib.camera.camera import CameraInfoPacket

h36m_skeleton = Skeleton(parents=[-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
                                  16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
                         joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                         joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])

class Human36mDataset(MocapDataset):
    def __init__(self, path, camera_param, remove_static_joints=True, camera_wise_performance=False, universal=False):
        super().__init__(fps=50, skeleton=h36m_skeleton)
        self.universal = universal
        camera_meta = json.load(open(camera_param, 'r'))
        # subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
        subjects = [
            'S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11',
            'S1_0.6', 'S5_0.6', 'S6_0.6', 'S7_0.6', 'S8_0.6', 'S9_0.6', 'S11_0.6',
            'S1_0.7', 'S5_0.7', 'S6_0.7', 'S7_0.7', 'S8_0.7', 'S9_0.7', 'S11_0.7',
            'S1_0.8', 'S5_0.8', 'S6_0.8', 'S7_0.8', 'S8_0.8', 'S9_0.8', 'S11_0.8',
            'S1_0.9', 'S5_0.9', 'S6_0.9', 'S7_0.9', 'S8_0.9', 'S9_0.9', 'S11_0.9',
            'S1_1.1', 'S5_1.1', 'S6_1.1', 'S7_1.1', 'S8_1.1', 'S9_1.1', 'S11_1.1'
        ]

        if camera_wise_performance:
            camera_dist = list()
            for cam in camera_meta:
                # camera_dist.append((cam['id'], cam['pitch'], cam['translation_scale'], cam['degree']))
                camera_dist.append(cam['id'])
            self.camera_dist = camera_dist

        camera_info = dict()
        for subject in subjects:
            camera_info.setdefault(subject, list())
            for cam in camera_meta:
                K = np.eye(3, dtype=np.float64)
                K[0, 0] = cam['focal_length'][0]
                K[1, 1] = cam['focal_length'][1]
                K[0, 2] = cam['center'][0]
                K[1, 2] = cam['center'][1]

                R = np.array(cam['R']).reshape(3, 3)
                dist_coeff = np.array(
                    cam['radial_distortion'][:2] + cam['tangential_distortion'] + cam['radial_distortion'][2:]
                ).reshape((5,))

                t = np.array(cam['translation'], dtype=np.float64).reshape(3, 1)
                camera_info[subject].append(CameraInfoPacket(P=None, K=K, R=R, t=t,
                                                             res_w=cam['res_w'], res_h=cam['res_h'],
                                                             azimuth=cam['azimuth'], dist_coeff=dist_coeff))
        self.camera_info = camera_info

        # Load serialized dataset
        data = np.load(path, allow_pickle=True)['positions_3d'].item()

        self._data = {}
        for subject, actions in data.items():
            self._data[subject] = {}
            for action_name, positions in actions.items():
                self._data[subject][action_name] = {
                    'positions': positions,
                }

        if remove_static_joints:
            if self.universal:
                self.remove_joints([4, 5, 9, 10, 11, 12, 13, 14, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31])
            else:
                # Bring the skeleton to 17 joints instead of the original 32
                self.remove_joints([4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31])

                # Rewire shoulders to the correct parents
                self._skeleton._parents[11] = 8
                self._skeleton._parents[14] = 8

    def supports_semi_supervised(self):
        return True

    @staticmethod
    def remove_irrelevant_kpts(keypoints, universal=False):
        """

        :param keypoints:
        :return:
        """
        if universal:
            origin_keypoints, origin_keypoints_metadata = keypoints['positions_2d'].item(), keypoints['metadata'].item()
            updated_keypoints, updated_keypoints_metadata = dict(), dict()

            human36m_kpt_index = [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16]
            updated_keypoints_metadata['layout_name'] = 'h36m'
            updated_keypoints_metadata['num_joints'] = len(human36m_kpt_index)
            updated_keypoints_metadata['keypoints_symmetry'] = [[4, 5, 6, 8, 9, 10], [1, 2, 3, 11, 12, 13]]

            for subject in origin_keypoints.keys():
                updated_keypoints.setdefault(subject, dict())
                for action in origin_keypoints[subject]:
                    updated_keypoints[subject].setdefault(action, list())
                    for cam_idx, kps in enumerate(origin_keypoints[subject][action]):
                        updated_keypoints[subject][action].append(kps[:, human36m_kpt_index, :])

            return updated_keypoints, updated_keypoints_metadata
        else:
            raise NotImplementedError
