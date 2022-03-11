

import copy
import numpy as np
from lib.skeleton.skeleton import Skeleton
from lib.dataset.mocap_dataset import MocapDataset
from lib.camera.camera import CameraInfoPacket

humaneva_skeleton = Skeleton(parents=[-1, 0, 1, 2, 3, 1, 5, 6, 0, 8, 9, 0, 11, 12, 1],
       joints_left=[2, 3, 4, 8, 9, 10],
       joints_right=[5, 6, 7, 11, 12, 13])

humaneva_cameras_intrinsic_params = [
    {
        'id': 'C1',
        'res_w': 640,
        'res_h': 480,
        'center': [299.773675, 232.347455],
        'focal_length': [765.789418, 765.330306],
        'radial_distortion': [-0.289415, 0.105962, 0.000000],
        'tangential_distortion': [-0.001358, 0.001130],
        'skew_coefficient': [0.000000],
        'azimuth': 0,  # Only used for visualization
    }
]

humaneva_cameras_extrinsic_params = {
    'S1':
        [
            {
                'R':
                    [
                        [-0.143362, 0.989627, 0.009217],
                        [0.167173, 0.033395 , -0.985362],
                        [-0.975449, -0.139723, -0.170226]
                    ],
                'translation': [-88.085494, 804.781346, 4315.339452]
            }
        ],
    'S2':
        [
            {
                'R':
                    [
                        [-0.152389, 0.988262, 0.010755],
                        [0.170400, 0.036991, -0.984680],
                        [-0.973520, -0.148222, -0.174037],
                    ],
                'translation': [-8.996069, 797.610556, 4365.802204]
            }
        ],
    'S3':
        [
            {
                'R':
                    [
                        [-0.143362, 0.989627, 0.009217],
                        [0.167173, 0.033395 , -0.985362],
                        [-0.975449, -0.139723, -0.170226]
                    ],
                'translation': [-88.085494, 804.781346, 4315.339452]
            }
        ]
}

class HumanEvaDataset(MocapDataset):
    def __init__(self, path, universal=False):
        super().__init__(fps=60, skeleton=humaneva_skeleton)
        self.universal = universal
        self._cameras = copy.deepcopy(humaneva_cameras_extrinsic_params)
        for cameras in self._cameras.values():
            for i, cam in enumerate(cameras):
                cam.update(humaneva_cameras_intrinsic_params[i])
                for k, v in cam.items():
                    if k not in ['id', 'res_w', 'res_h']:
                        cam[k] = np.array(v, dtype='float32')

                # Normalize camera frame
                if 'translation' in cam:
                    cam['translation'] = cam['translation'] / 1000  # mm to meters

        camera_info = dict()
        for subject in self._cameras:
            # camera_info.setdefault(subject, list())
            for cam in self._cameras[subject]:
                if 'translation' not in cam:
                    continue
                K = np.eye(3, dtype=np.float64)
                K[0, 0] = cam['focal_length'][0]
                K[1, 1] = cam['focal_length'][1]
                K[0, 2] = cam['center'][0]
                K[1, 2] = cam['center'][1]

                R = cam['R']
                dist_coeff = np.concatenate(
                    (cam['radial_distortion'][:2], cam['tangential_distortion'], cam['radial_distortion'][2:])
                ).astype(np.float32).reshape((5,))

                t = np.array(cam['translation'], dtype=np.float64).reshape(3, 1)
                for prefix in ['Train/', 'Validate/']:
                    camera_info.setdefault(prefix + subject, list())
                    camera_info[prefix + subject].append(CameraInfoPacket(P=None, K=K, R=R, t=t,
                                                             res_w=cam['res_w'], res_h=cam['res_h'],
                                                             azimuth=cam['azimuth'], dist_coeff=dist_coeff))

        self.camera_info = camera_info

        for subject in list(self._cameras.keys()):
            data = self._cameras[subject]
            del self._cameras[subject]
            for prefix in ['Train/', 'Validate/']:
                self._cameras[prefix + subject] = data
        
        # Load serialized dataset
        data = np.load(path, allow_pickle=True)['positions_3d'].item()
        
        self._data = {}
        for subject, actions in data.items():
            self._data[subject] = {}
            for action_name, positions in actions.items():
                self._data[subject][action_name] = {
                    'positions': positions
                }
        if self.universal:
            self._skeleton = Skeleton(parents=[-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
                                  16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
                         joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                         joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])
            self._skeleton.remove_joints([4, 5, 9, 10, 11, 12, 13, 14, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31])
            kpt_index = [0, 11, 12, 13, 8, 9, 10, 14, 2, 3, 4, 5, 6, 7]
            for subject in self._data.keys():
                for action in self._data[subject].keys():
                    s = self._data[subject][action]  # if remove_static_joints:
                    if 'positions' in s:  # Bring the skeleton to 14 joints instead of the original 32
                        s['positions'] = s['positions'][:, kpt_index]  # self.remove_joints([])
        else:
            pass

    @staticmethod
    def remove_irrelevant_kpts(keypoints, universal=False):
        """

        :param keypoints:
        :return:
        """
        origin_keypoints, origin_keypoints_metadata = keypoints['positions_2d'].item(), keypoints['metadata'].item()
        updated_keypoints, updated_keypoints_metadata = dict(), dict()

        if universal:
            kpt_index = [0, 11, 12, 13, 8, 9, 10, 14, 2, 3, 4, 5, 6, 7]
            updated_keypoints_metadata['layout_name'] = 'humaneva'
            updated_keypoints_metadata['num_joints'] = len(kpt_index)
            updated_keypoints_metadata['keypoints_symmetry'] = [[4, 5, 6, 8, 9, 10], [1, 2, 3, 11, 12, 13]]

            for subject in origin_keypoints.keys():
                updated_keypoints.setdefault(subject, dict())
                for action in origin_keypoints[subject]:
                    updated_keypoints[subject].setdefault(action, list())
                    for cam_idx, kps in enumerate(origin_keypoints[subject][action]):
                        updated_keypoints[subject][action].append(kps[:, kpt_index, :])
            return updated_keypoints, updated_keypoints_metadata
        else:
            for subject in origin_keypoints.keys():
                updated_keypoints.setdefault(subject, dict())
                for action in origin_keypoints[subject]:
                    updated_keypoints[subject].setdefault(action, list())
                    updated_keypoints[subject][action] = origin_keypoints[subject][action]
            return updated_keypoints, origin_keypoints_metadata