

import copy
import numpy as np
from lib.skeleton.skeleton import Human36mSkeleton
from lib.dataset.mocap_dataset import MocapDataset
from lib.camera.camera import CameraInfoPacket

mpii_3dhp_cameras_intrinsic_params = [
    {
        'id': '0',
        'center': [1024.704, 1051.394],
        'focal_length': [1497.693, 1497.103],
        'res_w': 2048,
        'res_h': 2048,
        'azimuth': 70,
    },
    {
        'id': '1',
        'center': [1030.519, 1052.626],
        'focal_length': [1495.217, 1495.52],
        'res_w': 2048,
        'res_h': 2048,
        'azimuth': 70,
    },
    {
        'id': '2',
        'center': [983.8873, 987.5902],
        'focal_length': [1495.587, 1497.828],
        'res_w': 2048,
        'res_h': 2048,
        'azimuth': 70,
    },
    {
        'id': '3',
        'center': [1029.06, 1041.409],
        'focal_length': [1495.886, 1496.033],
        'res_w': 2048,
        'res_h': 2048,
        'azimuth': 70,
    },
    {
        'id': '4',
        'center': [987.6075, 1019.069],
        'focal_length': [1490.952, 1491.108],
        'res_w': 2048,
        'res_h': 2048,
        'azimuth': 70,
    },
    {
        'id': '5',
        'center': [1012.331, 998.5009],
        'focal_length': [1500.414, 1499.971],
        'res_w': 2048,
        'res_h': 2048,
        'azimuth': 70,
    },
    {
        'id': '6',
        'center': [999.7319, 1010.251],
        'focal_length': [1498.471, 1498.8],
        'res_w': 2048,
        'res_h': 2048,
        'azimuth': 70,
    },
    {
        'id': '7',
        'center': [987.2716, 976.8773],
        'focal_length': [1498.831, 1499.674],
        'res_w': 2048,
        'res_h': 2048,
        'azimuth': 70,
    },
    {
        'id': '8',
        'center': [1017.387, 1043.032],
        'focal_length': [1500.172, 1500.837],
        'res_w': 2048,
        'res_h': 2048,
        'azimuth': 70,
    },
    {
        'id': '9',
        'center': [1010.423, 1037.096],
        'focal_length': [1501.554, 1501.9],
        'res_w': 2048,
        'res_h': 2048,
        'azimuth': 70,
    },
    {
        'id': '10',
        'center': [1041.614, 997.0433],
        'focal_length': [1498.423, 1498.585],
        'res_w': 2048,
        'res_h': 2048,
        'azimuth': 70,
    },
    {
        'id': '11',
        'center': [1009.802, 999.9984],
        'focal_length': [1495.779, 1493.703],
        'res_w': 2048,
        'res_h': 2048,
        'azimuth': 70,
    },
    {
        'id': '12',
        'center': [1000.56, 1014.975],
        'focal_length': [1501.326, 1501.491],
        'res_w': 2048,
        'res_h': 2048,
        'azimuth': 70,
    },
    {
        'id': '13',
        'center': [1005.702, 1004.214],
        'focal_length': [1496.961, 1497.378],
        'res_w': 2048,
        'res_h': 2048,
        'azimuth': 70,
    }
]

mpii_3dhp_cameras_extrinsic_params = [
    {
        'translation': [-0.5628666, 1.3981379999999999, 3.852623],
        'R':
            [
                [0.9650164, -0.262144, 0.00488022],
                [-0.004488356, -0.0351275, -0.9993728],
                [0.262151, 0.9643893, -0.03507521]
            ]
    },
    {
        'translation': [-1.429856, 0.7381779, 4.897966],
        'R':
            [
                [0.6050639, -0.7958773, -0.02184232],
                [-0.22647, -0.1457429, -0.9630526],
                [0.7632883, 0.587655, -0.2684261]
            ]
    },
    {
        'translation': [0.05725702, 1.307287, 2.7998220000000003],
        'R':
            [
                [-0.3608179, -0.932588, -0.009492658],
                [-0.0585942, 0.03282591, -0.9977421],
                [0.9307939, -0.359447, -0.06648842]
            ]
    },
    {
        'translation': [-0.2848168, 0.8079184, 3.1771599999999998],
        'R':
            [
                [-0.0721105, -0.9962325, -0.04817664],
                [-0.4393254, 0.07508985, -0.8951841],
                [0.895429, -0.04338695, -0.443085]
            ]
    },
    {
        'translation': [-1.563911, 0.8019607999999999, 3.5173159999999997],
        'R':
            [
                [0.3737275, -0.9224646, 0.09688602],
                [-0.009716132, -0.1083427, -0.9940662],
                [0.9274878, 0.3705685, -0.04945343]
            ]
    },
    {
        'translation': [0.35841340000000005, 0.9945657999999999, 3.439832],
        'R':
            [
                [-0.3521056, 0.9358659, 0.01328985],
                [-0.04961938, -0.004485628, -0.9987582],
                [-0.9346441, -0.3523278, 0.0480165]
            ]
    },
    {
        'translation': [0.5694388, 0.528871, 3.6873690000000003],
        'R':
            [
                [-0.9150326, -0.4004618, -0.04843184],
                [-0.1804886, 0.5138369, -0.8386868],
                [0.3607481, -0.7586845, -0.5424563]
            ]
    },
    {
        'translation': [1.378866, 1.270781, 2.631567],
        'R':
            [
                [-0.9995936, -0.001368653, 0.02847456],
                [-0.02843213, -0.0246889, -0.9992908],
                [0.002070688, -0.9996943, 0.02463995]
            ]
    },
    {
        'translation': [0.2213543, 0.65987, 3.644688],
        'R':
            [
                [0.000575281, 0.9981001, 0.06160985],
                [0.2082146, 0.06013997, -0.9762325],
                [-0.978083, 0.01338968, -0.2077844]
            ]
    },
    {
        'translation': [0.38862169999999996, 0.1375452, 4.216635],
        'R':
            [
                [0.04176839, 0.9990969, 0.00780962],
                [0.5555364, -0.01672664, -0.831324],
                [-0.8304425, 0.03906159, -0.5557333]
            ]
    },
    {
        'translation': [1.167962, 0.6176362000000001, 4.472351],
        'R':
            [
                [-0.8970265, 0.4204822, 0.1361548],
                [0.09417118, 0.4828178, -0.8706428],
                [-0.4318278, -0.7681679, -0.4726976]
            ]
    },
    {
        'translation': [0.1348272, 0.2515094, 4.570244],
        'R':
            [
                [0.9170455, 0.3465695, 0.1972746],
                [0.1720879, -0.7901813, 0.5882171],
                [0.3597408, -0.5054733, -0.7842726]
            ]
    },
    {
        'translation': [0.4124695, 0.5327588, 4.887095],
        'R':
            [
                [-0.7926738, -0.5951031, 0.1323657],
                [-0.396246, 0.66792, 0.6299778],
                [-0.4633114, 0.4469175, -0.7652499]
            ]
    },
    {
        'translation': [0.8671278, 0.8274571999999999, 3.985159],
        'R':
            [
                [-0.8701088, 0.4835728, -0.09522671],
                [0.4120245, 0.8197188, 0.3978655],
                [0.270456, 0.3069505, -0.9124883]
            ]
    }
]

subjects = [
    'S1_Seq1_0', 'S1_Seq1_1', 'S1_Seq1_2', 'S1_Seq1_3', 'S1_Seq1_4', 'S1_Seq1_5', 'S1_Seq1_6', 'S1_Seq1_7',
    'S1_Seq1_8', 'S1_Seq1_9', 'S1_Seq1_10', 'S1_Seq1_11', 'S1_Seq1_12', 'S1_Seq1_13', 'S1_Seq2_0', 'S1_Seq2_1',
    'S1_Seq2_2', 'S1_Seq2_3', 'S1_Seq2_4', 'S1_Seq2_5', 'S1_Seq2_6', 'S1_Seq2_7', 'S1_Seq2_8', 'S1_Seq2_9',
    'S1_Seq2_10', 'S1_Seq2_11', 'S1_Seq2_12', 'S1_Seq2_13', 'S2_Seq1_0', 'S2_Seq1_1', 'S2_Seq1_2', 'S2_Seq1_3',
    'S2_Seq1_4', 'S2_Seq1_5', 'S2_Seq1_6', 'S2_Seq1_7', 'S2_Seq1_8', 'S2_Seq1_9', 'S2_Seq1_10', 'S2_Seq1_11',
    'S2_Seq1_12', 'S2_Seq1_13', 'S2_Seq2_0', 'S2_Seq2_1', 'S2_Seq2_2', 'S2_Seq2_3', 'S2_Seq2_4', 'S2_Seq2_5',
    'S2_Seq2_6', 'S2_Seq2_7', 'S2_Seq2_8', 'S2_Seq2_9', 'S2_Seq2_10', 'S2_Seq2_11', 'S2_Seq2_12', 'S2_Seq2_13',
    'S3_Seq1_0', 'S3_Seq1_1', 'S3_Seq1_2', 'S3_Seq1_3', 'S3_Seq1_4', 'S3_Seq1_5', 'S3_Seq1_6', 'S3_Seq1_7',
    'S3_Seq1_8', 'S3_Seq1_9', 'S3_Seq1_10', 'S3_Seq1_11', 'S3_Seq1_12', 'S3_Seq1_13', 'S3_Seq2_0', 'S3_Seq2_1',
    'S3_Seq2_2', 'S3_Seq2_3', 'S3_Seq2_4', 'S3_Seq2_5', 'S3_Seq2_6', 'S3_Seq2_7', 'S3_Seq2_8', 'S3_Seq2_9',
    'S3_Seq2_10', 'S3_Seq2_11', 'S3_Seq2_12', 'S3_Seq2_13', 'S4_Seq1_0', 'S4_Seq1_1', 'S4_Seq1_2', 'S4_Seq1_3',
    'S4_Seq1_4', 'S4_Seq1_5', 'S4_Seq1_6', 'S4_Seq1_7', 'S4_Seq1_8', 'S4_Seq1_9', 'S4_Seq1_10', 'S4_Seq1_11',
    'S4_Seq1_12', 'S4_Seq1_13', 'S4_Seq2_0', 'S4_Seq2_1', 'S4_Seq2_2', 'S4_Seq2_3', 'S4_Seq2_4', 'S4_Seq2_5',
    'S4_Seq2_6', 'S4_Seq2_7', 'S4_Seq2_8', 'S4_Seq2_9', 'S4_Seq2_10', 'S4_Seq2_11', 'S4_Seq2_12', 'S4_Seq2_13',
    'S5_Seq1_0', 'S5_Seq1_1', 'S5_Seq1_2', 'S5_Seq1_3', 'S5_Seq1_4', 'S5_Seq1_5', 'S5_Seq1_6', 'S5_Seq1_7',
    'S5_Seq1_8', 'S5_Seq1_9', 'S5_Seq1_10', 'S5_Seq1_11', 'S5_Seq1_12', 'S5_Seq1_13', 'S5_Seq2_0', 'S5_Seq2_1',
    'S5_Seq2_2', 'S5_Seq2_3', 'S5_Seq2_4', 'S5_Seq2_5', 'S5_Seq2_6', 'S5_Seq2_7', 'S5_Seq2_8', 'S5_Seq2_9',
    'S5_Seq2_10', 'S5_Seq2_11', 'S5_Seq2_12', 'S5_Seq2_13', 'S6_Seq1_0', 'S6_Seq1_1', 'S6_Seq1_2', 'S6_Seq1_3',
    'S6_Seq1_4', 'S6_Seq1_5', 'S6_Seq1_6', 'S6_Seq1_7', 'S6_Seq1_8', 'S6_Seq1_9', 'S6_Seq1_10', 'S6_Seq1_11',
    'S6_Seq1_12', 'S6_Seq1_13', 'S6_Seq2_0', 'S6_Seq2_1', 'S6_Seq2_2', 'S6_Seq2_3', 'S6_Seq2_4', 'S6_Seq2_5',
    'S6_Seq2_6', 'S6_Seq2_7', 'S6_Seq2_8', 'S6_Seq2_9', 'S6_Seq2_10', 'S6_Seq2_11', 'S6_Seq2_12', 'S6_Seq2_13',
    'S7_Seq1_0', 'S7_Seq1_1', 'S7_Seq1_2', 'S7_Seq1_3', 'S7_Seq1_4', 'S7_Seq1_5', 'S7_Seq1_6', 'S7_Seq1_7',
    'S7_Seq1_8', 'S7_Seq1_9', 'S7_Seq1_10', 'S7_Seq1_11', 'S7_Seq1_12', 'S7_Seq1_13', 'S7_Seq2_0', 'S7_Seq2_1',
    'S7_Seq2_2', 'S7_Seq2_3', 'S7_Seq2_4', 'S7_Seq2_5', 'S7_Seq2_6', 'S7_Seq2_7', 'S7_Seq2_8', 'S7_Seq2_9',
    'S7_Seq2_10', 'S7_Seq2_11', 'S7_Seq2_12', 'S7_Seq2_13', 'S8_Seq1_0', 'S8_Seq1_1', 'S8_Seq1_2', 'S8_Seq1_3',
    'S8_Seq1_4', 'S8_Seq1_5', 'S8_Seq1_6', 'S8_Seq1_7', 'S8_Seq1_8', 'S8_Seq1_9', 'S8_Seq1_10', 'S8_Seq1_11',
    'S8_Seq1_12', 'S8_Seq1_13', 'S8_Seq2_0', 'S8_Seq2_1', 'S8_Seq2_2', 'S8_Seq2_3', 'S8_Seq2_4', 'S8_Seq2_5',
    'S8_Seq2_6', 'S8_Seq2_7', 'S8_Seq2_8', 'S8_Seq2_9', 'S8_Seq2_10', 'S8_Seq2_11', 'S8_Seq2_12', 'S8_Seq2_13',
    'TS1', 'TS3', 'TS4'
]

camera_params = dict()
for sbj in subjects:
    if sbj.startswith('S'):
        subject, seq, cid = sbj.split('_')
        cid = int(cid)
        camera_meta = dict()
        camera_meta.update(mpii_3dhp_cameras_extrinsic_params[cid])
        camera_meta.update(mpii_3dhp_cameras_intrinsic_params[cid])
        camera_params[sbj] = [camera_meta]
    if sbj.startswith('T'):
        camera_meta = dict()
        camera_meta.update(mpii_3dhp_cameras_extrinsic_params[8])
        camera_meta.update(mpii_3dhp_cameras_intrinsic_params[8])
        camera_params[sbj] = [camera_meta]

h36m_skeleton = Human36mSkeleton(
    parents=[-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
             16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
    joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
    joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31]
)


class Mpii3dhpDataset(MocapDataset):
    def __init__(self, path, universal=False):
        super().__init__(fps=50, skeleton=h36m_skeleton)
        self.universal = universal
        self._cameras = copy.deepcopy(camera_params)
        for cameras in self._cameras.values():
            for i, cam in enumerate(cameras):
                for k, v in cam.items():
                    if k not in ['id', 'res_w', 'res_h']:
                        cam[k] = np.array(v, dtype='float32')

                # Normalize camera frame
                # if 'translation' in cam:
                #     cam['translation'] = cam['translation'] / 1000  # mm to meters
                # DONE IN PREPARE

        camera_info = dict()
        for subject in self._cameras:
            camera_info.setdefault(subject, list())
            for cam in self._cameras[subject]:
                if 'translation' not in cam:
                    continue
                K = np.eye(3, dtype=np.float64)
                K[0, 0] = cam['focal_length'][0]
                K[1, 1] = cam['focal_length'][1]
                K[0, 2] = cam['center'][0]
                K[1, 2] = cam['center'][1]

                R = cam['R']

                t = np.array(cam['translation'], dtype=np.float64).reshape(3, 1)
                camera_info[subject].append(CameraInfoPacket(P=None, K=K, R=R, t=t,
                                                             res_w=cam['res_w'], res_h=cam['res_h'],
                                                             azimuth=cam['azimuth'],
                                                             dist_coeff=None, undistort=False))
        self.camera_info = camera_info

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
            self._skeleton.remove_joints([4, 5, 9, 10, 11, 12, 13, 14, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31])
            kpt_index = [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16]
            for subject in self._data.keys():
                for action in self._data[subject].keys():
                    s = self._data[subject][action]  # if remove_static_joints:
                    if 'positions' in s:  # Bring the skeleton to 14 joints instead of the original 32
                        s['positions'] = s['positions'][:, kpt_index]  # self.remove_joints([])
        else:
            self._skeleton.remove_joints([4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31])
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
        origin_keypoints, origin_keypoints_metadata = keypoints['positions_2d'].item(), keypoints['metadata'].item()
        updated_keypoints, updated_keypoints_metadata = dict(), dict()

        if universal:
            human36m_kpt_index = [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16]
            updated_keypoints_metadata['layout_name'] = '3dhp'
            updated_keypoints_metadata['num_joints'] = len(human36m_kpt_index)
            updated_keypoints_metadata['keypoints_symmetry'] = [[4, 5, 6, 8, 9, 10], [1, 2, 3, 11, 12, 13]]
        else:
            human36m_kpt_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            updated_keypoints_metadata['layout_name'] = '3dhp'
            updated_keypoints_metadata['num_joints'] = len(human36m_kpt_index)
            updated_keypoints_metadata['keypoints_symmetry'] = [[4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]]

        for subject in origin_keypoints.keys():
            updated_keypoints.setdefault(subject, dict())
            for action in origin_keypoints[subject]:
                updated_keypoints[subject].setdefault(action, list())
                for cam_idx, kps in enumerate(origin_keypoints[subject][action]):
                    updated_keypoints[subject][action].append(kps['positions_2d'][:, human36m_kpt_index, :])

        return updated_keypoints, updated_keypoints_metadata