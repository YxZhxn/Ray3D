import os
import sys
import copy
import ipdb
import json
import mat73
import numpy as np
import scipy.io as sio
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.camera.camera import CameraInfoPacket, catesian2homogenous


rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])  # rotate along the x axis 90 degrees


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
                [0.9650164, 0.00488022, 0.262144],
                [-0.004488356, -0.9993728, 0.0351275],
                [0.262151, -0.03507521, -0.9643893]
            ]
    },
    {
        'translation': [-1.429856, 0.7381779, 4.897966],
        'R':
            [
                [0.6050639, -0.02184232, 0.7958773],
                [-0.22647, -0.9630526, 0.1457429],
                [0.7632883, -0.2684261, -0.587655]
            ]
    },
    {
        'translation': [0.05725702, 1.307287, 2.7998220000000003],
        'R':
            [
                [-0.3608179, -0.009492658, 0.932588],
                [-0.0585942, -0.9977421, -0.03282591],
                [0.9307939, -0.06648842, 0.359447]
            ]
    },
    {
        'translation': [-0.2848168, 0.8079184, 3.1771599999999998],
        'R':
            [
                [-0.0721105, -0.04817664, 0.9962325],
                [-0.4393254, -0.8951841, -0.07508985],
                [0.895429, -0.443085, 0.04338695]
            ]
    },
    {
        'translation': [-1.563911, 0.8019607999999999, 3.5173159999999997],
        'R':
            [
                [0.3737275, 0.09688602, 0.9224646],
                [-0.009716132, -0.9940662, 0.1083427],
                [0.9274878, -0.04945343, -0.3705685]
            ]
    },
    {
        'translation': [0.35841340000000005, 0.9945657999999999, 3.439832],
        'R':
            [
                [-0.3521056, 0.01328985, -0.9358659],
                [-0.04961938, -0.9987582, 0.004485628],
                [-0.9346441, 0.0480165, 0.3523278]
            ]
    },
    {
        'translation': [0.5694388, 0.528871, 3.6873690000000003],
        'R':
            [
                [-0.9150326, -0.04843184, 0.4004618],
                [-0.1804886, -0.8386868, -0.5138369],
                [0.3607481, -0.5424563, 0.7586845]
            ]
    },
    {
        'translation': [1.378866, 1.270781, 2.631567],
        'R':
            [
                [-0.9995936, 0.02847456, 0.001368653],
                [-0.02843213, -0.9992908, 0.0246889],
                [0.002070688, 0.02463995, 0.9996943]
            ]
    },
    {
        'translation': [0.2213543, 0.65987, 3.644688],
        'R':
            [
                [0.000575281, 0.06160985, -0.9981001],
                [0.2082146, -0.9762325, -0.06013997],
                [-0.978083, -0.2077844, -0.01338968]
            ]
    },
    {
        'translation': [0.38862169999999996, 0.1375452, 4.216635],
        'R':
            [
                [0.04176839, 0.00780962, -0.9990969],
                [0.5555364, -0.831324, 0.01672664],
                [-0.8304425, -0.5557333, -0.03906159]
            ]
    },
    {
        'translation': [1.167962, 0.6176362000000001, 4.472351],
        'R':
            [
                [-0.8970265, 0.1361548, -0.4204822],
                [0.09417118, -0.8706428, -0.4828178],
                [-0.4318278, -0.4726976, 0.7681679]
            ]
    },
    {
        'translation': [0.1348272, 0.2515094, 4.570244],
        'R':
            [
                [0.9170455, 0.1972746, -0.3465695],
                [0.1720879, 0.5882171, 0.7901813],
                [0.3597408, -0.7842726, 0.5054733]
            ]
    },
    {
        'translation': [0.4124695, 0.5327588, 4.887095],
        'R':
            [
                [-0.7926738, 0.1323657, 0.5951031],
                [-0.396246, 0.6299778, -0.66792],
                [-0.4633114, -0.7652499, -0.4469175]
            ]
    },
    {
        'translation': [0.8671278, 0.8274571999999999, 3.985159],
        'R':
            [
                [-0.8701088, -0.09522671, -0.4835728],
                [0.4120245, 0.3978655, -0.8197188],
                [0.270456, -0.9124883, -0.3069505]
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


def read_ann(ann_file, mode):
    """

    :param ann_file:
    :param mode:
    :return:
    """
    if mode == 'train':
        return sio.loadmat(ann_file)
    if mode == 'test':
        return mat73.loadmat(ann_file)


def read_cali(cali_file, vid_idx, mode):
    """

    :param cali_file:
    :param vid_idx:
    :return:
    """
    Ks, Rs, Ts = [], [], []
    if mode == 'train':
        file = open(cali_file, 'r')
        content = file.readlines()
        for vid_i in vid_idx:
            K = np.array([float(s) for s in content[vid_i * 7 + 5][11:-2].split()])
            K = np.reshape(K, (4, 4))[:3, :3]
            RT = np.array([float(s) for s in content[vid_i * 7 + 6][11:-2].split()])
            RT = np.reshape(RT, (4, 4))
            R = RT[:3, :3]
            R = R @ np.linalg.inv(rot)
            T = RT[:3, 3] / 1000  # mm to m
            Ks.append(K)
            Rs.append(R)
            Ts.append(T)

    if mode == 'test':
        raise NotImplementedError

    return Ks, Rs, Ts


if __name__ == '__main__':
    # REFERENCE: https://github.com/nkolot/SPIN/blob/master/datasets/preprocess/mpi_inf_3dhp.py

    data_root = '/ssd/yzhan/data/benchmark/3D/mpi_inf_3dhp'
    res_w = 2048
    res_h = 2048

    # train
    train_subjects = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
    sequences = ['Seq1', 'Seq2']
    video_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    train_kpt_idx = [4, 23, 24, 25, 18, 19, 20, 3, 5, 6, 7, 9, 10, 11, 14, 15, 16]

    # test
    test_subjects = ['TS1', 'TS3', 'TS4']  # drop TS2, due to inaccurate extrinsic
    test_kpt_idx = [14, 8, 9, 10, 11, 12, 13, 15, 1, 16, 0, 5, 6, 7, 2, 3, 4]

    METADATA = {
        'layout': '3dhp',
        'num_joints': 17,
        'keypoints_symmetry': [[4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]]
    }
    data_3d = {}
    data_2d = {}
    intrinsics = {}
    extrinsics = {}

    for sbj in train_subjects:

        for seq in sequences:

            ann_meta = read_ann(os.path.join(data_root, sbj, seq, 'annot.mat'), mode='train')
            valid_cameras = ann_meta['cameras'].reshape(-1).tolist()
            valid_frames = ann_meta['frames'].reshape(-1).tolist()

            kpts_2d = ann_meta['annot2']
            kpts_3d = ann_meta['annot3']

            Ks, Rs, Ts = read_cali(os.path.join(data_root, sbj, seq, 'camera.calibration'), video_list, mode='train')
            assert len(Ks) == len(Rs) == len(Ts) == len(valid_cameras), 'camera miss match'

            for cam_idx in valid_cameras:
                subject = '{}_{}_{}'.format(sbj, seq, cam_idx)

                joints_2d = kpts_2d[cam_idx, 0][:len(valid_frames)].reshape(len(valid_frames), -1, 2)[:, train_kpt_idx]
                joints_3d = kpts_3d[cam_idx, 0][:len(valid_frames)].reshape(len(valid_frames), -1, 3)[:, train_kpt_idx]
                joints_3d /= 1000  # mm to m

                valid_joints_2d = list()
                valid_joints_3d = list()
                valid_file_names = list()
                num_invalid_frame = 0
                for frame_idx in range(len(valid_frames)):
                    joint_2d = joints_2d[frame_idx]
                    joint_3d = joints_3d[frame_idx]
                    x_in = np.logical_and(joint_2d[:, 0] < res_w, joint_2d[:, 0] >= 0)
                    y_in = np.logical_and(joint_2d[:, 1] < res_h, joint_2d[:, 1] >= 0)
                    ok_pts = np.logical_and(x_in, y_in)
                    if np.sum(ok_pts) < len(train_kpt_idx):
                        num_invalid_frame += 1
                        continue
                    frame_name = os.path.join(data_root, sbj, seq, 'imageSequence',
                                              'video_{}'.format(cam_idx), 'img_%06d.jpg' % (frame_idx + 1))
                    if not os.path.exists(frame_name):
                        num_invalid_frame += 1
                        continue
                    valid_joints_2d.append(joint_2d)
                    valid_joints_3d.append(joint_3d)
                    valid_file_names.append('img_%06d.jpg' % (frame_idx + 1))

                print('sbj -> {}, seq -> {}, camera -> {}, total frames -> {}, invalid frames -> {}'.format(
                    sbj, seq, cam_idx, len(valid_frames), num_invalid_frame)
                )

                valid_joints_2d = np.array(valid_joints_2d)
                valid_joints_3d = np.array(valid_joints_3d)
                assert valid_joints_2d.shape[0] == valid_joints_3d.shape[0] == len(valid_frames) - num_invalid_frame

                data_3d.setdefault(subject, dict())
                data_3d[subject].setdefault('Action', list())
                data_3d[subject]['Action'] = valid_joints_3d

                data_2d.setdefault(subject, dict())
                data_2d[subject].setdefault('Action', list())
                data_2d[subject]['Action'].append(
                    {
                        'file_name': valid_file_names,
                        'positions_2d': valid_joints_2d
                    }
                )

                intrinsics.setdefault(subject, [Ks[cam_idx].tolist()])
                extrinsics.setdefault(subject, [Rs[cam_idx].tolist(), Ts[cam_idx].tolist()])

    for sbj in test_subjects:
        ann_meta = read_ann(os.path.join(data_root, sbj, 'annot_data.mat'), mode='test')
        valid_frames = ann_meta['valid_frame'].reshape(-1).tolist()

        kpts_2d = ann_meta['annot2'].transpose(2, 1, 0)[:, test_kpt_idx]
        kpts_3d = ann_meta['annot3'].transpose(2, 1, 0)[:, test_kpt_idx]
        kpts_3d /= 1000  # mm to m

        valid_joints_2d = list()
        valid_joints_3d = list()
        valid_file_names = list()
        num_invalid_frame = 0
        for frame_idx, flag in enumerate(valid_frames):
            if flag == 0:
                num_invalid_frame += 1
                continue

            joint_2d = kpts_2d[frame_idx]
            joint_3d = kpts_3d[frame_idx]
            x_in = np.logical_and(joint_2d[:, 0] < res_w, joint_2d[:, 0] >= 0)
            y_in = np.logical_and(joint_2d[:, 1] < res_h, joint_2d[:, 1] >= 0)
            ok_pts = np.logical_and(x_in, y_in)
            if np.sum(ok_pts) < len(train_kpt_idx):
                num_invalid_frame += 1
                continue
            frame_name = os.path.join(data_root, sbj, 'imageSequence', 'img_%06d.jpg' % (frame_idx + 1))
            if not os.path.exists(frame_name):
                num_invalid_frame += 1
                continue
            valid_joints_2d.append(joint_2d)
            valid_joints_3d.append(joint_3d)
            valid_file_names.append('img_%06d.jpg' % (frame_idx + 1))

        print('sbj -> {}, total frames -> {}, invalid frames -> {}'.format(
            sbj, len(valid_frames), num_invalid_frame)
        )

        valid_joints_2d = np.array(valid_joints_2d)
        valid_joints_3d = np.array(valid_joints_3d)
        try:
            assert valid_joints_2d.shape[0] == valid_joints_3d.shape[0] == len(valid_frames) - num_invalid_frame
        except:
            ipdb.set_trace()

        data_3d.setdefault(sbj, dict())
        data_3d[sbj].setdefault('Action', list())
        data_3d[sbj]['Action'] = valid_joints_3d

        data_2d.setdefault(sbj, dict())
        data_2d[sbj].setdefault('Action', list())
        data_2d[sbj]['Action'].append(
            {
                'file_name': valid_file_names,
                'positions_2d': valid_joints_2d
            }
        )

    _cameras = copy.deepcopy(camera_params)
    for cameras in _cameras.values():
        for i, cam in enumerate(cameras):
            for k, v in cam.items():
                if k not in ['id', 'res_w', 'res_h']:
                    cam[k] = np.array(v, dtype='float32')

    camera_info = dict()
    for subject in _cameras:
        camera_info.setdefault(subject, list())
        for cam in _cameras[subject]:
            if 'translation' not in cam:
                continue
            K = np.eye(3, dtype=np.float)
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

    new_camera_info = dict()
    for subject in _cameras:
        new_camera_info.setdefault(subject, list())
        for cam in _cameras[subject]:
            if 'translation' not in cam:
                continue
            K = np.eye(3, dtype=np.float)
            K[0, 0] = cam['focal_length'][0]
            K[1, 1] = cam['focal_length'][1]
            K[0, 2] = cam['center'][0]
            K[1, 2] = cam['center'][1]

            R = cam['R']
            R = R @ np.linalg.inv(rot)
            t = np.array(cam['translation'], dtype=np.float64).reshape(3, 1)

            if subject.startswith('S'):
                cid = int(subject.split('_')[-1])
            else:
                cid = 8
            try:
                assert np.sum(K - Ks[cid]) < 1e-3
            except:
                ipdb.set_trace()
            assert np.sum(R - Rs[cid]) < 1e-6
            assert np.sum(t.reshape(3) - Ts[cid]) < 1e-6

            new_camera_info[subject].append(CameraInfoPacket(P=None, K=K, R=R, t=t,
                                                             res_w=cam['res_w'], res_h=cam['res_h'],
                                                             azimuth=cam['azimuth'],
                                                             dist_coeff=None, undistort=False))

    for ky in subjects:
        joint_2d, file_names = data_2d[ky]['Action'][0]['positions_2d'], data_2d[ky]['Action'][0]['file_name']
        joint_3d = data_3d[ky]['Action']
        cam = camera_info[ky][0]
        new_cam = new_camera_info[ky][0]
        world_3d = cam.camera2world(joint_3d)
        world_3d_update = world_3d.copy()
        for idx in range(world_3d.shape[0]):
            world_3d_update[idx] = (rot @ world_3d[idx].T).T
        projected_2d = new_cam.project(catesian2homogenous(world_3d_update))
        error = np.sum(joint_2d - projected_2d)
        print('{} error: {}'.format(ky, error/world_3d_update.shape[0]))
        data_3d[ky]['Action'] = world_3d_update

    np.savez(os.path.join(data_root, 'data_2d_3dhp_gt.npz'), metadata=METADATA, positions_2d=data_2d)
    np.savez(os.path.join(data_root, 'data_3d_3dhp.npz'), positions_3d=data_3d)
    json.dump(intrinsics, open(os.path.join(data_root, 'intrinsic.json'), 'w'), indent=4)
    json.dump(extrinsics, open(os.path.join(data_root, 'extrinsic.json'), 'w'), indent=4)