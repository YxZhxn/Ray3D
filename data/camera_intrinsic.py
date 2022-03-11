import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.camera.camera import CameraInfoPacket, catesian2homogenous

import ipdb
import copy
import json
import numpy as np

from camera_augmentation import h36m_cameras_intrinsic_params
from camera_augmentation import init_camera_h36m, get_camera_pose, camera_translation, mkdirs
from camera_augmentation import convertdegree2euler, rotate_camera, get_intrinsic, check_in_frame

H36M_KPT_IDX = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

RES_W, RES_H = 1000, 1000

AXIS_X, AXIS_Y, AXIS_Z = [1, 0, 0], [0, 1, 0], [0, 0, 1]

TRAIN_SUBJECTS = ['S1', 'S5', 'S6', 'S7', 'S8']

TEST_SUBJECTS = ['S9', 'S11']

IDX_TO_CAMERA = {
    0: '54138969',
    1: '55011271',
    2: '58860488',
    3: '60457274'
}

if __name__ == '__main__':

    FILE_TO_3D_POSE = '/ssd/yzhan/data/benchmark/3D/h36m/annotations/gt/data_3d_h36m.npz'
    PATH_TO_OUTPUT = '/ssd/ray3d/camera.intrinsic'

    SUBJECT = 'S1'
    YAW = 0
    TRANSLATION = 2.0
    PITCH = 0
    CENTER_POINT = [0, 0, 1.8]
    FOCAL_LENGTH_BAIS_RANGE = (np.arange(-50, 50, 10)).tolist()
    CENTER_POINT_BAIS_RANGE = (np.arange(-50, 50, 10)).tolist()
    cam_idx = 1  # '55011271'

    camera_info = init_camera_h36m()
    camera_pose = get_camera_pose(camera_info)
    pose_3d = np.load(FILE_TO_3D_POSE, allow_pickle=True)['positions_3d'].item()

    selected_cam = copy.deepcopy(camera_info[SUBJECT][cam_idx])
    Rw2c = selected_cam.Rw2c
    Tw2c = selected_cam.Tw2c
    t = np.array(CENTER_POINT).reshape(3, 1)

    # translation
    Tw2c_translation_aumented = camera_translation(Tw2c, t, TRANSLATION)

    # rotation
    yaw = convertdegree2euler(YAW)
    Rw2c_rotation_augmented, Tw2c_rotation_augmented = rotate_camera(Rw2c, Tw2c_translation_aumented, t,
                                                                     np.array(AXIS_Z), yaw)

    # pitch
    pitch = convertdegree2euler(PITCH)
    camera_position = - np.dot(Rw2c_rotation_augmented.T, Tw2c_rotation_augmented)

    axis = np.array([-camera_position[1][0], camera_position[0][0], 0])
    Rw2c_pitch_augmented, Tw2c_pitch_augmented = rotate_camera(Rw2c_rotation_augmented, Tw2c_rotation_augmented,
                                                               t, axis, pitch)
    camera_position = - np.dot(Rw2c_pitch_augmented.T, Tw2c_pitch_augmented)

    # change the focal length of intrinsic
    for f_bais in FOCAL_LENGTH_BAIS_RANGE:

        for c_bias in CENTER_POINT_BAIS_RANGE:

            K, dist_coeff = get_intrinsic(cam_idx, fx_bais=f_bais, fy_bais=f_bais, cx_bias=c_bias, cy_bias=c_bias)
            camera_augmented = CameraInfoPacket(
                P=None, K=K, R=Rw2c_pitch_augmented, t=Tw2c_pitch_augmented, dist_coeff=dist_coeff
            )

            # sanity check
            invalid = False
            pose_2d = {}
            for sbj in pose_3d.keys():
                pose_2d.setdefault(sbj, dict())
                for act in pose_3d[sbj].keys():
                    pose_2d[sbj].setdefault(act, list())
                    kpt_3d = pose_3d[sbj][act][:, H36M_KPT_IDX]
                    kpt_3d_hom = catesian2homogenous(kpt_3d)
                    kpt_2d = camera_augmented.project(kpt_3d_hom)
                    pose_2d[sbj][act].append(kpt_2d)
                    if not check_in_frame(kpt_2d, RES_W, RES_H):
                        invalid = True
                        break
                if invalid:
                    break

            if invalid:
                del pose_2d
                continue

            print('\tcam_idx:{}, FBAIS: {}, CBAIS: {}'.format(cam_idx, f_bais, c_bias))

            CAMERA_PARAM_PATH = os.path.join(PATH_TO_OUTPUT, '{}'.format('json'))
            POSE_2D_PATH = os.path.join(PATH_TO_OUTPUT, '{}'.format('npz'))
            mkdirs(CAMERA_PARAM_PATH)
            mkdirs(POSE_2D_PATH)

            cameras_params = []
            camera_param = {}

            camera_param['id'] = h36m_cameras_intrinsic_params[cam_idx]['id']
            camera_param['center'] = [camera_augmented.K[0, 2], camera_augmented.K[1, 2]]
            camera_param['focal_length'] = [camera_augmented.K[0, 0], camera_augmented.K[1, 1]]
            camera_param['radial_distortion'] = h36m_cameras_intrinsic_params[cam_idx]['radial_distortion']
            camera_param['tangential_distortion'] = h36m_cameras_intrinsic_params[cam_idx]['tangential_distortion']
            camera_param['res_w'] = h36m_cameras_intrinsic_params[cam_idx]['res_w']
            camera_param['res_h'] = h36m_cameras_intrinsic_params[cam_idx]['res_h']
            camera_param['azimuth'] = h36m_cameras_intrinsic_params[cam_idx]['azimuth']

            camera_param['R'] = Rw2c_pitch_augmented.tolist()
            camera_param['translation'] = Tw2c_pitch_augmented.tolist()

            # save camera parameter
            CAMERA_FILENAME = os.path.join(CAMERA_PARAM_PATH,
                                           'TRANSLATION{}_YAW{}_PITCH{}_CAM{}_FBAIS{}_CBIAS{}.json'.format(
                                               TRANSLATION, YAW,
                                               PITCH, cam_idx, f_bais, c_bias))
            with open(CAMERA_FILENAME, 'w') as file_obj:
                json.dump([camera_param], file_obj, indent=4)

            # save projected 2d pose
            POSE2D_FILENAME = os.path.join(POSE_2D_PATH,
                                           'TRANSLATION{}_YAW{}_PITCH{}_CAM{}_FBAIS{}_CBIAS{}.npz'.format(
                                               TRANSLATION, YAW,
                                               PITCH, cam_idx, f_bais, c_bias))
            METADATA = {
                'layout': 'h36m_aug',
                'num_joints': 17,
                'keypoints_symmetry': [[4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]]
            }
            np.savez(POSE2D_FILENAME, metadata=METADATA, positions_2d=pose_2d)
            del pose_2d