import ipdb
import json
import copy
import random
import numpy as np
from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from lib.camera.camera import CameraInfoPacket, catesian2homogenous, normalize_screen_coordinates
from lib.skeleton.skeleton import Skeleton


class PoseDataSet(Dataset):
    def __init__(self, data_config, model_config):
        self.data_config = data_config
        self.model_config = model_config

        self.skeleton = Skeleton(parents=[-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
                                  16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
                         joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                         joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])

        # load 3D
        self.data = self.load_world_3d_pose(self.data_config['GT_3D'])

        # if True, remove irrelevant 2D pose
        self.rm_irrlvnt_kpt = self.data_config['REMOVE_IRRELEVANT_KPTS'] or self.data_config['KEYPOINTS'] == 'universal'
        self.universal = self.data_config['KEYPOINTS'] == 'universal'
        if self.rm_irrlvnt_kpt:
            if self.universal:
                self.remove_joints([4, 5, 9, 10, 11, 12, 13, 14, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31])
            else:
                # Bring the skeleton to 17 joints instead of the original 32
                self.remove_joints([4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31])

                # Rewire shoulders to the correct parents
                self.skeleton._parents[11] = 8
                self.skeleton._parents[14] = 8

        self.left_joints, self.right_joints = self.set_3d_joints()

        # load camera info
        self.camera_meta = self.set_camera_meta()
        self.camera_info = self.set_camera_info()
        assert len(self.camera_meta) == len(self.camera_info)

        self.receptive_field = self.model_config['NUM_FRAMES']
        self.pad = (self.receptive_field - 1) // 2

    @staticmethod
    def load_world_3d_pose(dataset_path):
        """

        :param dataset_path:
        :return:
        """
        data = np.load(dataset_path, allow_pickle=True)['positions_3d'].item()
        _data = {}
        for subject, actions in data.items():
            _data[subject] = {}
            for action_name, positions in actions.items():
                _data[subject][action_name] = {
                    'positions': positions,
                }

        return _data

    def remove_joints(self, joints_to_remove):
        """

        :param joints_to_remove:
        :return:
        """
        kept_joints = self.skeleton.remove_joints(joints_to_remove)
        for subject in self.data.keys():
            for action in self.data[subject].keys():
                s = self.data[subject][action]
                if 'positions' in s:
                    s['positions'] = s['positions'][:, kept_joints]

    @abstractmethod
    def set_camera_meta(self):
        pass

    def get_camera_meta(self):
        return self.camera_meta

    def set_camera_info(self):
        camera_infos = list()
        for cam_idx in range(len(self.camera_meta)):
            cam = self.camera_meta[cam_idx]
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
            camera_info = CameraInfoPacket(
                P=None, K=K, R=R, t=t,
                res_w=cam['res_w'], res_h=cam['res_h'],
                azimuth=cam['azimuth'], dist_coeff=dist_coeff, undistort=False
            )
            camera_infos.append(camera_info)

        return camera_infos

    def get_camera_info(self):
        return self.camera_info

    def set_3d_joints(self):
        """

        :return:
        """
        joints_left, joints_right = list(self.skeleton.joints_left()), list(
            self.skeleton.joints_right())
        return joints_left, joints_right

    def get_3d_joints(self):
        return self.left_joints, self.right_joints

class TrainPoseDataSet(PoseDataSet):
    """
    dataset for multiple-gpu training
    """
    def __init__(self, data_config, model_config, flip_prob=0.5):
        super(TrainPoseDataSet, self).__init__(data_config, model_config)
        self.flip_prob = flip_prob
        self.subjects = self.data_config['TRAIN_SUBJECTS'].split(',')
        self.num_frames = self.set_num_frames()

    def set_num_frames(self):
        num_frame = 0
        for sbj in self.subjects:
            acts = self.data[sbj]
            for act in acts:
                pose_3d = acts[act]['positions']
                num_frame += pose_3d.shape[0]
        return num_frame * len(self.camera_meta)

    def get_num_frames(self):
        return self.num_frames

    def set_camera_meta(self):
        return json.load(open(self.data_config['TRAIN_CAMERA_PARAM'], 'r'))

    def __len__(self):
        return self.get_num_frames()

    def __getitem__(self, index):

        # (1) randomly select camera and create camera info packet
        cam_idx = random.randint(0, len(self.camera_meta)-1)
        camera_info = self.camera_info[cam_idx]

        # (2) randomly select subject
        sbj_idx = random.randint(0, len(self.subjects)-1)
        sbj = self.subjects[sbj_idx]

        # (3) randomly select action
        acts = list(self.data[sbj].keys())
        act_idx = random.randint(0, len(acts)-1)
        act = acts[act_idx]

        # (4) randomly select frames
        pose_3d = self.data[sbj][act]['positions']
        # frame_idx = random.randint(0, len(pose_3d)-self.receptive_field)
        frame_idx = random.randint(0, len(pose_3d)-1)

        start_frame_idx = max(0, frame_idx - self.pad)
        end_frame_idx = min(frame_idx + self.pad, len(pose_3d)-1)
        low_padding = start_frame_idx - (frame_idx - self.pad)
        high_padding = (frame_idx + self.pad) - end_frame_idx
        sampled_3d_world = pose_3d[start_frame_idx:end_frame_idx+1]
        target_3d_world = pose_3d[frame_idx:frame_idx+1]

        # (5) camera 3d pose
        if self.data_config['RAY_ENCODING']:
            # sampled_3d_camera = camera_info.world2normalized(sampled_3d_world)
            target_3d_world = camera_info.world2normalized(target_3d_world)
        else:
            # sampled_3d_camera = camera_info.world2camera(sampled_3d_world)
            target_3d_world = camera_info.world2camera(target_3d_world)

        # (6) image 2d pose
        sampled_2d_image = camera_info.project(catesian2homogenous(sampled_3d_world))
        if self.data_config['RAY_ENCODING']:
            frame_dim, kpt_dim, vec_dim = sampled_2d_image.shape
            kps_ray = np.zeros((frame_dim, kpt_dim, vec_dim + 1))
            kps_ray[:, :kpt_dim] = camera_info.get_cam_ray_given_uv(sampled_2d_image)
            sampled_2d_image = kps_ray
        else:
            sampled_2d_image[..., :2] = normalize_screen_coordinates(sampled_2d_image[..., :2], w=camera_info.res_w, h=camera_info.res_h)

        # (7) flip augmentation
        if np.random.rand() <= self.flip_prob:
            target_3d_world[:, :, 0] *= -1
            target_3d_world[:, self.left_joints + self.right_joints] = target_3d_world[:, self.right_joints + self.left_joints]
            sampled_2d_image[:, :, 0] *= -1
            sampled_2d_image[:, self.left_joints + self.right_joints] = sampled_2d_image[:, self.right_joints + self.left_joints]

        # (8) padding
        sampled_2d_image = np.pad(sampled_2d_image, ((low_padding, high_padding), (0, 0), (0, 0)), 'edge')

        target_3d_world = torch.from_numpy(target_3d_world.astype('float32'))
        sampled_2d_image = torch.from_numpy(sampled_2d_image.astype('float32'))
        cam_idx = torch.tensor(cam_idx).long()
        return target_3d_world, sampled_2d_image, cam_idx


class TestPoseDataSet(PoseDataSet):
    """
    dataset for multiple-gpu testing
    """
    def __init__(self, data_config, model_config):
        super(TestPoseDataSet, self).__init__(data_config, model_config)
        self.subjects = self.data_config['TEST_SUBJECTS'].split(',')
        self.pose_meta = self.set_pose_meta()

    def set_camera_meta(self):
        return json.load(open(self.data_config['TEST_CAMERA_PARAM'], 'r'))

    def set_pose_meta(self):
        pose_meta = list()
        for cam_idx in range(len(self.camera_meta)):
            for sbj in self.subjects:
                acts = self.data[sbj]
                for act in acts:
                    pose_meta.append((cam_idx, sbj, act,))
        return pose_meta

    def __len__(self):
        return len(self.pose_meta)

    def __getitem__(self, index):

        cam_idx, sbj, act = self.pose_meta[index]
        camera_info = self.camera_info[cam_idx]
        pose_3d = self.data[sbj][act]['positions']

        sampled_3d_world = pose_3d
        if self.data_config['RAY_ENCODING']:
            sampled_3d_camera = camera_info.world2normalized(sampled_3d_world)
        else:
            sampled_3d_camera = camera_info.world2camera(sampled_3d_world)

        # (6) image 2d pose
        sampled_2d_image = camera_info.project(catesian2homogenous(sampled_3d_world))
        if self.data_config['RAY_ENCODING']:
            frame_dim, kpt_dim, vec_dim = sampled_2d_image.shape
            kps_ray = np.zeros((frame_dim, kpt_dim, vec_dim + 1))
            kps_ray[:, :kpt_dim] = camera_info.get_cam_ray_given_uv(sampled_2d_image)
            sampled_2d_image = kps_ray
        else:
            sampled_2d_image[..., :2] = normalize_screen_coordinates(
                sampled_2d_image[..., :2],
                w=camera_info.res_w, h=camera_info.res_h
            )

        # (7) padding
        sampled_2d_image = np.pad(sampled_2d_image, ((self.pad, self.pad), (0, 0), (0, 0)), 'edge')

        sampled_3d_camera = torch.from_numpy(sampled_3d_camera.astype('float32'))
        sampled_2d_image = torch.from_numpy(sampled_2d_image.astype('float32'))
        return sampled_3d_camera, sampled_2d_image, cam_idx
