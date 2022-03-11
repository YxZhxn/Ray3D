import os
import copy
import numpy as np
from lib.utils.utils import deterministic_random
from lib.camera.camera import normalize_screen_coordinates


class Data:

    def __init__(self, data_config):
        """

        :param data_config:
        """
        self.data_config = data_config

        # if True, we load 3D pose
        self.gt_eval = self.data_config['WORLD_3D_GT_EVAL']
        # if True, remove irrelevant 2D pose
        self.rm_irrlvnt_kpt = self.data_config['REMOVE_IRRELEVANT_KPTS'] or self.data_config['KEYPOINTS'] == 'universal'

        # load 3D
        dataset_path_3d = self.data_config['GT_3D']
        self.load_world_3d_pose(dataset_path_3d)
        if self.data_config['RAY_ENCODING']:
            self.calculate_ray_3d_pose()
        else:
            self.calculate_camera_3d_pose()

        # load 2D
        dataset_path_2d = self.data_config['GT_2D']
        self.load_pixel_2d_pose(dataset_path_2d)
        self.file_names = self.load_view(dataset_path_2d, self.data_config['FRAME_PATH'])

        if self.data_config['INTRINSIC_ENCODING']:
            self.calculate_intrinsic_2d_pose()

        elif self.data_config['RAY_ENCODING']:
            self.calculate_ray_2d_pose()

        else:
            self.normalize_pixel_2d_pose()

        # sanity check
        self.sanity_check()

    # -------------------------------- #

    def load_world_3d_pose(self, dataset_path):
        """

        :param dataset_path:
        :return:
        """
        if self.data_config['DATASET'] == 'h36m_aug':
            from .h36m_aug_dataset import Human36mDataset
            self.dataset = Human36mDataset(
                dataset_path,
                self.data_config['CAMERA_PARAM'],
                camera_wise_performance=self.data_config['CAMERA_WISE_PERFORMANCE'] if 'CAMERA_WISE_PERFORMANCE' in self.data_config else False,
                universal=self.data_config['KEYPOINTS'] == 'universal'
            )

        else:
            raise ValueError('Invalid dataset: {}'.format(self.data_config['DATASET']))

    def calculate_camera_3d_pose(self):
        """
        convert 3D pose from world to camera and save in relative format with respect to pelvis
        :return:
        """
        if self.gt_eval:
            for subject in self.dataset.subjects():
                for action in self.dataset[subject].keys():
                    anim = self.dataset[subject][action]
                    if 'positions' in anim:
                        positions_3d = []

                        # Method of ours
                        for cam_idx, camera in enumerate(self.dataset.camera_info[subject]):
                            positions_3d.append(camera.world2camera(anim['positions']))
                        anim['positions_3d'] = positions_3d

    def calculate_ray_3d_pose(self):
        """
        convert 3D pose from world to intermediate space
        :return:
        """
        if self.gt_eval:
            for subject in self.dataset.subjects():
                for action in self.dataset[subject].keys():
                    anim = self.dataset[subject][action]
                    if 'positions' in anim:
                        positions_3d = []
                        for cam_idx, camera in enumerate(self.dataset.camera_info[subject]):
                            camera = self.dataset.camera_info[subject][cam_idx]
                            positions_3d.append(camera.world2normalized(anim['positions']))
                        anim['positions_3d'] = positions_3d

    def load_pixel_2d_pose(self, dataset_path):
        """
        load external 2D detections
        :return:
        """
        keypoints = np.load(dataset_path, allow_pickle=True)

        if self.rm_irrlvnt_kpt:
            self.keypoints, self.keypoints_metadata = self.remove_irrelevant_kpts(keypoints)
        else:
            self.keypoints, self.keypoints_metadata = keypoints['positions_2d'].item(), keypoints['metadata'].item()

    def load_view(self, dataset_path, frame_path):
        """
        load filename of detection bounding box
        :param dataset_path:
        :return:
        """
        keypoints = np.load(dataset_path, allow_pickle=True)
        keypoints = keypoints['positions_2d'].item()

        file_names = dict()
        for subject in keypoints.keys():
            file_names.setdefault(subject, dict())
            for action in keypoints[subject].keys():
                file_names[subject].setdefault(action, list())
                for cam_idx in range(len(keypoints[subject][action])):
                    if isinstance(keypoints[subject][action][cam_idx], dict):
                        names = keypoints[subject][action][cam_idx]['file_name']
                        frames = list()
                        for name in names:
                            if self.data_config['DATASET'] == '3dhp':
                                if subject.startswith('S'):
                                    sbj, seq, cid = subject.split('_')
                                    frame_name = os.path.join(frame_path, sbj, seq,
                                                              'imageSequence', 'video_{}'.format(cid), name)
                                if subject.startswith('T'):
                                    frame_name = os.path.join(frame_path, subject,
                                                              'imageSequence', name)

                            try:
                                assert os.path.exists(frame_name)
                            except:
                                import ipdb
                                ipdb.set_trace()
                            frames.append(frame_name)
                        file_names[subject][action].append(frames)
                    else:
                        return None

        return file_names

    def remove_irrelevant_kpts(self, keypoints):
        return self.dataset.remove_irrelevant_kpts(keypoints, self.data_config['KEYPOINTS'] == 'universal')

    def normalize_pixel_2d_pose(self):
        """
        normalize external 2D detections
        :return:
        """
        for subject in self.dataset.subjects():
            for action in self.keypoints[subject]:
                for cam_idx, kps in enumerate(self.keypoints[subject][action]):
                    # Normalize camera frame
                    cam = self.dataset.camera_info[subject][cam_idx]
                    kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam.res_w, h=cam.res_h)
                    self.keypoints[subject][action][cam_idx] = kps

    def calculate_intrinsic_2d_pose(self):
        """
        normalize external 2D detections
        :return:
        """
        for subject in self.dataset.subjects():
            for action in self.keypoints[subject]:
                for cam_idx, kps in enumerate(self.keypoints[subject][action]):
                    camera = self.dataset.camera_info[subject][cam_idx]
                    self.keypoints[subject][action][cam_idx] = camera.encode_uv_with_intrinsic(kps)

    def calculate_ray_2d_pose(self):
        """
        normalize external 2D detections
        :return:
        """
        for subject in self.dataset.subjects():
            for action in self.keypoints[subject]:
                for cam_idx, kps in enumerate(self.keypoints[subject][action]):
                    camera = self.dataset.camera_info[subject][cam_idx]
                    frame_dim, kpt_dim, vec_dim = kps.shape
                    kps_ray = np.zeros((frame_dim, kpt_dim, vec_dim + 1))
                    kps_ray[:, :kpt_dim] = camera.get_cam_ray_given_uv(kps)
                    self.keypoints[subject][action][cam_idx] = kps_ray

    def sanity_check(self):
        """
        make sure that both number of 2D detections and of 3D ground-truths are aligned.
        :return:
        """
        if self.gt_eval:
            for subject in self.dataset.subjects():
                assert subject in self.keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
                for action in self.dataset[subject].keys():
                    assert action in self.keypoints[subject], \
                        'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
                    if 'positions_3d' not in self.dataset[subject][action]:
                        continue

                    for cam_idx in range(len(self.keypoints[subject][action])):

                        # We check for >= instead of == because some videos in H3.6M contain extra frames
                        mocap_length = self.dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                        assert self.keypoints[subject][action][cam_idx].shape[0] >= mocap_length

                        if self.keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                            # Shorten sequence
                            self.keypoints[subject][action][cam_idx] = self.keypoints[subject][action][cam_idx][
                                                                       :mocap_length]

                    assert len(self.keypoints[subject][action]) == len(self.dataset[subject][action]['positions_3d'])

    # -------------------------------- #

    def get_dataset(self):
        """
        retrieve dataset which contains world 3D pose, camera 3D pose and camera parameters
        :return:
        """
        return self.dataset

    def get_keypoints(self):
        """
        retrieve pixel 2D pose
        :return:
        """
        return self.keypoints

    def get_2d_kpts(self):
        """

        :return:
        """
        keypoints_symmetry = self.keypoints_metadata['keypoints_symmetry']
        kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
        return kps_left, kps_right

    def get_3d_joints(self):
        """

        :return:
        """
        joints_left, joints_right = list(self.dataset.skeleton().joints_left()), list(
            self.dataset.skeleton().joints_right())
        return joints_left, joints_right

    # -------------------------------- #

    def fetch_via_subject(self, subjects, action_filter=None, subset=1, parse_3d_poses=True):
        """

        :param subjects:
        :param action_filter:
        :param subset:
        :param parse_3d_poses:
        :return:
        """
        out_poses_3d = []
        out_poses_2d = []
        out_camera_params = []

        for subject in subjects:
            for action in self.keypoints[subject].keys():
                poses_2d = self.keypoints[subject][action]
                poses_3d = self.dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_2d)):  # Iterate across cameras
                    out_poses_2d.append(copy.deepcopy(poses_2d[i]))
                    out_poses_3d.append(copy.deepcopy(poses_3d[i]))
                    camera = self.dataset.camera_info[subject][i]
                    out_camera_params.append([camera] * poses_2d[i].shape[0])

        if len(out_camera_params) == 0:
            out_camera_params = None
        if len(out_poses_3d) == 0:
            out_poses_3d = None

        stride = self.data_config['DOWNSAMPLE']
        if subset < 1:
            for i in range(len(out_poses_2d)):
                n_frames = int(round(len(out_poses_2d[i]) // stride * subset) * stride)
                start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
                out_poses_2d[i] = out_poses_2d[i][start:start + n_frames:stride]
                out_camera_params[i] = out_camera_params[i][start:start + n_frames:stride]
                out_poses_3d[i] = out_poses_3d[i][start:start + n_frames:stride]
        elif stride > 1:
            # Downsample as requested
            for i in range(len(out_poses_2d)):
                out_poses_2d[i] = out_poses_2d[i][::stride]
                out_camera_params[i] = out_camera_params[i][::stride]
                out_poses_3d[i] = out_poses_3d[i][::stride]

        return out_camera_params, out_poses_3d, out_poses_2d

    def fetch_via_action(self, actions, camera_idx=None):
        """

        :param actions:
        :return:
        """
        out_poses_3d = []
        out_poses_2d = []
        out_camera_params = []

        for subject, action in actions:
            poses_2d = self.keypoints[subject][action]
            poses_3d = self.dataset[subject][action]['positions_3d']
            assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
            for i in range(len(poses_2d)):  # Iterate across cameras
                if camera_idx is not None:
                    if i != camera_idx:
                        continue
                out_poses_2d.append(copy.deepcopy(poses_2d[i]))
                out_poses_3d.append(copy.deepcopy(poses_3d[i]))
                camera = self.dataset.camera_info[subject][i]
                out_camera_params.append(camera)

        if len(out_poses_3d) == 0:
            out_poses_3d = None
        if len(out_camera_params) == 0:
            out_camera_params = None

        stride = self.data_config['DOWNSAMPLE']
        if stride > 1:
            # Downsample as requested
            for i in range(len(out_poses_2d)):
                out_poses_2d[i] = out_poses_2d[i][::stride]
                if out_poses_3d is not None:
                    out_poses_3d[i] = out_poses_3d[i][::stride]

        return out_camera_params, out_poses_3d, out_poses_2d
