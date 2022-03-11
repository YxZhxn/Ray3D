


import cv2
import math
import torch
import numpy as np
from operator import mul


def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2

    if isinstance(X, np.ndarray):
        # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
        return X / w * 2 - [1, h / w]
    else:
        return X / w * 2 - torch.tensor([1, h / w]).to(X.device)


def image_coordinates(X, w, h):
    assert X.shape[-1] == 2
    if isinstance(X, np.ndarray):
        # Reverse camera frame normalization
        return (X + [1, h / w]) * w / 2
    else:
        return (X + torch.tensor([1, h / w]).to(X.device)) * w / 2


def euler2rotation(euler):
    """
    Calculate rotation matrix when euler angles are provided
    Reference: https://learnopencv.com/rotation-matrix-to-euler-angles/
    :param euler: numpy array with shape of (3, 1)
    :return: numpy array with shape of (3, 3)
    """
    assert euler.shape == (3, 1)
    return cv2.Rodrigues(euler)[0]


def rotation2euler(rot):
    """
    Calculate euler angles when rotation matrix is provided
    Reference: https://learnopencv.com/rotation-matrix-to-euler-angles/
    :param rot: numpy array with shape of (3, 3)
    :return: numpy array with shape of (3, 1)
    """
    assert rot.shape == (3, 3)
    return cv2.Rodrigues(rot)[0]


def rotation2quaternion(rot):
    """
    Calculate quaternion when rotation matrix is provided
    Reference: https://gist.github.com/shubh-agrawal/76754b9bfb0f4143819dbd146d15d4c8
    Reference: http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    :param rot: numpy matrix with shape of 3 x 3
    :return: numpy array with shape of (4, )
    """
    assert rot.shape == (3, 3)
    quaternion = np.zeros(4)
    trace = np.trace(rot)
    if trace > 0.:
        s = np.sqrt(trace + 1)
        quaternion[3] = s * 0.5
        s = 0.5 / s
        quaternion[0] = (rot[2, 1] - rot[1, 2]) * s
        quaternion[1] = (rot[0, 2] - rot[2, 0]) * s
        quaternion[2] = (rot[1, 0] - rot[0, 1]) * s

    else:
        i = np.argmax(rot.diagonal())
        j = (i + 1) % 3
        k = (i + 2) % 3

        s = np.sqrt(rot[i, i] - rot[j, j] - rot[k, k] + 1.)
        quaternion[i] = s * 0.5
        s = 0.5 / s
        quaternion[3] = (rot[k, j] - rot[j, k]) * s
        quaternion[j] = (rot[j, i] + rot[i, j]) * s
        quaternion[k] = (rot[k, i] + rot[i, k]) * s

    return quaternion


def euler2quaternion(euler):
    """
    Calculate quaternion when euler angles are provided
    :param euler: numpy array with shape of (3, 1)
    :return: numpy array with shape of (4, )
    """
    rot = euler2rotation(euler)
    quaternion = rotation2quaternion(rot)
    return quaternion


def quaternion2rotation(quat):
    """
    Calculate rotation matrix when quaternion is provided
    Reference: https://marc-b-reynolds.github.io/quaternions/2017/08/08/QuatRotMatrix.html
    :param quat: numpy array with shape of (4, )
    :return: numpy array with shape of (3, 3)
    """
    assert quat.shape == (4, )
    rot = np.zeros((3, 3))

    x = quat[0]
    y = quat[1]
    z = quat[2]
    w = quat[3]

    tx = 2 * x
    ty = 2 * y
    tz = 2 * z
    xx = tx * x
    yy = ty * y
    zz = tz * z
    xy = ty * x
    xz = tz * x
    yz = ty * z
    wx = tx * w
    wy = ty * w
    wz = tz * w

    rot[0, 0] = 1. - (yy + zz)
    rot[1, 1] = 1. - (xx + zz)
    rot[2, 2] = 1. - (xx + yy)
    rot[1, 0] = xy + wz
    rot[0, 1] = xy - wz
    rot[2, 0] = xz - wy
    rot[0, 2] = xz + wy
    rot[2, 1] = yz + wx
    rot[1, 2] = yz - wx

    return rot


def quaternion2euler(quat):
    """
    Calculate euler angles when quaternion is provided
    :param quat: numpy array with shape of (4, )
    :return: numpy array with shape of (3, 1)
    """
    rot = quaternion2rotation(quat)
    euler = rotation2euler(rot)
    return euler


def catesian2homogenous(arr_cart):
    """
    Convert catesian to homogenous
    :param arr_cart:
    :return:
    """
    if isinstance(arr_cart, np.ndarray):
        arr_hom = np.concatenate((arr_cart, np.ones(arr_cart.shape[:-1] + (1,))), axis=-1)
    else:
        arr_hom = torch.cat((arr_cart, torch.ones(arr_cart.shape[:-1] + (1,)).to(arr_cart.device)), dim=-1)
    return arr_hom


def homogenous2catesian(arr_hom):
    """
    Convert homogenous to catesian
    :param arr_hom:
    :return:
    """
    if isinstance(arr_hom, np.ndarray):
        arr_hom[..., :-1] /= np.repeat(arr_hom[..., -1:], arr_hom.shape[-1]-1, axis=-1)
        arr_cart = arr_hom[..., :-1]
    else:
        org_dim = arr_hom.shape
        new_dim = [1 for _ in org_dim[:-1]] + [org_dim[-1] -1,]
        arr_hom[..., :-1] = arr_hom[..., :-1] / arr_hom[..., -1:].repeat(*new_dim)
        arr_cart = arr_hom[..., :-1]
    return arr_cart


def dotproduct(v1, v2):
    """

    :param v1:
    :param v2:
    :return:
    """
    return sum(map(mul, v1, v2))


def length(v):
    """

    :param v:
    :return:
    """
    return math.sqrt(dotproduct(v, v))


def angle(v1, v2):
    """

    :param v1:
    :param v2:
    :return:
    """
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


class CameraInfoPacket(object):

    def __init__(self, P=None, K=None, R=None, t=None, dist_coeff=None,
                 res_w=None, res_h=None, azimuth=None, undistort=True, lite=False):
        """
        P = K[R|t]
        One must either supply P or K, R, t.
        :param P: camera matrix, (3, 4)
        :param K: intrinsic matrix, (3, 3)
        :param R: rotation matrix, (3, 3)
        :param t: translation vector, (3, 1)
        :param dist_coeff: distortion coefficient, (5,)
        :param res_w: pixel width of frame
        :param res_h: pixel height of frame
        :param azimuth: azimuth for visualization
        :param undistort: flag to undo the distortion
        :param lite: lite version of CameraInfoPacket
        """

        if P is None:
            assert K.shape == (3, 3)
            assert R.shape == (3, 3)
            assert t.shape == (3, 1)
            P = K.astype(np.float64) @ np.hstack([R.astype(np.float64), t.astype(np.float64)])

        self.P = P.astype(np.float64)  # projection matrix
        self.K = K.astype(np.float64)  # intrinsic matrix

        if lite:
            return

        self.dist_coeff = dist_coeff.astype(np.float64) if dist_coeff is not None else None # radial distortion and tangential distortion
        self.res_w = res_w
        self.res_h = res_h
        self.azimuth = azimuth
        self.undistort = undistort

        self.Rw2c = R.astype(np.float64)  # rotation matrix from world to cam
        self.Tw2c = t.astype(np.float64)  # translation vector, the position of the origin of the world coordinate
                                          # system expressed in coordinates of the camera-centered coordinate system
        self.Rc2w = self.Rw2c.T
        self.Tc2w = -self.Rw2c.T @ self.Tw2c

        self.cam_ray_world = self.get_cam_ray_world()
        self.cam_pitch_rad = self.get_cam_pitch_rad()
        self.cam_pitch_deg = self.get_cam_pitch_angle()
        self.cam_orig_world = self.get_cam_coord_world()

        self.Rc2n, self.Tc2n = self.get_norm_coord_config()

        self.Rn2c = self.Rc2n.T
        self.Tn2c = -self.Rc2n.T @ self.Tc2n

        self.Rw2n = self.Rc2n @ self.Rw2c
        self.Tw2n = self.Rc2n @ self.Tw2c + self.Tc2n

        self.Rn2w = self.Rc2w @ self.Rn2c
        self.Tn2w = -self.Rn2w @ self.Tc2n - self.Rc2w @ self.Tw2c

        self.cam_ray_norm = self.get_cam_ray_normalized()

        principal_point_x = self.K[0, 2]
        principal_point_y = self.K[1, 2]

        if self.undistort:
            self.pp_cam = self.undistort_point(
                np.asarray([principal_point_x, principal_point_y], dtype=np.float64).reshape((1, 1, 2))
            ).reshape(-1, 2)
        else:
            self.pp_cam = np.array([principal_point_x, principal_point_y], dtype=np.float64).reshape((-1, 2))

    def get_cam_coord_world(self):
        """
        return world coordinate of camera origin
        # https://en.wikipedia.org/wiki/Camera_resectioning
        :return:
        """
        return -self.Rw2c.T @ self.Tw2c

    def get_cam_ray_world(self):
        """
        return the ray of camera in world coordinate system
        # define the vector that starts from camera center to principal(focal) point as representation of the camera.
        # suppose that the focal point is normalized,
        # we convert the vector to world space to represent the ray of the camera.
        :return:
        """
        focal_pt_cam = np.asarray([0, 0, 1], np.float64)
        P_w = self.Rc2w @ focal_pt_cam
        return P_w[0:3].reshape((3, 1))

    def get_cam_ray_normalized(self):
        """
        return the ray of camera in normalized coord system
        :return:
        """
        focal_pt_cam = np.asarray([0, 0, 1], np.float64)
        P_n = self.Rc2n @ focal_pt_cam
        return P_n[0:3].reshape((3, 1))

    def get_cam_pitch_rad(self):
        """
        return camera pitch in radius
        # here we assume the camera is looking towards to the ground
        :return:
        """
        ray_upright = np.zeros((3, 1)).astype(np.float64)
        ray_upright[2] = 1
        return angle(self.cam_ray_world, ray_upright) - np.pi / 2

    def get_cam_pitch_angle(self):
        """
        return camera pitch in degree
        :return:
        """
        return self.get_cam_pitch_rad() * 180.0 / np.pi

    def get_norm_coord_config(self):
        """
        rotate the camera about the x-axis to eliminate the pitch.
        in normalized world coordinate, we set the translation as the height of camera,
        which is the position of the origin of the normalized coordinate system
        expressed in coordinates of the camera-centered coordinate system.
        :return:
        """
        Rc2n = np.eye(3, dtype=np.float64)
        Rc2n[1, 1] = math.cos(self.cam_pitch_rad)
        Rc2n[1, 2] = math.sin(self.cam_pitch_rad)
        Rc2n[2, 1] = -math.sin(self.cam_pitch_rad)
        Rc2n[2, 2] = math.cos(self.cam_pitch_rad)
        Rc2n = Rc2n.astype(np.float64)

        Tc2n = np.zeros((3, 1)).astype(np.float64)
        err_str = 'camera height should be larger than 0 if the world coordinate system is set up on the ground'
        assert self.cam_orig_world[2] > 0, err_str
        Tc2n[1] = -self.cam_orig_world[2]

        return Rc2n, Tc2n

    def camera2world(self, pt):
        """
        pt is the 3d coord in camera_coordinate system
        :param pt:
        :return:
        """
        if isinstance(pt, np.ndarray):
            return pt @ self.Rc2w.T + self.Tc2w.T
        else:
            return pt @ torch.from_numpy(self.Rc2w.T).float().to(pt.device) + torch.from_numpy(self.Tc2w.T).float().to(pt.device)

    def world2camera(self, pt):
        """

        :return:
        """
        if isinstance(pt, np.ndarray):
            return pt @ self.Rw2c.T + self.Tw2c.T
        else:
            return pt @ torch.from_numpy(self.Rw2c.T).float().to(pt.device) + torch.from_numpy(self.Tw2c.T).float().to(pt.device)

    def camera2normalized(self, pt):
        """
        pt is the 3d coord in camera_coordinate system
        :param pt:
        :return:
        """
        if isinstance(pt, np.ndarray):
            return pt @ self.Rc2n.T + self.Tc2n.T
        else:
            return pt @ torch.from_numpy(self.Rc2n.T).float().to(pt.device) + torch.from_numpy(self.Tc2n.T).float().to(pt.device)

    def normalized2camera(self, pt):
        """
        pt is the 3d coord in normalized system
        :param pt:
        :return:
        """
        if isinstance(pt, np.ndarray):
            return pt @ self.Rn2c.T + self.Tn2c.T
        else:
            return pt @ torch.from_numpy(self.Rn2c.T).float().to(pt.device) + torch.from_numpy(self.Tn2c.T).float().to(pt.device)

    def world2normalized(self, pt):
        """
        pt is the 3d coord in world coordinate system
        :param pt:
        :return:
        """
        if isinstance(pt, np.ndarray):
            return pt @ self.Rw2n.T + self.Tw2n.T
        else:
            return pt @ torch.from_numpy(self.Rw2n.T).float().to(pt.device) + torch.from_numpy(self.Tw2n.T).float().to(pt.device)

    def normalized2world(self, pt):
        """
        pt is the 3d coord in normalized coordinate system
        :param pt:
        :return:
        """
        if isinstance(pt, np.ndarray):
            return pt @ self.Rn2w.T + self.Tn2w.T
        else:
            return pt @ torch.from_numpy(self.Rn2w.T).float().to(pt.device) + torch.from_numpy(self.Tn2w.T).float().to(pt.device)

    def undistort_point(self, points2d):
        """

        :param points2d:
        :return:
        """
        batch_size, num_kpt, feat_dim = points2d.shape
        points2d = np.reshape(points2d, (-1, 1, feat_dim))
        points2d = cv2.undistortPoints(points2d, self.K, self.dist_coeff, P=self.K)
        return np.reshape(points2d, (batch_size, num_kpt, feat_dim))

    def encode_uv_with_intrinsic(self, uv):
        """

        :param uv: shape of (2, )
        :return: shape of (2, )
        """
        batch_size, num_kpt, feat_dim = uv.shape
        fx = self.K[0, 0]
        fy = self.K[1, 1]

        pt = np.zeros([batch_size, num_kpt, feat_dim], dtype=np.float64)

        if self.undistort:
            uv = self.undistort_point(uv)

        pt[..., 0] = (uv[..., 0] - self.pp_cam[..., 0]) / fx
        pt[..., 1] = (uv[..., 1] - self.pp_cam[..., 1]) / fy

        return pt

    def decouple_uv_with_intrinsic(self, uv):
        """

        :param uv: shape of (2, )
        :return: shape of (2, )
        """
        batch_size, num_kpt, feat_dim = uv.shape
        fx = self.K[0, 0]
        fy = self.K[1, 1]

        pt = np.zeros([batch_size, num_kpt, feat_dim], dtype=np.float64)

        pt[..., 0] = uv[..., 0] * fx + self.pp_cam[..., 0]
        pt[..., 1] = uv[..., 1] * fy + self.pp_cam[..., 1]

        return pt

    def get_cam_ray_given_uv(self, uv):
        """

        :param uv: shape of (2, )
        :return: shape of (3, )
        """
        batch_size, num_kpt, feat_dim = uv.shape
        pt_cam = np.ones([batch_size, num_kpt, feat_dim + 1], dtype=np.float64)

        pt_cam[..., :2] = self.encode_uv_with_intrinsic(uv)

        return pt_cam @ self.Rc2n.T

    def get_uv_given_cam_ray(self, pt):
        """

        :param pt: shape of (3, )
        :return: shape of (2, )
        """
        pt_cam = pt @ self.Rn2c.T

        uv_with_intrinsic = pt_cam[..., :2]

        return self.decouple_uv_with_intrinsic(uv_with_intrinsic)

    def project(self, X):
        """
        Project 3D homogenous points X (4 * n) and normalize coordinates.
        Return projected 2D points (2 x n coordinates)
        :param X:
        :return:
        """
        if isinstance(X, np.ndarray):
            x = X @ self.P.T
            x[..., 0] = x[..., 0] / x[..., 2]
            x[..., 1] = x[..., 1] / x[..., 2]
            return x[..., :2]
        else:
            x = X @ torch.from_numpy(self.P.T).float().to(X.device)
            org_dim = x.shape
            new_dim = [item for item in org_dim[:-1]] + [org_dim[-1]-1]
            ret = torch.zeros(*new_dim).to(x.device)
            ret[..., 0] = x[..., 0] / x[..., 2]
            ret[..., 1] = x[..., 1] / x[..., 2]
            return ret