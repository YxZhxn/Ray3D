


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

        principal_point_x = self.K[0, 2]
        principal_point_y = self.K[1, 2]

        if self.undistort:
            self.pp_cam = self.undistort_point(
                np.asarray([principal_point_x, principal_point_y], dtype=np.float64).reshape((1, 1, 2))
            ).reshape(-1, 2)
        else:
            self.pp_cam = np.array([principal_point_x, principal_point_y], dtype=np.float64).reshape((-1, 2))

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

    def undistort_point(self, points2d):
        """

        :param points2d:
        :return:
        """
        batch_size, num_kpt, feat_dim = points2d.shape
        points2d = np.reshape(points2d, (-1, 1, feat_dim))
        points2d = cv2.undistortPoints(points2d, self.K, self.dist_coeff, P=self.K)
        return np.reshape(points2d, (batch_size, num_kpt, feat_dim))

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
