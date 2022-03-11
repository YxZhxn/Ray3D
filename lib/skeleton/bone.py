import torch


def get_3d_pose_from_bone_vector(bone_vect, root_origin):
    """
    NOTE: one must add the root origin to recover 3d pose
    :param bone_vect:
    :return:
    """

    # 16 x 17
    convet_mat_inv = torch.Tensor([
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # 0 basement
        [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # 0 1
        [-1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # 1 2
        [-1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # 2 3
        [ 0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # 0 4
        [ 0,  0,  0, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # 4 5
        [ 0,  0,  0, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # 5 6
        [ 0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # 0 7
        [ 0,  0,  0,  0,  0,  0, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0],  # 7 8
        [ 0,  0,  0,  0,  0,  0, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0],  # 8 9
        [ 0,  0,  0,  0,  0,  0, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0],  # 9 10
        [ 0,  0,  0,  0,  0,  0, -1, -1,  0,  0, -1,  0,  0,  0,  0,  0],  # 8 11
        [ 0,  0,  0,  0,  0,  0, -1, -1,  0,  0, -1, -1,  0,  0,  0,  0],  # 11 12
        [ 0,  0,  0,  0,  0,  0, -1, -1,  0,  0, -1, -1, -1,  0,  0,  0],  # 12 13
        [ 0,  0,  0,  0,  0,  0, -1, -1,  0,  0,  0,  0,  0, -1,  0,  0],  # 8 14
        [ 0,  0,  0,  0,  0,  0, -1, -1,  0,  0,  0,  0,  0, -1, -1,  0],  # 14 15
        [ 0,  0,  0,  0,  0,  0, -1, -1,  0,  0,  0,  0,  0, -1, -1, -1],  # 15 16
    ]).transpose(1, 0)

    batch_size, num_frames, num_bones, num_feat = bone_vect.shape
    assert num_bones == convet_mat_inv.shape[0] and num_feat == 3

    convet_mat_inv = convet_mat_inv.to(bone_vect.device)
    convet_mat_inv = convet_mat_inv.repeat([batch_size, num_frames, 1, 1]).view(-1, num_bones, num_bones + 1)
    bone_vect_T = bone_vect.view(-1, num_bones, num_feat).contiguous().permute(0, 2, 1).contiguous()
    pose3d = torch.matmul(bone_vect_T, convet_mat_inv)
    pose3d = pose3d.permute(0, 2, 1).contiguous().view(batch_size, num_frames, num_bones + 1, 3).contiguous() + root_origin
    return pose3d


def get_bone_vector_from_3d_pose(pose_3d):
    """

    :param pose_3d:
    :return:
    """

    # 17 x 16
    convet_mat = torch.Tensor([
        [1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # 0 1
        [0,  1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # 1 2
        [0,  0,  1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # 2 3
        [1,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # 0 4
        [0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # 4 5
        [0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # 5 6
        [1,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # 0 7
        [0,  0,  0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  0,  0,  0,  0],  # 7 8
        [0,  0,  0,  0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  0,  0,  0],  # 8 9
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  0,  0],  # 9 10
        [0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0, -1,  0,  0,  0,  0,  0],  # 8 11
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0],  # 11 12
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -1,  0,  0,  0],  # 12 13
        [0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0, -1,  0,  0],  # 8 14
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  -1, 0],  # 14 15
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -1],  # 15 16
    ]).transpose(1, 0)

    batch_size, num_frames, num_joints, num_feat = pose_3d.shape
    assert num_joints == convet_mat.shape[0] and num_feat == 3

    convet_mat = convet_mat.to(pose_3d.device).repeat([batch_size, num_frames, 1, 1]).view(-1, num_joints, num_joints - 1)
    pose_3d = pose_3d.view(-1, num_joints, 3).contiguous().permute(0, 2, 1).contiguous()
    bone_vect = torch.matmul(pose_3d, convet_mat)
    bone_vect = bone_vect.permute(0, 2, 1).contiguous().view(batch_size, num_frames, num_joints-1, 3).contiguous()
    return bone_vect


def get_bone_length_from_3d_pose(pose_3d):
    """

    :param pose_3d:
    :return:
    """
    bone_vect = get_bone_vector_from_3d_pose(pose_3d)
    bone_length = torch.norm(bone_vect, dim=-1, keepdim=True)
    return bone_length


def get_bone_unit_vector_from_3d_pose(pose_3d):
    """

    :param pose_3d:
    :return:
    """
    bone_vect = get_bone_vector_from_3d_pose(pose_3d)
    bone_length = get_bone_length_from_3d_pose(pose_3d)
    bone_unit_vect = bone_vect / bone_length
    return bone_unit_vect