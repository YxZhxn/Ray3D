

import numpy as np


class Skeleton:
    def __init__(self, parents, joints_left, joints_right):
        assert len(joints_left) == len(joints_right)

        self._parents = np.array(parents)
        self._joints_left = joints_left
        self._joints_right = joints_right
        self._compute_metadata()

    def num_joints(self):
        return len(self._parents)

    def parents(self):
        return self._parents

    def has_children(self):
        return self._has_children

    def children(self):
        return self._children

    def remove_joints(self, joints_to_remove):
        """
        Remove the joints specified in 'joints_to_remove'.
        """
        valid_joints = []
        for joint in range(len(self._parents)):
            if joint not in joints_to_remove:
                valid_joints.append(joint)

        for i in range(len(self._parents)):
            while self._parents[i] in joints_to_remove:
                self._parents[i] = self._parents[self._parents[i]]

        index_offsets = np.zeros(len(self._parents), dtype=int)
        new_parents = []
        for i, parent in enumerate(self._parents):
            if i not in joints_to_remove:
                new_parents.append(parent - index_offsets[parent])
            else:
                index_offsets[i:] += 1
        self._parents = np.array(new_parents)

        if self._joints_left is not None:
            new_joints_left = []
            for joint in self._joints_left:
                if joint in valid_joints:
                    new_joints_left.append(joint - index_offsets[joint])
            self._joints_left = new_joints_left
        if self._joints_right is not None:
            new_joints_right = []
            for joint in self._joints_right:
                if joint in valid_joints:
                    new_joints_right.append(joint - index_offsets[joint])
            self._joints_right = new_joints_right

        self._compute_metadata()

        # valid_joints = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
        return valid_joints

    def joints_left(self):
        return self._joints_left

    def joints_right(self):
        return self._joints_right

    def _compute_metadata(self):
        self._has_children = np.zeros(len(self._parents)).astype(bool)
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._has_children[parent] = True

        self._children = []
        for i, parent in enumerate(self._parents):
            self._children.append([])
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._children[parent].append(i)


class Human36mSkeleton(Skeleton):
    # HUMAN_36M_KEYPOINTS = [
    #     'mid_hip',           # 0
    #     'right_hip',         # 1
    #     'right_knee',        # 2
    #     'right_ankle',       # 3
    #     'right_foot_base',   # 4
    #     'right_foot_tip',    # 5
    #     'left_hip',          # 6
    #     'left_knee',         # 7
    #     'left_ankle',        # 8
    #     'left_foot_base',    # 9
    #     'left_foot_tip',     #10
    #     'mid_hip_2',         #11
    #     'mid_spine',         #12
    #     'neck',              #13
    #     'chin',              #14
    #     'head',              #15
    #     'neck_2',            #16
    #     'left_shoulder',     #17
    #     'left_elbow',        #18
    #     'left_wrist',        #19
    #     'left_wrist_2',      #20
    #     'left_palm',         #21
    #     'left_thumb',        #22
    #     'left_thumb_2',      #23
    #     'neck_3',            #24
    #     'right_shoulder',    #25
    #     'right_elbow',       #26
    #     'right_wrist',       #27
    #     'right_wrist_2',     #28
    #     'right_palm',        #29
    #     'right_thumb',       #30
    #     'right_thumb_3'      #31
    # ]
    def __init__(self, parents, joints_left, joints_right):
        super().__init__(parents, joints_left, joints_right)

        self.kpt_name = ['mid_hip',
                          'right_hip', 'right_knee', 'right_ankle',
                          'left_hip', 'left_knee', 'left_ankle',
                          'mid_spine', 'neck', 'chin', 'head',
                          'left_shoulder', 'left_elbow', 'left_wrist',
                          'right_shoulder', 'right_elbow', 'right_wrist',
                          ]
        self.kpt_idx = [0, 1, 2, 3, 4, 5, 6, 7,  8,  9,  10, 11, 12, 13, 14, 15, 16]