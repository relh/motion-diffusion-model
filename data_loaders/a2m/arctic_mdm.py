#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
from einops import rearrange
from PIL import Image
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix, matrix_to_axis_angle
from torchvision import transforms


if __name__ == "__main__":
    # Running as a script
    from data_loaders.a2m.dataset import Dataset
else:
    # Running as an imported module
    from .dataset import Dataset

#from torch.utils.data import Dataset

class ARCTIC_MDM(Dataset):
    dataname = "arctic"

    def __init__(self, datapath="/home/relh/Code/hand_trajectories/cache/", split="train", **kargs):
        self.datapath = datapath

        super().__init__(**kargs)

        tensordatafilepath = os.path.join(datapath, "arctic_mdm.pt")
        data = torch.load(tensordatafilepath) 

        self._pose = [x for x in data["_pose"]]
        self._num_frames_in_video = [x for x in data["_num_frames_in_video"]]
        self._joints = [x for x in data["_joints"]]

        self._train = list(range(len(self._pose)))

    def _load_joints3D(self, ind, frame_ix):
        return self._joints[ind][frame_ix]

    def _load_rotvec(self, ind, frame_ix):
        pose = self._pose[ind][frame_ix]
        pose = pose.reshape(pose.shape[0], -1, 3)
        return pose


class ARCTIC(Dataset):
    def __init__(self, args=None, split='train', num_frames=100):
        if args is None:
            sys.path.append('/home/relh/Code/hand_trajectories/')
            from main import build_args
            args = build_args()

        # TODO use num_frames
        self.args = args
        self.split = split

        sys.path.append('/home/relh/Code/hand_trajectories/third_party/arctic/')
        from src.callbacks.process.process_flying import process_data
        from src.datasets.seq_arctic_dataset import SeqArcticDataset
        from src.parsers.parser import construct_args

        arctic_args = construct_args(img_res=args.img_res, fast_dev_run=args.debug)
        self.dataset = SeqArcticDataset(args=arctic_args, split=self.split)
        self.process_fn = process_data

        # data points are either left or right hand at origin
        # augment is add a small amount of translation noise to everything
        os.makedirs('/home/relh/Code/hand_trajectories/cache/processed_input/', exist_ok=True) 
        os.makedirs('/home/relh/Code/hand_trajectories/cache/processed_batch/', exist_ok=True) 


    def __getitem__(self, index):
        cache_name = (f'/home/relh/Code/hand_trajectories/cache/processed_input/{index}.pt')
        dominant_hand = 'right' if index % 2 == 0 else 'left'

        if os.path.exists(cache_name):
            returner = torch.load(cache_name) 
        else:
            # trajectory details
            data, misc = self.dataset[index // 2]

            right = process_hand_data(data, 'right')
            left = process_hand_data(data, 'left')
            obj = {'j_pos': torch.as_tensor(data['world_coord']['kp3d'][:-1])}

            root = (right if dominant_hand == 'right' else left)['g_trans'][0].clone()
            right['g_trans'] -= root
            right['j_pos']   -= root
            left['g_trans']  -= root
            left['j_pos']    -= root
            obj['j_pos']     -= root

            # conditioning details
            cond_signal = build_conditioning_signal(data, hand=dominant_hand)

            returner = {'obj': obj, 'right': right, 'left': left, 'cond': cond_signal}
            torch.save(returner, cache_name)

        # need to match MDM format 
        return returner

    def __len__(self):
        return len(self.dataset) * 2 


def build_conditioning_signal(data, hand='right'):
    # get object point cloud (Po)
    Po = torch.tensor(data['world_coord']['verts.object'][0], dtype=torch.float32)  # First frame

    # get rigid object pose relative to the hand (Ro)
    obj_rot = torch.tensor(data['params']['obj_rot'][0], dtype=torch.float32)  # First frame
    obj_trans = torch.tensor(data['params']['obj_trans'][0], dtype=torch.float32)  # First frame
    hand_rot = torch.tensor(data['params']['rot_' + hand[0]][0], dtype=torch.float32)  # First frame
    hand_trans = torch.tensor(data['params']['trans_' + hand[0]][0], dtype=torch.float32)  # First frame
    Ro = compute_relative_pose(obj_rot, obj_trans, hand_rot, hand_trans)

    # get MANO shape parameters (b)
    b = torch.tensor(data['params']['shape_' + hand[0]], dtype=torch.float32)

    # get starting pose of the hand (X0)
    X0 = data['params']['pose_' + hand[0]][0]  # First pose of the sequence

    return {'Po': Po, 'Ro': Ro, 'b': b, 'X0': X0}


def compute_relative_pose(obj_rot, obj_trans, hand_rot, hand_trans):
    # convert axis-angle to rotation matrix
    obj_rot_matrix = axis_angle_to_matrix(obj_rot.view(1, 3)).view(3, 3)
    hand_rot_matrix = axis_angle_to_matrix(hand_rot.view(1, 3)).view(3, 3)

    # construct 4x4 transformation matrices
    obj_pose = torch.eye(4)
    obj_pose[:3, :3] = obj_rot_matrix
    obj_pose[:3, 3] = obj_trans

    hand_pose = torch.eye(4)
    hand_pose[:3, :3] = hand_rot_matrix
    hand_pose[:3, 3] = hand_trans

    # compute the inverse of the hand pose
    hand_pose_inv = torch.inverse(hand_pose)

    # compute the relative pose
    relative_pose = torch.matmul(hand_pose_inv, obj_pose)
    return relative_pose


def process_hand_data(data, hand='right', time_step=1):
    # get joint positions
    joint_positions = torch.tensor(data['world_coord']['joints.' + hand], dtype=torch.float32)

    # get MANO poses and concat
    wrist_poses = torch.tensor(data['params']['rot_' + hand[0]], dtype=torch.float32).view(-1, 1, 3)
    mano_poses = torch.tensor(data['params']['pose_' + hand[0]], dtype=torch.float32).view(-1, 15, 3)
    poses = torch.cat([wrist_poses, mano_poses], dim=1)

    # transform poses from aa to 6d
    rotation_matrices = axis_angle_to_matrix(poses.view(-1, 3)).view(-1, 16, 3, 3)
    rotation_6d = matrix_to_rotation_6d(rotation_matrices.view(-1, 3, 3)).view(-1, 16, 6)

    # pad 16 joints to 21 
    full_rotation_6d = torch.cat([rotation_6d, torch.zeros(rotation_6d.shape[0], 5, 6)], dim=1)

    # calc joint velocities and angular velocities
    joint_velocities = joint_positions[1:] - joint_positions[:-1] 
    angular_velocities = calculate_angular_velocity(poses, time_step)

    # get global translation and velocity
    global_translation = torch.tensor(data['params']['trans_' + hand[0]], dtype=torch.float32)
    global_velocity = global_translation[1:] - global_translation[:-1]

    return {
        'j_pos': joint_positions[:-1],
        'j_rot': full_rotation_6d[:-1],
        'j_vel': joint_velocities,
        'j_ang_vel': angular_velocities,
        'g_trans': global_translation[:-1],
        'g_vel': global_velocity
    }


def calculate_angular_velocity(poses, time_step=1):
    """
    Calculate angular velocities from axis-angle representations.
    
    Args:
    poses (Tensor): A tensor of shape (frames, joints, 3) representing axis-angle rotations.
    time_step (float): The time interval between frames.

    Returns:
    Tensor: Angular velocities for each joint.
    """
    num_frames = poses.size(0)
    num_joints = poses.size(1)

    # Initialize angular velocity tensor
    angular_velocities = torch.zeros(num_frames - 1, num_joints, 3)

    for i in range(1, num_frames):
        # Calculate the change in pose (axis-angle) for each joint
        delta_pose = poses[i] - poses[i - 1]

        # Angular velocity: change in angle (in axis-angle representation) / time_step
        angular_velocity = delta_pose / time_step

        # Store the result
        angular_velocities[i - 1] = angular_velocity

    return angular_velocities


def build_mdm():
    train_ds = ARCTIC_MDM(args=None, split='train')

    a_joints = []
    a_pose = []
    a_num_frames_in_video = []
    for x in range(0, len(train_ds), 2): 
        print(x)
        returner = train_ds[x]
        print(returner['obj']['j_pos'].shape)
        _joints = torch.cat([returner['right']['j_pos'],\
                             returner['left']['j_pos'],\
                             returner['obj']['j_pos']], dim=1)

        _pose = matrix_to_axis_angle(rotation_6d_to_matrix(torch.cat([returner['right']['j_rot'], returner['left']['j_rot']], dim=1)))
        _pose = torch.cat((_pose, torch.zeros(_pose.shape[0], 32, 3)), dim=1)
        _pose = _pose.view(_pose.shape[0], -1) 

        _num_frames_in_video = _pose.shape[0]

        a_joints.append(_joints)
        a_pose.append(_pose)
        a_num_frames_in_video.append(_num_frames_in_video)

    data = {'_joints': a_joints, '_pose': a_pose, '_num_frames_in_video': a_num_frames_in_video}
    torch.save(data, '/home/relh/Code/hand_trajectories/cache/arctic_mdm.pt')


if __name__ == "__main__":
    train_ds = ARCTIC_MDM(args=None, split='train')
