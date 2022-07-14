from __future__ import division

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
import numpy as np
import cv2
from os.path import join

import config
import constants
from utils.imutils import crop, flip_img, flip_pose, flip_kp, transform, rot_aa
import glob
import pytorch3d
from pytorch3d import ops, io, transforms

class AtlasDataset(Dataset):
    """
    Base Dataset Class - Handles data loading and augmentation.
    Able to handle heterogeneous datasets (different annotations available for different datasets).
    You need to update the path to each dataset in utils/config.py.
    """

    def __init__(self, options, dataset, ignore_3d=False, use_augmentation=True, is_train=True):
        super(AtlasDataset, self).__init__()
        self.dataset = dataset
        self.is_train = is_train
        self.options = options
        self.img_dir = config.DATASET_FOLDERS[dataset]
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
        self.data = np.load(config.DATASET_FILES[is_train][dataset])
        self.imgname = self.data['imgname']
        self.mapname = self.data['mapname']
        self.simname = self.data['simname']
        
        # Get paths to gt masks, if available
        try:
            self.maskname = self.data['maskname']
        except KeyError:
            pass
        try:
            self.partname = self.data['partname']
        except KeyError:
            pass

        # Bounding boxes are assumed to be in the center and scale format
        self.scale = self.data['scale']
        self.center = self.data['center']
        
        # If False, do not do augmentation
        self.use_augmentation = use_augmentation
        
        # Get gt SMPL parameters, if available
        self.pose = self.data['pose'].astype(np.float)
        self.betas = self.data['shape'].astype(np.float)
        self.trans = self.data['trans'].astype(np.float)
        if 'has_smpl' in self.data:
            self.has_smpl = self.data['has_smpl']
        else:
            self.has_smpl = np.ones(len(self.imgname))
        if ignore_3d:
            self.has_smpl = np.zeros(len(self.imgname))
        
        # Get gt 3D pose, if available
        try:
            self.pose_3d = self.data['S']
            self.has_pose_3d = 1
        except KeyError:
            self.has_pose_3d = 0
        if ignore_3d:
            self.has_pose_3d = 0
        
        # Get 2D keypoints
        try:
            keypoints_gt = self.data['part']
        except KeyError:
            keypoints_gt = np.zeros((len(self.imgname), 24, 3))
        try:
            keypoints_openpose = self.data['openpose']
        except KeyError:
            keypoints_openpose = np.zeros((len(self.imgname), 25, 3))
        self.keypoints = np.concatenate([keypoints_openpose, keypoints_gt], axis=1)

        # Get gender data, if available
        try:
            gender = self.data['gender']
            self.gender = np.array([0 if str(g) == 'm' else 1 for g in gender]).astype(np.int32)
        except KeyError:
            self.gender = -1*np.ones(len(self.imgname)).astype(np.int32)
        
        self.length = self.scale.shape[0]

    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0            # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0            # rotation
        sc = 1            # scaling
        if self.is_train:
            # We flip with probability 1/2
            if np.random.uniform() <= 0.5:
                flip = 1
            
            # Each channel is multiplied with a number 
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            pn = np.random.uniform(1-self.options.noise_factor, 1+self.options.noise_factor, 3)
            
            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            rot = min(2*self.options.rot_factor,
                    max(-2*self.options.rot_factor, np.random.randn()*self.options.rot_factor))
            
            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(1+self.options.scale_factor,
                    max(1-self.options.scale_factor, np.random.randn()*self.options.scale_factor+1))
            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0
        
        return flip, pn, 0, sc
    
    def load_points(self, path):
        lists = glob.glob(path)
        mesh = io.load_objs_as_meshes(lists)
        pts, nms = ops.sample_points_from_meshes(mesh, self.options.num_points // len(lists), return_normals=True)
        pts = pts.reshape([-1,3])
        return pts, nms.reshape([-1,3]) * 0.1
    
    def map_processing(self, map_img, center, scale, rot, flip, pn):
        """data aug for pytorch3d"""
        tr = pytorch3d.transforms.RotateAxisAngle(rot, axis='Z')
        pt = tr.transform_points(map_img)
        if flip:
            pt[:,0] = -pt[:,0]
        return pt

    def map_processing_legacy(self, map_img, center, scale, rot, flip, pn):
        """Process rgb image and do augmentation."""
        map_img = np.roll(map_img, 64, axis=1)
        rot_mat = np.eye(3)
        if not rot == 0:
            rot_rad = rot * np.pi / 180
            sn,cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0,:2] = [cs, -sn]
            rot_mat[1,:2] = [sn, cs]
        map_img = np.reshape(map_img, [256*256,4])
        map_img[:,:3] = np.einsum('ij,kj->ki', rot_mat, map_img[:,:3]) 
        map_img = np.reshape(map_img, [256, 256, 4])
        # flip the image 
        if flip:
            map_img = map_img[:,::-1,:]
            map_img[:,:,0] = -map_img[:,:,0]
        # (3,224,224),float,[0,1]
        map_img = np.transpose(map_img.astype('float32'),(2,0,1))
        return map_img

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def __getitem__(self, index):
        # index = 0
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()

        # Get augmentation parameters
        flip, pn, rot, sc = self.augm_params()
        #print(rot)
        # Get SMPL parameters, if available
        if self.has_smpl[index]:
            pose = self.pose[index].copy()
            betas = self.betas[index].copy()
            trans = torch.from_numpy(self.trans[index].copy()).float()
        else:
            pose = np.zeros(72)
            betas = np.zeros(10)

        # Store image before normalization to use it in visualization
        clothmap, normmap = self.load_points(join(self.img_dir, self.simname[index]))
        clothmap = self.map_processing(clothmap - trans, center, sc*scale, rot, flip, pn)

        normmap = self.map_processing(normmap, center, sc*scale, rot, flip, pn)
        item['map'] = clothmap.float()
        item['normmap'] = normmap.float()
        item['pose'] = torch.from_numpy(self.pose_processing(pose, rot, flip)).float()
        item['betas'] = torch.from_numpy(betas).float()

        item['rot_angle'] = np.float32(rot)
        item['gender'] = self.gender[index]
        item['sample_index'] = index
        item['dataset_name'] = self.dataset
        item['dataname'] = self.imgname[index]

        return item

    def __len__(self):
        return len(self.imgname)
