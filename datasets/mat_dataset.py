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

class MatDataset(Dataset):
    """
    Base Dataset Class - Handles data loading and augmentation.
    Able to handle heterogeneous datasets (different annotations available for different datasets).
    You need to update the path to each dataset in utils/config.py.
    """

    def __init__(self, options, dataset, ignore_3d=False, use_augmentation=True, is_train=True):
        super(MatDataset, self).__init__()
        self.dataset = dataset
        self.is_train = is_train
        self.options = options
        self.img_dir = config.DATASET_FOLDERS[dataset]
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
        self.data = np.load(config.DATASET_FILES[is_train][dataset])
        self.imgname = self.data['imgname']
        self.imgf = self.data['imgf']
        self.garf = self.data['garf']
        # print(self.imgf[0]-self.imgf[1])
        # print(self.garf[0]-self.garf[1])
        # import sys;sys.exit(0)

        # Bounding boxes are assumed to be in the center and scale format
        self.scale = self.data['scale']
        self.center = self.data['center']
        
        # If False, do not do augmentation
        self.use_augmentation = use_augmentation
        self.materials = self.data['matname']
        
        self.length = self.scale.shape[0]

    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0            # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0            # rotation
        sc = 1            # scaling
        if False and self.is_train:
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
        
        return flip, pn, rot/2, sc

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn):
        """Process rgb image and do augmentation."""
        rgb_img = crop(rgb_img, center, scale, 
                      (constants.IMG_RES, constants.IMG_RES), rot=rot)
        # flip the image 
        if flip:
            rgb_img = flip_img(rgb_img)
        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
        rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
        rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
        # (3,224,224),float,[0,1]
        rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
        return rgb_img

    def __getitem__(self, index):
        # index = 0
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()

        # Get augmentation parameters
        flip, pn, rot, sc = self.augm_params()
        
        # Load image
        imgs = []
        garf, imgf = [],[]
        length = self.options.seq_len
        # feature_dat = np.load(join(self.img_dir, self.imgname[index][0][:-8], 'features.npz'))
        # for i in range(length):
            # imgname = join(self.img_dir, self.imgname[index][i])
            # try:
            #     img = cv2.imread(imgname)[:,:,::-1].copy().astype(np.float32)
            # except TypeError:
            #     print(imgname)
            #     import sys;sys.exit(0)
            # orig_shape = np.array(img.shape)[:2]

            # Process image
            # img = self.rgb_processing(img, center, sc*scale, rot, flip, pn)
            # img = torch.from_numpy(img).float()
            # imgs.append(self.normalize_img(img))
            # frameid = int(self.imgname[index][i][-8:-4])
            # garf.append(torch.from_numpy(feature_dat['garf'][frameid]).float())
            # imgf.append(torch.from_numpy(feature_dat['imgf'][frameid]).float())
        # Store image before normalization to use it in visualization
        # item['img'] = torch.stack(imgs, dim=0) # 25*3*224*224
        item['garf'] = torch.from_numpy(self.garf[index][:length].copy()) #*np.random.uniform(1,1,self.garf[index].shape)).float() #torch.stack(garf, dim=0)
        item['imgf'] = torch.from_numpy(self.imgf[index][:length].copy()) #*np.random.uniform(0,10,self.imgf[index].shape)).float() #torch.stack(imgf, dim=0)
        matid = self.materials[index]
        item['stretch_mat'] = matid // 9
        item['bend_mat'] = matid % 9
        densities = [0.324,0.284,0.187,0.276,0.224,0.228,0.220,0.113,0.128,0.204]
        item['density'] = densities[0]

        item['scale'] = float(sc * scale)
        item['center'] = center.astype(np.float32)
        # item['orig_shape'] = orig_shape
        item['is_flipped'] = flip
        item['rot_angle'] = np.float32(rot)
        item['sample_index'] = index
        item['dataset_name'] = self.dataset
        item['dataname'] = self.imgname[index][0]

        return item

    def __len__(self):
        return len(self.imgname)
