import os
import sys
import cv2
import glob
import h5py
import numpy as np
import argparse
import pickle as pkl
import torch
from torch import nn
from models import SMPL, hmr
from utils.imutils import crop
import constants
import config
from torchvision.transforms import Normalize
import random
# from utils.geometry import batch_rodrigues, batch_rodrigues_back

siz=800.0
cam_K = np.array([
    [siz*50/36,0,400],
    [0,siz*50/36,400],
    [0,0,1]])
cam_RT=np.array([
    [1,0,0,0],
    [0,-0.9996,0.0279,-0.2390],
    [0,-0.0279,-0.9996,3.1846]])
cam_P=np.array([
    [1111.1111,-11.1687,-399.8441,1273.8273],
    [0,-1121.8466,-368.82,1008.2589],
    [0,-0.0279,-0.9996,3.1846]])

is_train=lambda x:x%10!=0

def rgb_processing(rgb_img, center, scale):
    """Process rgb image and do augmentation."""
    rgb_img = crop(rgb_img, center, scale, 
                  (224, 224))
    # in the rgb image we add pixel noise in a channel-wise manner
    rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]))
    rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]))
    rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]))
    # (3,224,224),float,[0,1]
    rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
    return rgb_img

def synthetic_M_extract(dataset_path, out_path, isTrain=True):

    smpl = SMPL('data/smpl/',
                batch_size=250,
                create_transl=False)
    coder = hmr(config.SMPL_MEAN_PARAMS, pretrained=True, smpl=smpl)
    coder = nn.DataParallel(coder).cuda()
    coder.eval()
    checkpoint_file = 'logs_combined/0430_2051/checkpoints/2020_05_02-01_00_45.pt'
    state_dict = torch.load(checkpoint_file)['model']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.smpl'):
            continue
        new_state_dict[k] = v
    # load params
    coder.load_state_dict(new_state_dict, strict=False)
    print('Decoder loaded')

    # scale factor
    scaleFactor = 1.2
    blender_cam = np.matmul(cam_K, cam_RT)
    delta = 5
    length = 25

    # structs we use
    imgnames_, scales_, centers_, matnames_ = [], [], [], []
    garfs_, imgfs_ = [], []

    # get a list of .pkl files in the directory
    test_or_train = 'train' if isTrain else 'test'
    # go through all the .pkl files
    # for C in range(105):
    files = glob.glob('{}img_data_M_1/*H01'.format(dataset_path))
    files.sort()
    ind = 0
    random.seed("1234abcd")
    for i, dirname in enumerate(files):
        if not os.path.exists(os.path.join(dirname,'0249.jpg')):
            print('skipping {}'.format(dirname))
            continue
        print('working {}'.format(dirname))
        gts = pkl.load(open(os.path.join(dirname, 'gt.pkl'),'rb'),encoding='latin1')
        # pose(250*72), shape(10)
        gt_pose = torch.tensor(gts['pose'],dtype=torch.float32)
        gt_shape = torch.tensor(gts['shape'],dtype=torch.float32).unsqueeze(0).repeat(250,1)
        gt_trans = torch.tensor(gts['trans'],dtype=torch.float32)
        pred_output = smpl(betas=gt_shape.cuda(), body_pose=gt_pose[:,1:].cuda(), global_orient=gt_pose[:,0:1].cuda())
        center0  = 0
        scale0 = 0
        features = np.load(os.path.join(dirname, 'features.npz'))
        inds = []
        for frame in range(250-delta*(length-1)):
            trainind = random.randint(0,9)
            if is_train(trainind) != isTrain:
                continue
            img_name = []
            # joints
            gt_joints = pred_output.joints[frame].cpu().numpy()
            gt2d = np.matmul(blender_cam[:,:3], np.transpose(gt_joints)) + blender_cam[:,3:4]
            gt2d = np.transpose(gt2d)
            gt2d = gt2d / gt2d[:,2:3]
            # real gt_pose
            gt_pose_train = gt_pose[frame].numpy()
            # center and scale
            bbox = [min(gt2d[:,0]), min(gt2d[:,1]),
                max(gt2d[:,0]), max(gt2d[:,1])]
            center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
            scale = scaleFactor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200
            mat_name = int(dirname.split('/')[-1][5:7])
            #sequence
            for step in range(length):
                curframe = frame + step * delta
                img_name.append(os.path.join(dirname, '{:04d}.jpg'.format(curframe))[3:])
            #summary
            imgnames_.append(img_name)
            matnames_.append(mat_name)
            centers_.append(center)
            scales_.append(scale)
            inds.append(np.arange(frame, frame+(length-1)*delta+1, delta))
            if frame == 0:
                center0 = center
                scale0 = scale
        imgfs_.append(features['imgf'][np.array(inds)])
        garfs_.append(features['garf'][np.array(inds)])
        # if i <= 10:
        #     continue
        continue
        imgs = []
        normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
        for frame in range(250):
            imgname = os.path.join(dirname, '{:04d}.jpg'.format(frame))
            img = cv2.imread(imgname)[:,:,::-1].copy().astype(np.float32)
            img = rgb_processing(img, center0, scale0)
            img = torch.from_numpy(img).float()
            imgs.append(normalize_img(img))
        images = torch.stack(imgs, dim=0)
        with torch.no_grad():
            _, _, _, garf, imgf = coder(images, output_xf=True)
        feature_file = os.path.join(dirname, 'features.npz')
        np.savez(feature_file, garf=garf.cpu().numpy(), imgf=imgf.cpu().numpy())
        # print(garf[0,:10])
        # print(imgf[0,:10])
        # print(images[0,:,0,0])
        # import sys;sys.exit(0)
        # if i > 10:
        #     break
    imgfs_ = np.concatenate(imgfs_, axis=0)
    garfs_ = np.concatenate(garfs_, axis=0)
    print(imgfs_.shape)
    print(len(imgnames_))

    # store data
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path,
        'synthetic_M_H01_1_{}.npz'.format(test_or_train))
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       imgf=imgfs_,
                       garf=garfs_,
                       matname=matnames_)
