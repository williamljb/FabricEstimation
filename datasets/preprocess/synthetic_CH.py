import os
import sys
import cv2
import glob
import h5py
import numpy as np
import argparse
import pickle as pkl
import torch
from models import SMPL
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

train_set = [i for i in range(10) if i % 10 != 6]
test_set = [i for i in range(10) if i % 10 == 6]

def synthetic_CH_extract(dataset_path, out_path, isTrain=True):

    smpl = SMPL('data/smpl/',
                batch_size=250,
                create_transl=False)

    # scale factor
    scaleFactor = 1.2
    blender_cam = np.matmul(cam_K, cam_RT)

    # structs we use
    imgnames_, scales_, centers_, parts_, simnames_ = [], [], [], [], []
    poses_, shapes_, mapnames_, openposes_, transs_ = [], [], [], [], []

    # get a list of .pkl files in the directory
    test_or_train = 'train' if isTrain else 'test'
    # go through all the .pkl files
    # for C in range(105):
    for C0 in (train_set if isTrain else test_set):
        C=C0*10
        for H in (train_set if isTrain else test_set):
            dir_name = 'C{:03d}M00H{:02d}'.format(C,H)
            img_data_dir = os.path.join('img_data_CH', dir_name)
            map_data_dir = os.path.join('map_data_splitCH', dir_name)
            sim_data_dir = os.path.join('sim_data_CH', dir_name)
            if not os.path.exists(os.path.join(dataset_path, map_data_dir,'0249.npz')):
                print('skipping {}'.format(dir_name))
                continue
            print('working {}'.format(dir_name))
            gts = pkl.load(open(os.path.join(dataset_path, sim_data_dir, 'gt.pkl'),'rb'),encoding='latin1')
            # pose(250*72), shape(10)
            gt_pose = torch.tensor(gts['pose'],dtype=torch.float32)
            gt_shape = torch.tensor(gts['shape'],dtype=torch.float32).unsqueeze(0).repeat(250,1)
            gt_trans = torch.tensor(gts['trans'],dtype=torch.float32)
            pred_output = smpl(betas=gt_shape, body_pose=gt_pose[:,1:], global_orient=gt_pose[:,0:1])
            for frame in range(250):
                img_name = os.path.join(img_data_dir, '{:04d}.jpg'.format(frame))
                map_name = os.path.join(map_data_dir, '{:04d}.npz'.format(frame))
                sim_name = os.path.join(sim_data_dir, '{:04d}_*.obj'.format(frame))
                # joints
                gt_joints = pred_output.joints[frame].numpy()
                gt2d = np.matmul(blender_cam[:,:3], np.transpose(gt_joints)) + blender_cam[:,3:4]
                gt2d = np.transpose(gt2d)
                gt2d = gt2d / gt2d[:,2:3]
                # real gt_pose
                gt_pose_train = gt_pose[frame].numpy()
#                 gt_pose_train[:3] = cv2.Rodrigues(np.matmul(
#                     cv2.Rodrigues(np.array([np.pi,0,0]))[0],
#                     cv2.Rodrigues(gt_pose_train[:3])[0]
#                     ))[0][:,0]
                # center and scale
                bbox = [min(gt2d[:,0]), min(gt2d[:,1]),
                    max(gt2d[:,0]), max(gt2d[:,1])]
                center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
                scale = scaleFactor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200
                #summary
                imgnames_.append(img_name)
                mapnames_.append(map_name)
                simnames_.append(sim_name)
                poses_.append(gt_pose_train)
                shapes_.append(gt_shape[frame].numpy())
                transs_.append(gt_trans[frame].numpy())
                centers_.append(center)
                scales_.append(scale)
                parts_.append(gt2d[25:])
                openposes_.append(gt2d[:25])

    # store data
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path,
        'compare_CH_{}.npz'.format(test_or_train))
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       pose=poses_,
                       shape=shapes_,
                       trans=transs_,
                       part=parts_,
                       mapname=mapnames_,
                       simname=simnames_,
                       openpose=openposes_)
