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

def eval_mgn_extract(dataset_path, out_path, isTrain=True):

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
    dirs = glob.glob('../Multi-Garment_dataset/*')
    dirs.sort()
    for dir_name in dirs:
            img_data_dir = dir_name
            map_data_dir = dir_name
            sim_data_dir = dir_name
            print('working {}'.format(dir_name))
            gts = pkl.load(open(os.path.join(sim_data_dir, 'registration.pkl'),'rb'),encoding='latin1')
            # pose(250*72), shape(10)
            gt_pose = torch.tensor(gts['pose'],dtype=torch.float32).unsqueeze(0)
            gt_shape = torch.tensor(gts['betas'],dtype=torch.float32).unsqueeze(0)
            gt_trans = torch.tensor(gts['trans'],dtype=torch.float32)
            pred_output = smpl(betas=gt_shape, body_pose=gt_pose[:,1:], global_orient=gt_pose[:,0:1])
            img_name = os.path.join(img_data_dir, '{:04d}.jpg'.format(0))
            map_name = '???'
            sim_name = os.path.join(sim_data_dir, 'smpl_registered.obj')
            # joints
            gt_joints = pred_output.joints[0].numpy()
            gt2d = np.matmul(blender_cam[:,:3], np.transpose(gt_joints)) + blender_cam[:,3:4]
            gt2d = np.transpose(gt2d)
            gt2d = gt2d / gt2d[:,2:3]
            # real gt_pose
            gt_pose_train = gt_pose[0].numpy()
#                 gt_pose_train[:3] = cv2.Rodrigues(np.matmul(
#                     cv2.Rodrigues(np.array([np.pi,0,0]))[0],
#                     cv2.Rodrigues(gt_pose_train[:3])[0]
#                     ))[0][:,0]
            # center and scale
            bbox = [min(gt2d[:,0]), min(gt2d[:,1]),
                max(gt2d[:,0]), max(gt2d[:,1])]
            center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
            scale = scaleFactor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200
            # with open(os.path.join(sim_data_dir, 'gtsmpl.obj'), 'w') as f:
            #     for p in pred_output.vertices[0].detach().cpu().numpy():
            #         f.write('v {} {} {}\n'.format(p[0],p[1],p[2]))
            #     for p in smpl.faces:
            #         f.write('f {} {} {}\n'.format(p[0]+1,p[1]+1,p[2]+1))
            scan_name = os.path.join(sim_data_dir, 'scan.obj')
            labels = np.load(os.path.join(sim_data_dir,'scan_labels.npy'))
            if (labels==0).sum() >= 0.5*len(labels) or (labels==5).sum() == 0:
                print('skipping ',sim_data_dir)
                continue
            vts = []
            clothvt = []
            tot = 0
            with open(scan_name, 'r') as f:
                for i, line in enumerate(f):
                    if line[0] == 'v' and line[1] == ' ':
                        if labels[tot] != 0:
                            vts.append(line)
                            clothvt.append([float(k) for k in line[:-1].split(' ')[1:]])
                        tot += 1
            # with open(os.path.join(sim_data_dir, 'gtcloth.obj'), 'w') as f:
            #     for line in vts:
            #         f.write(line)
            clothvt = np.array(clothvt, dtype=np.float32)
            #summary
            imgnames_.append(img_name[3:])
            mapnames_.append(clothvt)
            simnames_.append(sim_name[3:])
            poses_.append(gt_pose_train)
            shapes_.append(gt_shape[0].numpy())
            transs_.append(gt_trans.numpy())
            centers_.append(center)
            scales_.append(scale)
            parts_.append(gt2d[25:])
            openposes_.append(gt2d[:25])

    # store data
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path,
        'eval_mgn_{}.npz'.format(test_or_train))
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
