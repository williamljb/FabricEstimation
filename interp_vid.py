"""
Demo code

To run our method, you need a bounding box around the person. The person needs to be centered inside the bounding box and the bounding box should be relatively tight. You can either supply the bounding box directly or provide an [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) detection file. In the latter case we infer the bounding box from the detections.

In summary, we provide 3 different ways to use our demo code and models:
1. Provide only an input image (using ```--img```), in which case it is assumed that it is already cropped with the person centered in the image.
2. Provide an input image as before, together with the OpenPose detection .json (using ```--openpose```). Our code will use the detections to compute the bounding box and crop the image.
3. Provide an image and a bounding box (using ```--bbox```). The expected format for the json file can be seen in ```examples/im1010_bbox.json```.

Example with OpenPose detection .json
```
python3 demo.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.png --openpose=examples/im1010_openpose.json
```
Example with predefined Bounding Box
```
python3 demo.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.png --bbox=examples/im1010_bbox.json
```
Example with cropped and centered image
```
python3 demo.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.png
```

Running the previous command will save the results in ```examples/im1010_{shape,shape_side}.png```. The file ```im1010_shape.png``` shows the overlayed reconstruction of human shape. We also render a side view, saved in ```im1010_shape_side.png```.
"""

import torch
from torchvision.transforms import Normalize
import numpy as np
import cv2
import argparse
import json
import torch.nn as nn

from models import hmr, SMPL, Atlas, Baseline
from utils.imutils import crop, rot_aa
from utils.renderer import Renderer
import config
import constants

import pickle as pkl
from utils.geometry import batch_rodrigues, sample_cores, align, align_1, batch_quad, quat_to_rotmat
import trimesh

import matplotlib.pyplot as plt
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
    point_mesh_face_distance,
)
from geomloss import SamplesLoss
import pykeops
from fml.nn import SinkhornLoss
from torch_cluster import fps
import glob
import pytorch3d
from pytorch3d import io, ops

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='logs_atlas/0425_2103/checkpoints/2020_04_28-12_26_09.pt', help='Path to pretrained checkpoint')
parser.add_argument('--img0', type=str, default='../img_data_CH/C000M00H07/0004.jpg', help='Path to input image')
parser.add_argument('--img1', type=str, default='../img_data_CH/C001M00H07/0004.jpg', help='Path to input image')
parser.add_argument('--bbox', type=str, default=None, help='Path to .json file containing bounding box coordinates')
parser.add_argument('--openpose', type=str, default=None, help='Path to .json containing openpose detections')
parser.add_argument('--outfile', type=str, default=None, help='Filename of output images. If not set use input filename.')
#[0,1;7;4][7,56;7;4][35;3;4,40][18;4;4,35][62,41;7;4]
objlist=[
[18,4,35],
[18,4,4],
[0,7,4],
[1,7,4],
[7,7,4],
[56,7,4],
[62,7,4],
[41,7,4],
[35,3,4],
[35,3,40],
]
geomloss = SamplesLoss(loss="sinkhorn", p=2, blur=.05) #chamfer_distance #

def load_points(path, num_points):
    lists = glob.glob(path)
    mesh = io.load_objs_as_meshes(lists)
    # torch.manual_seed(0)
    pts, nms = ops.sample_points_from_meshes(mesh, num_points // len(lists), return_normals=True)
    return pts.reshape([1,-1,3]), nms.reshape([1,-1,3]) * 0.1

def filter(pts, mesh):
    dist, idx, nn = pytorch3d.ops.knn_points(pts, mesh.verts_packed().unsqueeze(0), return_nn=True)
    dirs = (pts[0]-nn[0,:,0,:])/dist[0]
    mask = ((dirs*mesh.verts_normals_packed()[idx[0,:,0],:]).sum(dim=1,keepdim=True)<1e-3) #| (dist[0]<1e-3)
    print(mask.sum())
    return pts[0,mask[:,0],:]

if __name__ == '__main__':
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    npo = 16384
    nfps = 256
    nsample = 128
    nr = 1
    dim = 3
    n=len(objlist)
    #gt_cloth
    smpl = SMPL(config.SMPL_MODEL_DIR,
                batch_size=1,
                create_transl=False).to(device)
    gt_cloths,gt_poses,gt_rotmats,gt_shapes = [],[],[],[]
    for i in range(n):
        img = '../img_data_CH/C{:03d}M00H0{}/{:04d}.jpg'.format(objlist[i][0],objlist[i][1],objlist[i][2])
        nam=img.split('/')[-2]
        a=pkl.load(open('../sim_data_CH/{}/gt.pkl'.format(nam),'rb'),encoding='latin1')
        ind = objlist[i][2]
        gt_cloth, _ = load_points('../sim_data_CH/{}/{:04d}_*.obj'.format(nam,ind), npo)
        gt_cloth = (gt_cloth - a['trans'][ind]).float()
        gt_cloth = gt_cloth.cuda()
        gt_pose = torch.tensor(a['pose'][ind],dtype=torch.float32).cuda()
        gt_rotmat = batch_rodrigues(gt_pose.view(-1,3)).view(-1, 24, 3, 3)
        gt_shape = torch.tensor(a['shape'],dtype=torch.float32).unsqueeze(0).cuda()
        smpl_body = smpl(betas=gt_shape, body_pose=gt_rotmat[:,1:], global_orient=gt_rotmat[:,0].unsqueeze(1), pose2rot=False)
        gt_cloth = align(gt_cloth, gt_rotmat[:,0], smpl_body.joints[:,8:9,:])
        gt_cloths.append(gt_cloth)
        gt_poses.append(gt_pose)
        gt_rotmats.append(gt_rotmat)
        gt_shapes.append(gt_shape)


    # Load pretrained model
    model = nn.DataParallel(Atlas(nfps, nsample, nr, dim)).to(device)
    checkpoint = torch.load(args.checkpoint)
    state_dict = checkpoint['model']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if not k.startswith('module.'):
            name = 'module.'+k
        else:
            name = k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=True)
    # Load SMPL model
    model.eval()

    # with open('demoout/human.obj', 'w') as f:
    #     for p in smpl_body0.vertices[0].cpu().numpy():
    #         f.write('v {} {} {}\n'.format(p[0], p[1], p[2]))
    #     for p in smpl.faces:
    #         f.write('f {} {} {}\n'.format(p[0]+1, p[1]+1, p[2]+1))
    recs = []
    zgs,zls=[],[]

    with torch.no_grad():
        for i in range(n):
            _, _, _, z_g, z_l = model(gt_cloths[i], None, torch.cat([gt_rotmats[i].reshape(1,24*9)[:,9:],gt_shapes[i]],dim=1))
            zgs.append(z_g)
            zls.append(z_l)

        for ido in range(n-1):
            z_g0 = zgs[ido]; z_g1 = zgs[ido+1]
            z_l0 = zls[ido]; z_l1 = zls[ido+1]
            gt_pose0 = gt_poses[ido]; gt_pose1 = gt_poses[ido+1]
            gt_shape0 = gt_shapes[ido];
            for i in range(25):
                frameid = ido*25+i
                w = 1-i/25.
                z_g = z_g0*w + z_g1*(1-w)
                z_l = z_l0*w + z_l1*(1-w)
                curpose = batch_quad(gt_pose0.view(-1,3))*w + batch_quad(gt_pose1.view(-1,3))*(1-w)
                rotmat = quat_to_rotmat(curpose).view(-1, 24, 3, 3)
                body = smpl(betas=gt_shape0, body_pose=rotmat[:,1:], global_orient=rotmat[:,0].unsqueeze(1), pose2rot=False)
                with open('demoout/human{}.obj'.format(frameid), 'w') as f:
                    for p in body.vertices[0].cpu().numpy():
                        f.write('v {} {} {}\n'.format(p[0], p[1], p[2]))
                    for p in smpl.faces:
                        f.write('f {} {} {}\n'.format(p[0]+1, p[1]+1, p[2]+1))
                rec, _ = model.module.decode(z_g, z_l, torch.cat([rotmat.reshape(1,24*9)[:,9:],gt_shape0],dim=1))
                rec = rec.permute(0,2,3,1).reshape(1,-1,dim)
                # rec = sample_cores(rec, 1, 16384, 1).permute(0,2,3,1).reshape(1,-1,dim)
                rec = align_1(rec, rotmat[:,0], body.joints[:,8:9,:]).detach()
                with open('demoout/cloth{}.obj'.format(frameid), 'w') as f:
                    for p in rec.reshape(-1,dim).cpu().numpy():
                        f.write('v {} {} {}\n'.format(p[0],p[1],p[2]))

