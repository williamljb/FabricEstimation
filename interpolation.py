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
['demoout/002.obj'],
['demoout/202.obj'],
['demoout/501.obj'],
['demoout/801.obj'],
['demoout/1001.obj']
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

def register(p, q, tgts):
    print(p,q)
    # if p==0:
    #     return
    targets = tgts
    mesh = io.load_objs_as_meshes(objlist[p]).cuda()
    sub = ops.SubdivideMeshes()
    mesh = (sub(mesh))
    n = len(objlist[p])
    deform_verts = torch.zeros(mesh.verts_packed().shape, device=torch.device("cuda:0"), requires_grad=True)
    # for q in range(11):
    sample_trg = pytorch3d.structures.Pointclouds([targets[q][0].detach()]) #targets[q].detach() #
    optimizer = torch.optim.Adam([deform_verts], lr=1e-3)#e-2, momentum=0.9)#
    edges = mesh.edges_packed()
    for i in range(200):
        optimizer.zero_grad()
        new_src_mesh = mesh.offset_verts(deform_verts)
        # sample_src = ops.sample_points_from_meshes(new_src_mesh, 16384 // n).reshape(1,-1,3)
        # sample_src = new_src_mesh.verts_packed().reshape(1,-1,3)
        # loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)

        loss_chamfer = pytorch3d.loss.point_mesh_face_distance(new_src_mesh, sample_trg)
        # loss_chamfer = geomloss(sample_trg[0], sample_src[0])
        # loss_chamfer = point_mesh_face_distance(new_src_mesh, sample_trg)
        def_edge = deform_verts[edges, :]
        loss_neigh = (def_edge[:,0]-def_edge[:,1]).norm(dim=1).norm(dim=0,p=float('inf'))
        loss_edge = mesh_edge_loss(new_src_mesh)
        loss_normal = mesh_normal_consistency(new_src_mesh)
        loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
        # Weighted sum of the losses
        loss = loss_chamfer*1 + loss_neigh*1 + loss_edge*0.1 + loss_normal * 0.01 + loss_laplacian * 0.01
        # print(loss_chamfer*1,loss_neigh*1,loss_edge*0.1,loss_normal * 0.01,loss_laplacian * 0.01)
        # Optimization step
        loss.backward()
        optimizer.step()
    for k in range(n):
        final_verts, final_faces = (new_src_mesh).get_mesh_verts_faces(k)
        final_obj = 'demoout/reg{}_{}_{}_{}.obj'.format(p,q,k,'final')
        io.save_obj(final_obj, final_verts, final_faces)
    # import sys;sys.exit(0)

if __name__ == '__main__':
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    npo = 16384
    nfps = 256
    nsample = 128
    nr = 1
    dim = 3
    #gt_cloth
    smpl = SMPL(config.SMPL_MODEL_DIR,
                batch_size=1,
                create_transl=False).to(device)
    nam0=args.img0.split('/')[-2]
    a0=pkl.load(open('../sim_data_CH/{}/gt.pkl'.format(nam0),'rb'),encoding='latin1')
    ind0 = int(args.img0.split('.')[-2].split('/')[-1])
    gt_cloth0, _ = load_points('../sim_data_CH/{}/{:04d}_*.obj'.format(nam0,ind0), npo)
    gt_cloth0 = (gt_cloth0 - a0['trans'][ind0]).float()
    gt_cloth0 = gt_cloth0.cuda()
    gt_pose0 = torch.tensor(a0['pose'][ind0],dtype=torch.float32).cuda()
    gt_rotmat0 = batch_rodrigues(gt_pose0.view(-1,3)).view(-1, 24, 3, 3)
    gt_shape0 = torch.tensor(a0['shape'],dtype=torch.float32).unsqueeze(0).cuda()
    smpl_body0 = smpl(betas=gt_shape0, body_pose=gt_rotmat0[:,1:], global_orient=gt_rotmat0[:,0].unsqueeze(1), pose2rot=False)
    gt_cloth0 = align(gt_cloth0, gt_rotmat0[:,0], smpl_body0.joints[:,8:9,:])

    nam1=args.img1.split('/')[-2]
    a1=pkl.load(open('../sim_data_CH/{}/gt.pkl'.format(nam1),'rb'),encoding='latin1')
    ind1 = int(args.img1.split('.')[-2].split('/')[-1])
    gt_cloth1, _ = load_points('../sim_data_CH/{}/{:04d}_*.obj'.format(nam1,ind1), npo)
    gt_cloth1 = (gt_cloth1 - a1['trans'][ind1]).float()
    gt_cloth1 = gt_cloth1.cuda()
    gt_pose1 = torch.tensor(a1['pose'][ind1],dtype=torch.float32).cuda()
    gt_rotmat1 = batch_rodrigues(gt_pose1.view(-1,3)).view(-1, 24, 3, 3)
    gt_shape1 = torch.tensor(a1['shape'],dtype=torch.float32).unsqueeze(0).cuda()
    smpl_body1 = smpl(betas=gt_shape1, body_pose=gt_rotmat1[:,1:], global_orient=gt_rotmat1[:,0].unsqueeze(1), pose2rot=False)
    gt_cloth1 = align(gt_cloth1, gt_rotmat1[:,0], smpl_body1.joints[:,8:9,:])


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

    with torch.no_grad():
        _, _, _, z_g0, z_l0 = model(gt_cloth0, None, torch.cat([gt_rotmat0.reshape(1,24*9)[:,9:],gt_shape0],dim=1))
        _, _, _, z_g1, z_l1 = model(gt_cloth1, None, torch.cat([gt_rotmat1.reshape(1,24*9)[:,9:],gt_shape1],dim=1))

        for i in range(11):
            w = 1-i/10.
            z_g = z_g0*w + z_g1*(1-w)
            z_l = z_l0*w + z_l1*(1-w)
            curpose = batch_quad(gt_pose0.view(-1,3))*w + batch_quad(gt_pose1.view(-1,3))*(1-w)
            rotmat = quat_to_rotmat(curpose).view(-1, 24, 3, 3)
            body = smpl(betas=gt_shape0, body_pose=rotmat[:,1:], global_orient=rotmat[:,0].unsqueeze(1), pose2rot=False)
            with open('demoout/human{}.obj'.format(i), 'w') as f:
                for p in body.vertices[0].cpu().numpy():
                    f.write('v {} {} {}\n'.format(p[0], p[1], p[2]))
                for p in smpl.faces:
                    f.write('f {} {} {}\n'.format(p[0]+1, p[1]+1, p[2]+1))
            rec, c = model.module.decode(z_g, z_l, torch.cat([rotmat.reshape(1,24*9)[:,9:],gt_shape0],dim=1))
            rec = rec.permute(0,2,3,1).reshape(1,-1,dim)
            rec = sample_cores(rec, 1, 16384, 1).permute(0,2,3,1).reshape(1,-1,dim)
            c = c.permute(0,2,3,1).reshape(1,-1,dim)
            rec = align_1(rec, rotmat[:,0], body.joints[:,8:9,:]).detach()
            recs.append(rec)
            # with open('demoout/core{}.obj'.format(i), 'w') as f:
            #     for p in c.reshape(-1,dim).cpu().numpy():
            #         f.write('v {} {} {}\n'.format(p[0],p[1],p[2]))
    l=[0,2,5,8,10]
    for i,j in enumerate(l):
        register(i,j,recs)
    for i in range(11):
        with open('demoout/cloth{}.obj'.format(i), 'w') as f:
            for p in recs[i].reshape(-1,dim).cpu().numpy():
                f.write('v {} {} {}\n'.format(p[0],p[1],p[2]))

