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
parser.add_argument('--bbox', type=str, default=None, help='Path to .json file containing bounding box coordinates')
parser.add_argument('--openpose', type=str, default=None, help='Path to .json containing openpose detections')
parser.add_argument('--outfile', type=str, default=None, help='Filename of output images. If not set use input filename.')
geomloss = SamplesLoss(loss="sinkhorn", p=2, blur=.05) #chamfer_distance #

def load_points(path, num_points):
    lists = glob.glob(path)
    mesh = io.load_objs_as_meshes(lists)
    # torch.manual_seed(0)
    pts, nms = ops.sample_points_from_meshes(mesh, num_points // len(lists), return_normals=True)
    return pts.reshape([1,-1,3]), nms.reshape([1,-1,3]) * 0.1

def register(targets, mesh):
    sub = ops.SubdivideMeshes()
    mesh = (sub(mesh))
    deform_verts = torch.zeros(mesh.verts_packed().shape, device=torch.device("cuda:0"), requires_grad=True)
    # for q in range(11):
    sample_trg = pytorch3d.structures.Pointclouds([targets]) #targets[q].detach() #
    optimizer = torch.optim.Adam([deform_verts], lr=1e-4)#e-2, momentum=0.9)#
    edges = mesh.edges_packed()
    for i in range(200):
        optimizer.zero_grad()
        new_src_mesh = mesh.offset_verts(deform_verts)
        # sample_src = ops.sample_points_from_meshes(new_src_mesh, 16384).reshape(1,-1,3)
        # sample_src = new_src_mesh.verts_packed().reshape(1,-1,3)
        # loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)

        loss_chamfer = pytorch3d.loss.point_mesh_face_distance(new_src_mesh, sample_trg)
        # loss_chamfer = geomloss(sample_trg[0], sample_src[0])
        # loss_chamfer = point_mesh_face_distance(new_src_mesh, sample_trg)
        def_edge = deform_verts[edges, :]
        loss_neigh = (def_edge[:,0]-def_edge[:,1]).norm(dim=1).norm(dim=0,p=float('inf'))
        loss_edge = mesh_edge_loss(new_src_mesh)
        loss_normal = mesh_normal_consistency(new_src_mesh) #broken???
        loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
        # Weighted sum of the losses
        loss = loss_chamfer*1 + loss_neigh*1 + loss_edge*0.1 + loss_normal * 0.00001 + loss_laplacian * 0.01
        print(loss_chamfer*1,loss_neigh*1,loss_edge*0.1,loss_normal * 0.00001,loss_laplacian * 0.01)
        # Optimization step
        loss.backward()
        optimizer.step()
    final_verts, final_faces = (new_src_mesh).get_mesh_verts_faces(0)
    final_obj = 'demoout/reg.obj'.format()
    io.save_obj(final_obj, final_verts, final_faces)
    # import sys;sys.exit(0)

if __name__ == '__main__':
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    pcfile = 'demoout/ours_cloth.obj'
    meshfile = 'demoout/new.obj'
    mesh = io.load_objs_as_meshes([meshfile]).cuda()
    pts = []
    with open(pcfile, 'r') as f:
        for line in f:
            if line[0] == 'v' and line[1] == ' ':
                pt = [float(line[:-1].split(' ')[i+1]) for i in range(3)]
                pts.append(pt)
    pts = torch.tensor(pts).cuda()
    pts = sample_cores(pts.unsqueeze(0), 1, 16384, 1).permute(0,2,3,1).reshape(-1,3)
    register(pts, mesh)


