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

from models import hmr, SMPL
from models.atlas_fps import Atlas
from utils.imutils import crop
from utils.renderer import Renderer
import config
import constants

import pickle as pkl
from utils.geometry import batch_rodrigues, batch_rodrigues_back
import trimesh

import matplotlib.pyplot as plt
from pytorch3d.loss import chamfer_distance
from geomloss import SamplesLoss
import pykeops
from fml.nn import SinkhornLoss
from torch_cluster import fps
import glob
import pytorch3d
from pytorch3d import io, ops

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True, help='Path to pretrained checkpoint')
parser.add_argument('--img', type=str, required=True, help='Path to input image')
parser.add_argument('--bbox', type=str, default=None, help='Path to .json file containing bounding box coordinates')
parser.add_argument('--openpose', type=str, default=None, help='Path to .json containing openpose detections')
parser.add_argument('--outfile', type=str, default=None, help='Filename of output images. If not set use input filename.')

def sample_points(gt_abscloth, batch_size, num_fps, num_sample):
    from scipy.spatial import cKDTree
#     num_multi = num_sample
#     ind = torch.multinomial(gt_abscloth[:,3,:], num_multi, replacement=True).view(batch_size,1,num_sample)
#     ind0 = torch.arange(batch_size).view(batch_size,1,1)
#     ind1 = torch.arange(3).view(1,3,1)
#     gt_abscloth = gt_abscloth[:,:3,:][ind0,ind1,ind]
    gt_abscloth = gt_abscloth.permute(0,2,1)
    dim = gt_abscloth.shape[1]

    num_tot = gt_abscloth.shape[2]+1
    gt_abscloth = gt_abscloth.permute(0,2,1)
    com = gt_abscloth.mean(dim=1,keepdim=True)
    clothNcom = torch.cat([com, gt_abscloth],dim=1).reshape(-1,dim)
    batchx = torch.arange(batch_size).view(batch_size,1).repeat(1,num_tot).view(-1).cuda()
#     print(gt_abscloth.shape)
#     print(num_tot*batch_size)
    ind = fps(clothNcom, batchx, ratio=(num_fps+0.9)/num_tot,random_start=False)
    cores = clothNcom[ind]
    cores = cores.view(batch_size,-1,dim)[:,1:,:] # bs * num_fps * 3
    
    #nearest point split
    # batchx = torch.arange(batch_size).view(batch_size,1).repeat(1,num_tot-1).view(-1).cuda()
    # batchy = torch.arange(batch_size).view(batch_size,1).repeat(1,num_fps).view(-1).cuda()
    # cluster = nearest(gt_abscloth.view(-1,dim), cores.reshape(-1,dim), batchx, batchy).reshape(batch_size,num_tot-1)
    # cluster = cluster - torch.arange(batch_size).view(batch_size,1).cuda()*num_fps
    # ind = []
    # for i in range(num_fps):
    #     ind.append(torch.multinomial((cluster==i)*1., num_sample, replacement=True).reshape(batch_size,num_sample,1))
    # ind = torch.stack(ind, dim=1) # bs * num_fps * num_sample * 1

    # knn from scipy
    ind = []
    cores_cpu = cores.cpu().numpy()
    for i in range(batch_size):
        kdtree = cKDTree(gt_abscloth[i].cpu().numpy())
        _, ii = kdtree.query(cores_cpu[i], num_sample) # nfps*nsample
        ind.append(ii)
    ind = torch.LongTensor(ind).unsqueeze(3).cuda() #bs*nfps*nsample*1

    ind0 = torch.arange(batch_size).view(batch_size,1,1,1)
    ind1 = torch.arange(dim).view(1,1,1,dim)
    pts0 = gt_abscloth[ind0,ind,ind1] - cores.unsqueeze(2) # bs * num_fps * num_sample * 3

    coresout = cores.permute(0,2,1).unsqueeze(2).contiguous() #bs*3*1*nfps
    pts0out = pts0.permute(0,3,1,2).contiguous() #bs*3*nfps*nsample
    
    return pts0out, coresout


def get_body_pts( batch_size, vertices):
    p = pkl.load(open(constants.SAMPLE_FILE, 'rb'))
    pvts = torch.tensor(p['vertices']).long()
    pws = torch.tensor(p['weights']).float().unsqueeze(1)
    flip_vertices = vertices + 0
    body_pts = flip_vertices[:,constants.VALID_PTS0,:].reshape([batch_size,-1,3])
    vts = torch.arange(batch_size).long().reshape([-1,1,1]).repeat(1,256*256,3)
    tmp = body_pts[vts,pvts.repeat(batch_size,1,1)]
    fin = torch.matmul(pws.cuda(), tmp)
    body_pts = fin.reshape([batch_size,256,256,3]).permute(0,3,1,2)
    return body_pts.roll(64,3)

def recover_obj(disp_map, legal_map, body_pts_ori, img0=None):
    p = pkl.load(open('dat.pkl', 'rb'))
    size = 256
    # body_pts = trimesh.load('/scratch1/CMH/sim_data_CH/C035M00H08/obs0123_00.obj', process=False)
    # print(body_pts.vertices.shape)
    # with open('/nfshomes/liangjb/Downloads/tmp1.obj', 'w') as f:
    #     for pt in body_pts.vertices:
    #         f.write('v {} {} {}\n'.format(pt[0],pt[1],pt[2]))
    body_pts = get_body_pts(1,body_pts_ori)[0].permute(1,2,0).reshape([256*256,3]).cpu().numpy()
    with open('./tmpbody.obj', 'w') as f:
        for pt in body_pts.reshape([65536,3]):
            f.write('v {} {} {}\n'.format(pt[0],pt[1],pt[2]))
    disp = disp_map.reshape([-1,3]).cpu().detach().numpy()
    cloth_pts = disp + body_pts
    img = cloth_pts.reshape([256,256,3])
    retimg = img
    if img0 is None:
        img0 = img
        img = img - np.min(img0)
    else:
        # img = img-np.min(img0)
        img = img - img0
        print(np.max(img),np.min(img))
        print(np.where(img==np.min(img)))
        img = np.abs(img)
    img = img / (np.max(img0)-np.min(img0))
    plt.imshow(img)
    plt.show()
    legal_map = legal_map.reshape([-1]).cpu().detach().numpy()
    valid_pts = cloth_pts[legal_map > 0.5,:]
    with open('./tmp.obj', 'w') as f:
        for pt in valid_pts:
            f.write('v {} {} {}\n'.format(pt[0],pt[1],pt[2]))
    return retimg

def load_points(path, num_points):
    lists = glob.glob(path)
    mesh = io.load_objs_as_meshes(lists)
    pts, nms = ops.sample_points_from_meshes(mesh, num_points // len(lists), return_normals=True)
    return pts.reshape([1,-1,3]), nms.reshape([1,-1,3]) * 0.1

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
    nam=args.img.split('/')[-2]
    a=pkl.load(open('../sim_data_CH/{}/gt.pkl'.format(nam),'rb'),encoding='latin1')
    ind = int(args.img.split('.')[-2].split('/')[-1])
    gt_cloth, gt_norm = load_points('../sim_data_CH/{}/{:04d}_*.obj'.format(nam,ind), npo)
    gt_cloth = (gt_cloth - a['trans'][ind]).float()
    if dim == 6:
        gt_cloth = torch.cat([gt_cloth, gt_norm], dim=2)
    gt_cloth = gt_cloth.cuda()
    
    gt_pose_train = a['pose'][ind]
    gt_pose = torch.tensor(gt_pose_train,dtype=torch.float32).cuda()
    gt_rotmat = batch_rodrigues(gt_pose.view(-1,3)).view(-1, 24, 3, 3)
    gt_shape = torch.tensor(a['shape'],dtype=torch.float32).unsqueeze(0).cuda()
    gt_vert = smpl(betas=gt_shape, body_pose=gt_rotmat[:,1:], global_orient=gt_rotmat[:,0].unsqueeze(1), pose2rot=False).vertices
    gt_bdp = get_body_pts(1, gt_vert)
    with open('demoout/gt_cloth.obj', 'w') as f:
        for p in gt_cloth[0].cpu().numpy():
            f.write('v {} {} {}\n'.format(p[0],p[1],p[2]))
    num_sample = nsample
    gt_abscloth, gt_cores = sample_points(gt_cloth, 1, nfps, num_sample)
    with open('demoout/gtcore.obj', 'w') as f:
        for p in gt_cores[0].permute(1,2,0)[0].cpu().numpy():
            f.write('v {} {} {}\n'.format(p[0],p[1],p[2]))
    gtout = gt_abscloth[0].permute(1,2,0)+gt_cores[0].permute(2,1,0)
    with open('demoout/gt.obj', 'w') as f:
        for p in gtout.reshape(-1,dim).cpu().numpy():
            f.write('v {} {} {}\n'.format(p[0],p[1],p[2]))
    # Load pretrained model
    model = nn.DataParallel(Atlas(nfps, num_sample, nr, dim)).to(device)
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
    # for i in range(nr):
    #     with open('demoout/prim{}.obj'.format(i), 'w') as f:
    #         for p in getattr(model.module.decoder,'patch{}'.format(i))[0].permute(1,0).detach().cpu().numpy():
    #             f.write('v {} {} {}\n'.format(p[0],p[1],p[2]))

    with torch.no_grad():
        pred_cloth, pred_cores = model(gt_abscloth,gt_cores, torch.cat([gt_rotmat.reshape(1,24*9),gt_shape],dim=1))
        pred_cores = pred_cores.permute(0,2,3,1)[0]
        pred_cloth = pred_cloth.permute(0,2,3,1)[0]
    # print(gt_abscloth.shape)
    # print(pred_cloth.shape)
    geomloss = SamplesLoss(loss="sinkhorn", p=2, blur=.05) #chamfer_distance #
    # pykeops.clean_pykeops()
    print(geomloss(pred_cores, gt_cores.permute(0,2,3,1)[0])[0])
    print(chamfer_distance(pred_cloth.reshape(1,-1,dim), gtout.reshape(1,-1,dim))[0])
    pred_cores = pred_cores[0].cpu().numpy()
    with open('demoout/outcore.obj', 'w') as f:
        for p in pred_cores:
            f.write('v {} {} {}\n'.format(p[0],p[1],p[2]))
    pred_cloth = pred_cloth.reshape([-1,dim])
    with open('demoout/out.obj', 'w') as f:
        for p in pred_cloth:
            f.write('v {} {} {}\n'.format(p[0],p[1],p[2]))
