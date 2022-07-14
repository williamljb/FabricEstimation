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

from models import hmr, SMPL, Atlas, baseline_hmr
from utils.imutils import crop
from utils.renderer import Renderer
import config
import constants

import pickle as pkl
from utils.geometry import batch_rodrigues, batch_rodrigues_back, align, align_1
import trimesh

import matplotlib.pyplot as plt
import glob
import pytorch3d
from pytorch3d import io, ops
from pytorch3d.loss import chamfer_distance
from geomloss import SamplesLoss

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='logs_combined/std2/checkpoints/2020_05_10-14_59_14.pt', help='Path to pretrained checkpoint')
parser.add_argument('--pretrained_atlas', default='logs_atlas/0425_2103/checkpoints/2020_04_28-12_26_09.pt', help='Load a pretrained checkpoint at the beginning training') 
parser.add_argument('--img', type=str, required=True, help='Path to input image')
parser.add_argument('--bbox', type=str, default=None, help='Path to .json file containing bounding box coordinates')
parser.add_argument('--openpose', type=str, default=None, help='Path to .json containing openpose detections')
parser.add_argument('--outfile', type=str, default=None, help='Filename of output images. If not set use input filename.')

def bbox_from_openpose(openpose_file, rescale=1.2, detection_thresh=0.2):
    """Get center and scale for bounding box from openpose detections."""
    with open(openpose_file, 'r') as f:
        keypoints = json.load(f)['people'][0]['pose_keypoints_2d']
    keypoints = np.reshape(np.array(keypoints), (-1,3))
    valid = keypoints[:,-1] > detection_thresh
    valid_keypoints = keypoints[valid][:,:-1]
    center = valid_keypoints.mean(axis=0)
    bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)).max()
    # adjust bounding box tightness
    scale = bbox_size / 200.0
    scale *= rescale
    return center, scale

def bbox_from_json(bbox_file):
    """Get center and scale of bounding box from bounding box annotations.
    The expected format is [top_left(x), top_left(y), width, height].
    """
    with open(bbox_file, 'r') as f:
        bbox = np.array(json.load(f)['bbox']).astype(np.float32)
    ul_corner = bbox[:2]
    center = ul_corner + 0.5 * bbox[2:]
    width = max(bbox[2], bbox[3])
    scale = width / 200.0
    # make sure the bounding box is rectangular
    return center, scale

def visualize_img(img, gt_kp, gt_vert):
    """
    Overlays gt_kp and pred_kp on img.
    Draws vert with text.
    Renderer is an instance of SMPLRenderer.
    """
    siz = 224.0
    cam_K=np.array([
        [siz*50/36,0,siz/2],
        [0,siz*50/36,siz/2],
        [0,0,1]])
    cam_RT=np.array([
        [1,0,0,0],
        [0,-0.9996,0.0279,-0.2390],
        [0,-0.0279,-0.9996,3.1846]])
    blender_cam = np.matmul(cam_K, cam_RT)
    gt2d = np.matmul(blender_cam[:,:3], np.transpose(gt_vert)) + blender_cam[:,3:4]
    gt2d = np.transpose(gt2d)
    gt2d = gt2d[:,:2] / gt2d[:,2:3]
    gt2d = np.round(gt2d).astype(int)

    # for i in gt2d:
    #     cv2.circle(img, (i[0], i[1]), 1, [255,0,0], -1) #red
    # plt.ion()
    # plt.imshow(img)
    # plt.show()
    # import ipdb; ipdb.set_trace()
    return img

def process_image(img_file, bbox_file, openpose_file, input_res=224):
    """Read image, do preprocessing and possibly crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    """
    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    img = cv2.imread(img_file)[:,:,::-1].copy() # PyTorch does not support negative stride at the moment
    if bbox_file is None and openpose_file is None:
        # Assume that the person is centerered in the image
        height = img.shape[0]
        width = img.shape[1]
        center = np.array([width // 2, height // 2])
        scale = max(height, width) / 200
    else:
        if bbox_file is not None:
            center, scale = bbox_from_json(bbox_file)
        elif openpose_file is not None:
            center, scale = bbox_from_openpose(openpose_file)
    img0 = img
    img = crop(img, center, scale, (input_res, input_res))
    img = img.astype(np.float32) / 255.
    # img = img[:,::-1,:].copy()
    img = torch.from_numpy(img).permute(2,0,1)
    norm_img = normalize_img(img.clone())[None]
    return img, norm_img

    img = crop(img0, center, scale, (800, 800))
    img = img.astype(np.float32) / 255.
    # img = img[:,::-1,:].copy()
    img = torch.from_numpy(img).permute(2,0,1)
    return img, norm_img


def load_decoder(decoder, checkpoint_file):
    state_dict = torch.load(checkpoint_file)['model']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # if not k.startswith('module.de'):
        #     continue
        new_state_dict[k] = v
    # load params
    decoder.load_state_dict(new_state_dict, strict=True)
    print('Decoder loaded')

def load_points(path, num_points):
    lists = glob.glob(path)
    mesh = io.load_objs_as_meshes(lists)
    # torch.manual_seed(0)
    pts, nms = ops.sample_points_from_meshes(mesh, num_points // len(lists), return_normals=True)
    return pts.reshape([1,-1,3]), nms.reshape([1,-1,3]) * 0.1

def save_human(file, pred_vertices, smpl):
    with open(file, 'w') as f:
        for p in pred_vertices[0].cpu().numpy():
            f.write('v {} {} {}\n'.format(p[0], p[1], p[2]))
        for p in smpl.faces:
            f.write('f {} {} {}\n'.format(p[0]+1, p[1]+1, p[2]+1))

def do_move(pred_rotmat0, pred_betas1, smpl, rod, st, jointid, do_back):
    for i in range(25):
        rot = torch.tensor(cv2.Rodrigues(rod*i/25.)[0]).unsqueeze(0).unsqueeze(0).cuda().float()
        pred_rotmat = torch.cat([pred_rotmat0[:,:jointid,:,:], torch.matmul(rot,pred_rotmat0[:,jointid:jointid+1,:,:]), 
            pred_rotmat0[:,jointid+1:,:,:]], dim=1)
        pred_output = smpl(betas=pred_betas1, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices
        save_human('demoout/human{:03d}.obj'.format(i+st), pred_vertices, smpl)
    if do_back==0:
        return
    for i in range(25):
        rot = torch.tensor(cv2.Rodrigues(rod*(1.-i/25.))[0]).unsqueeze(0).unsqueeze(0).cuda().float()
        pred_rotmat = torch.cat([pred_rotmat0[:,:jointid,:,:], torch.matmul(rot,pred_rotmat0[:,jointid:jointid+1,:,:]), 
            pred_rotmat0[:,jointid+1:,:,:]], dim=1)
        pred_output = smpl(betas=pred_betas1, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices
        save_human('demoout/human{:03d}.obj'.format(i+st+25), pred_vertices, smpl)

def adjust(pred_rotmat, rot, jid):
    rot = torch.tensor(cv2.Rodrigues(rot)[0]).unsqueeze(0).unsqueeze(0).cuda().float()
    targetrot = torch.matmul(rot, pred_rotmat[:,jid:jid+1,:,:])
    pred_rotmat = torch.cat([pred_rotmat[:,:jid,:,:], targetrot, pred_rotmat[:,jid+1:,:,:]], dim=1)
    return pred_rotmat

if __name__ == '__main__':
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Load pretrained model
    smpl = SMPL(config.SMPL_MODEL_DIR,
                batch_size=1,
                create_transl=False).to(device)
    model = nn.DataParallel(hmr(config.SMPL_MEAN_PARAMS, smpl=smpl)).to(device)

    npo = 16384
    nfps = 256
    nsample = 128
    nr = 1
    dim = 3
    decoder = Atlas(nfps, nsample, nr, dim)
    decoder = nn.DataParallel(decoder).cuda()
    load_decoder(decoder, args.pretrained_atlas)

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
    del new_state_dict['module.smpl.betas']
    del new_state_dict['module.smpl.global_orient']
    del new_state_dict['module.smpl.body_pose']
    model.load_state_dict(new_state_dict, strict=False)

    # Load SMPL model
    model.eval()

    # Setup renderer for visualization
    renderer = Renderer(focal_length=constants.FOCAL_LENGTH, img_res=constants.IMG_RES, faces=smpl.faces)


    # Preprocess input image and generate predictions
    img, norm_img = process_image(args.img, args.bbox, args.openpose, input_res=constants.IMG_RES)

    with torch.no_grad():
        pred_rotmat, pred_betas, pred_camera, pred_cloth = model(norm_img.to(device))
        #print(pred_rotmat[0,8])
        ##ankle
        #rad=np.pi/3
        #rot=torch.tensor([[1,0,0],[0,np.cos(rad),-np.sin(rad)],[0,np.sin(rad),np.cos(rad)]], dtype=torch.float32).cuda().reshape([1,1,3,3])
        #pred_rotmat = torch.cat([pred_rotmat[:,:7,:,:], rot, rot, pred_rotmat[:,9:,:,:]], dim=1)
        #xaxis = np.array([1,0.,0.])
        #zaxis = np.array([0.,0.,1])
        ##root
        #pred_rotmat = adjust(pred_rotmat, -np.pi/36 * zaxis, 0)
        ##right hip
        #pred_rotmat = adjust(pred_rotmat, np.pi/18 * zaxis, 2)
        ##left hip
        #pred_rotmat = adjust(pred_rotmat, np.pi/18 * zaxis, 1)
        ##right knee
        #pred_rotmat = adjust(pred_rotmat, np.pi/36*3 * xaxis, 2)
        #pred_rotmat = adjust(pred_rotmat, -np.pi/36*3 * xaxis, 5)
        ##left knee
        #pred_rotmat = adjust(pred_rotmat, np.pi/36*3 * xaxis, 1)
        #pred_rotmat = adjust(pred_rotmat, -np.pi/36*3 * xaxis, 4)

        #pred_betas1 = torch.tensor([[1.0847,   2.58532,  1.40503,  0.69658,  1.56748,  0.06177, -0.32627,  0.42097, -0.08695,  0.24551]]).cuda()
        pred_betas1 = pred_betas
        pred_betas1[0,0] -= 1
        pred_betas1[0,1] -= 0
        pred_betas1[0,2] += 5
        pred_output = smpl(betas=pred_betas1, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices

        z_g = pred_cloth[:,:512].unsqueeze(2)
        z_l = pred_cloth[:,512:].reshape(1, 8, nfps)
        pred_abscloth, pred_cores = decoder.module.decode(z_g, z_l, 
            torch.cat([pred_rotmat.reshape(1,24*9)[:,9:],pred_betas],dim=1))
        pred_abscloth = pred_abscloth.permute(0,2,3,1).reshape(1,-1,dim)
        pred_cores = pred_cores.permute(0,2,3,1).reshape(1,-1,dim)

    #visualize cloth
        pred_abscloth = align_1(pred_abscloth, pred_rotmat[:,0], pred_output.joints[:,8:9,:])
        with open('demoout/ours_cloth.obj', 'w') as f:
            for p in pred_abscloth[0].cpu().numpy():
                f.write('v {} {} {}\n'.format(p[0],p[1],p[2]))
        save_human('demoout/human.obj', pred_vertices, smpl)
        #do_move(pred_rotmat, pred_betas1, smpl, np.array([np.radians(45.),0.,0.]), 0, 4,0)
        # do_move(pred_rotmat, pred_betas1, smpl, np.array([0.,0.,np.radians(45.)]), 0, 1,1)
        # do_move(pred_rotmat, pred_betas1, smpl, np.array([np.radians(-45.),0.,0.]), 50, 1,1)

    # Calculate camera parameters for rendering
    pred_vertices = pred_vertices[0].cpu().numpy()
    camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1)
    camera_translation = camera_translation[0].cpu().numpy()
    img = img.permute(1,2,0).cpu().numpy()
    
    # Render parametric shape
    img_shape = renderer(pred_vertices, camera_translation, img)
    visualize_img(img, pred_vertices, pred_output.joints[0].cpu().numpy())
    
    # Render side views
    aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0]
    center = pred_vertices.mean(axis=0)
    rot_vertices = np.dot((pred_vertices - center), aroundy) + center
    
    # Render non-parametric shape
    img_shape_side = renderer(rot_vertices, camera_translation, np.ones_like(img))

    #outfile = args.img.split('.')[0] if args.outfile is None else args.outfile
    outfile = 'demoout/output'

    # Save reconstructions
    cv2.imwrite(outfile + '_shape.png', 255 * img_shape[:,:,::-1])
    cv2.imwrite(outfile + '_shape_side.png', 255 * img_shape_side[:,:,::-1])
