import torch
import torch.nn as nn
import numpy as np
from torchgeometry import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis
import cv2

from datasets import MixedDataset, BaseDataset
from models import hmr, SMPL, Atlasnet, Atlas, AtlasDecoder, AtlasEncoder, baseline_hmr
from smplify import SMPLify
from utils.geometry import batch_rodrigues, perspective_projection, estimate_translation, sample_cores, align
from utils.renderer import Renderer
from utils import BaseTrainer

import config
import constants
from .fits_dict import FitsDict
import time
import pickle as pkl

from pytorch3d.loss import chamfer_distance
from geomloss import SamplesLoss

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class Trainer(BaseTrainer):

    def load_decoder(self, checkpoint_file):
        state_dict = torch.load(checkpoint_file)['model']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # if not k.startswith('module.de'):
            # if k.startswith('module.de_l'):
            #     continue
            new_state_dict[k] = v
        # load params
        # self.decoder.load_state_dict(new_state_dict, strict=True)
        # self.encoder.load_state_dict(new_state_dict, strict=True)
        self.coder.load_state_dict(new_state_dict, strict=True)
        print('Decoder loaded')
    
    def init_fn(self):
        self.train_ds = BaseDataset(self.options, 'synthetic_CH', ignore_3d=self.options.ignore_3d, is_train=True)
        self.test_ds = BaseDataset(self.options, 'synthetic_CH', ignore_3d=self.options.ignore_3d, is_train=False)

        self.smpl = SMPL(config.SMPL_MODEL_DIR,
                         batch_size=self.options.batch_size,
                         create_transl=False).to(self.device)
        self.model = hmr(config.SMPL_MEAN_PARAMS, pretrained=True, smpl=self.smpl)
        self.model = nn.DataParallel(self.model).to(self.device)
        # self.decoder = AtlasDecoder(self.options.num_fps,
        #     self.options.num_sample, self.options.num_patches, self.options.dim)
        # self.decoder = nn.DataParallel(self.decoder).to(self.device)
        # self.encoder = AtlasEncoder(self.options.num_fps,
        #     self.options.num_sample, self.options.num_patches, self.options.dim)
        # self.encoder = nn.DataParallel(self.encoder).to(self.device)
        self.coder = Atlas(self.options.num_fps,
            self.options.num_sample, self.options.num_patches, self.options.dim)
        self.coder = nn.DataParallel(self.coder).to(self.device)
        self.load_decoder(self.options.pretrained_atlas)
        self.optimizer = torch.optim.Adam([{'params':self.model.module.ori_param(),'lr':self.options.lr*1},
            {'params':self.model.module.new_param()}],
                                          lr=self.options.lr)
        # Per-vertex loss on the shape
        self.criterion_shape = nn.L1Loss().to(self.device)
        # Keypoint (2D and 3D) loss
        # No reduction because confidence weighting needs to be applied
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        # Loss for SMPL parameter regression
        self.criterion_regr = nn.MSELoss().to(self.device)
        self.criterion_img = nn.MSELoss().to(self.device)
        self.criterion_class = nn.BCEWithLogitsLoss().to(self.device)
        self.criterionVGG = VGGLoss().to(self.device)
        self.models_dict = {'model': self.model}
        self.optimizers_dict = {'optimizer': self.optimizer}
        self.focal_length = constants.FOCAL_LENGTH
        self.pers_rot = torch.FloatTensor([[1.,0,0],[0,-1,0],[0,0,-1]]).to(self.device).unsqueeze(0)
        self.geomloss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)

        # Initialize SMPLify fitting module
        if self.options.pretrained_checkpoint is not None:
            self.load_pretrained(checkpoint_file=self.options.pretrained_checkpoint)

        # Create renderer
        self.renderer = Renderer(focal_length=self.focal_length, img_res=self.options.img_res, faces=self.smpl.faces)

    def finalize(self):
        pass

    def cloth_loss(self, pred_cloth, gt_cloth):
        return chamfer_distance(pred_cloth, gt_cloth)[0]

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d, openpose_weight, gt_weight):
        """ Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        conf[:, :25] *= openpose_weight
        conf[:, 25:] *= gt_weight
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
        return loss

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d):
        """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence.
        """
        pred_keypoints_3d = pred_keypoints_3d[:, 25:, :]
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
        conf = conf[has_pose_3d == 1]
        pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2,:] + pred_keypoints_3d[:, 3,:]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            return (conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def shape_loss(self, pred_vertices, gt_vertices, has_smpl):
        """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
        pred_vertices_with_shape = pred_vertices[has_smpl == 1]
        gt_vertices_with_shape = gt_vertices[has_smpl == 1]
        if len(gt_vertices_with_shape) > 0:
            return self.criterion_shape(pred_vertices_with_shape, gt_vertices_with_shape)
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl):
        pred_rotmat_valid = pred_rotmat[has_smpl == 1]
        gt_rotmat_valid = batch_rodrigues(gt_pose.view(-1,3)).view(-1, 24, 3, 3)[has_smpl == 1]
        pred_betas_valid = pred_betas[has_smpl == 1]
        gt_betas_valid = gt_betas[has_smpl == 1]
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss_regr_pose, loss_regr_betas

    def cloth_loss_g(self, pred_cloth, gt_cloth):
        # return chamfer_distance(pred_cloth, gt_cloth)
        # return self.sinkhorn(pred_cloth,gt_cloth)
        # return chamfer_distance(pred_cloth.reshape(-1,1,3), gt_cloth.reshape(-1,1,3))[0]
        loss = 0
        for i in range(pred_cloth.shape[0]):
            loss = loss + self.geomloss(pred_cloth[i], gt_cloth[i])
        return loss / pred_cloth.shape[0]


    def train_step(self, input_batch, is_train=True):
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        # Get data from the batch
        images = input_batch['img'] # input image
        gt_keypoints_2d = input_batch['keypoints'] # 2D keypoints
        gt_pose = input_batch['pose'] # SMPL pose parameters
        gt_pose_for_cloth = input_batch['pose_befrot'] # SMPL pose parameters
        gt_betas = input_batch['betas'] # SMPL beta parameters
        gt_joints = input_batch['pose_3d'] # 3D pose
        gt_cloth = input_batch['map']
        gt_norm = input_batch['normmap']
        has_smpl = input_batch['has_smpl'].byte() # flag that indicates whether SMPL parameters are valid
        has_pose_3d = input_batch['has_pose_3d'].byte() # flag that indicates whether 3D pose is valid
        is_flipped = input_batch['is_flipped'] # flag that indicates whether image was flipped during data augmentation
        rot_angle = input_batch['rot_angle'] # rotation angle used for data augmentation
        dataset_name = input_batch['dataset_name'] # name of the dataset the image comes from
        indices = input_batch['sample_index'] # index of example inside its dataset
        batch_size = images.shape[0]

        # Get GT vertices and model joints
        # Note that gt_model_joints is different from gt_joints as it comes from SMPL
        gt_out = self.smpl(betas=gt_betas, body_pose=gt_pose[:,3:], global_orient=gt_pose[:,:3])
        gt_model_joints = gt_out.joints
        gt_vertices = gt_out.vertices

        # De-normalize 2D keypoints from [-1,1] to pixel space
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        gt_keypoints_2d_orig[:, :, :-1] = 0.5 * self.options.img_res * (gt_keypoints_2d_orig[:, :, :-1] + 1)

        # Estimate camera translation given the model joints and 2D keypoints
        # by minimizing a weighted least squares loss
        gt_cam_t = estimate_translation(gt_model_joints, gt_keypoints_2d_orig, focal_length=self.focal_length, img_size=self.options.img_res)

        # Feed images in the network to predict camera and SMPL parameters
        pred_rotmat, pred_betas, pred_camera, pred_cloth = self.model(images)
        # #for baseline only!
        # pred_rotmat=batch_rodrigues(gt_pose.reshape([-1,3])).view(batch_size, 24, 3, 3)
        # pred_betas=gt_betas
        # pred_cam_t=gt_cam_t

        pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints

        # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
        # This camera translation can be used in a full perspective projection
        pred_cam_t = torch.stack([pred_camera[:,1],
                                  pred_camera[:,2],
                                  2*self.focal_length/(self.options.img_res * pred_camera[:,0] +1e-9)],dim=-1)
        #fix flip displacement by +2*root
        # pred_cam_t[:,1:] = pred_cam_t[:,1:] + pred_joints[:,8,1:]*2


        camera_center = torch.zeros(batch_size, 2, device=self.device)
        pred_keypoints_2d = perspective_projection(pred_joints,
                                                   rotation=self.pers_rot.expand(batch_size, -1, -1),
                                                   translation=pred_cam_t,
                                                   focal_length=self.focal_length,
                                                   camera_center=camera_center)
        # Normalize keypoints to [-1,1]
        pred_keypoints_2d = pred_keypoints_2d / (self.options.img_res / 2.)

        if self.options.dim == 3:
            gt_abscloth0 = gt_cloth
        else:
            gt_abscloth0 = torch.cat([gt_cloth, gt_norm], dim=2)
        dim = self.options.dim
        num_sample = self.options.num_sample
        num_fps = self.options.num_fps

        # Compute loss on cloth
        z_g = pred_cloth[:,:512].unsqueeze(2)
        z_l = pred_cloth[:,512:].reshape(batch_size, 8, self.options.num_fps)
        gt_rot = batch_rodrigues(gt_pose_for_cloth.reshape([-1,3])).reshape([batch_size,24*9])
        gtbody_rep = torch.cat([gt_rot[:,9:],gt_betas],dim=1)
        if is_train:
            body_rep = torch.cat([gt_rot[:,9:],gt_betas],dim=1)
        else:
            body_rep = torch.cat([gt_rot[:,9:],gt_betas],dim=1)
            # body_rep = torch.cat([pred_rotmat.reshape(batch_size,9*24)[:,9:],pred_betas],dim=1)
        gt_abscloth0 = align(gt_abscloth0, gt_rot[:,:9].reshape(batch_size, 3, 3), gt_out.joints[:,8:9,:])
        # pred_abscloth = self.decoder(z_g, z_l,body_rep).permute(0,2,1)
        # loss_cloth = self.cloth_loss(pred_abscloth, gt_abscloth)

        # _, gt_g, gt_l = self.encoder(gt_abscloth, body_rep)
        # loss_cloth = self.criterion_regr(gt_g, z_g) + self.criterion_regr(gt_l, z_l)
        # pred_abscloth = torch.cat([z_g.reshape(batch_size,-1),z_l.reshape(batch_size,-1)],dim=1)
        # gt_abscloth = torch.cat([gt_g.reshape(batch_size,-1), gt_l.reshape(batch_size,-1)], dim=1)

        pred_cloth, pred_cores = self.coder.module.decode(z_g, z_l, body_rep)
        gt_abscloth, gt_atlas, gt_cores, gt_g, gt_l = self.coder(gt_abscloth0, None, gtbody_rep)
        pred_cores = pred_cores.permute(0,2,3,1)
        gt_cores = gt_cores.permute(0,2,3,1)
        pred_cloth = pred_cloth.permute(0,2,3,1)
        gt_abscloth = gt_abscloth.permute(0,2,3,1)+pred_cores.permute(0,2,1,3)
        # Compute loss on cloth
        loss_cloth_g = self.cloth_loss(pred_cores.reshape(batch_size, num_fps, dim), 
            gt_cores.reshape(batch_size, -1, dim))
        loss_cloth_l = self.cloth_loss(pred_cloth.reshape(batch_size*num_fps, num_sample, dim), 
            gt_abscloth.reshape(batch_size*num_fps, num_sample, dim))
        loss_cloth_l = loss_cloth_l + self.cloth_loss(pred_cloth.reshape(batch_size, num_fps*num_sample, dim), 
            gt_abscloth.reshape(batch_size, -1, dim))
        #only for testing!
        if not is_train:
            loss_cloth_l = self.cloth_loss(pred_cloth.reshape(batch_size, num_fps*num_sample, dim), 
                gt_abscloth0.reshape(batch_size, -1, dim))
        loss_cloth0 = loss_cloth_g + loss_cloth_l
        latent_g = 0.1*self.criterion_regr(gt_g, z_g)
        latent_l = 0.1*self.criterion_regr(gt_l, z_l)
        loss_cloth_p = loss_cloth_g + loss_cloth_l 
        loss_latent = (latent_g + latent_l)
        loss_cloth = loss_cloth_p + loss_latent
        loss_cloth_geom = (pred_cores.reshape(batch_size, num_fps, dim)-gt_cores.reshape(batch_size, -1, dim)).norm(dim=2).mean() #self.cloth_loss_g(pred_cloth.reshape(batch_size, num_fps*num_sample, dim), gt_abscloth0)

        # Compute loss on SMPL parameters
        loss_regr_pose, loss_regr_betas = self.smpl_losses(pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl)

        # Compute 2D reprojection loss for the keypoints
        loss_keypoints = self.keypoint_loss(pred_keypoints_2d, gt_keypoints_2d,
                                            self.options.openpose_train_weight,
                                            self.options.gt_train_weight)

        # Compute 3D keypoint loss
        loss_keypoints_3d = (pred_joints-gt_model_joints).norm(dim=2).mean() # self.keypoint_3d_loss(pred_joints, gt_joints, has_pose_3d)

        # Per-vertex loss for the shape
        loss_shape = self.shape_loss(pred_vertices, gt_vertices, has_smpl)

        # Compute total loss
        # The last component is a loss that forces the network to predict positive depth values
        loss = self.options.shape_loss_weight * loss_shape +\
               self.options.keypoint_loss_weight * loss_keypoints +\
               self.options.keypoint_loss_weight * loss_keypoints_3d +\
               self.options.pose_loss_weight * loss_regr_pose + \
               self.options.beta_loss_weight * loss_regr_betas +\
               ((torch.exp(-pred_camera[:,0]*10)) ** 2 ).mean() +\
               self.options.cloth_loss_weight * loss_cloth
        # loss = self.options.cloth_loss_weight * loss_cloth
        loss *= 60


        # Do backprop
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Pack output arguments for tensorboard logging
        output = {'pred_vertices': pred_vertices.detach(),
                  'opt_vertices': gt_vertices,
                  'opt_joints': gt_keypoints_2d_orig,
                  'pred_kp': 0.5 * self.options.img_res * (pred_keypoints_2d.detach()+1),
                  'pred_cam_t': pred_cam_t.detach(),
                  'pred_cloth': pred_cloth.detach(),
                  'gt_cloth': gt_abscloth0.detach(),
                  'opt_cam_t': gt_cam_t}
        losses = {'losses/total': loss.detach().item(),
                  'losses/keypoints': loss_keypoints.detach().item(),
                  'losses/cloth': loss_cloth0.detach().item(),
                  'losses/cloth_g': loss_cloth_g.detach().item(),
                  'losses/cloth_l': loss_cloth_l.detach().item(),
                  'losses/cloth_geom': loss_cloth_geom.detach().item(),
                  'losses/latent': loss_latent.detach().item(),
                  'losses/latent_g': latent_g.detach().item(),
                  'losses/latent_l': latent_l.detach().item(),
                  'losses/keypoints_3d': loss_keypoints_3d.detach().item(),
                  'losses/regr_pose': loss_regr_pose.detach().item(),
                  'losses/regr_betas': loss_regr_betas.detach().item(),
                  'losses/shape': loss_shape.detach().item()}

        return output, losses

    def train_summaries(self, input_batch, output, losses, out_img=False, is_train=True):
        images = input_batch['img']
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)

        pred_vertices = output['pred_vertices']
        opt_vertices = output['opt_vertices']
        pred_cam_t = output['pred_cam_t']
        opt_cam_t = output['opt_cam_t']
        pred_cloth = output['pred_cloth']
        gt_cloth = output['gt_cloth']
        gt_kp = output['opt_joints']
        pred_kp = output['pred_kp']
        for loss_name, val in losses.items():
            if not is_train:
                loss_name = 'test_'+loss_name
            self.summary_writer.add_scalar(loss_name, val, self.step_count)
        self.summary_writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], self.step_count)
        # if losses['losses/cloth_g'] > 1:
        #     import sys;sys.exit(0)
        # if not out_img:
        #     return
        # images_pred = self.renderer.visualize_tb(pred_vertices, pred_cam_t, images, pred_kp)
        # images_opt = self.renderer.visualize_tb(opt_vertices, opt_cam_t, images, gt_kp)
        # self.summary_writer.add_image('pred_shape', images_pred, self.step_count)
        # self.summary_writer.add_image('gt_shape', images_opt, self.step_count)
        # if losses['losses/cloth'] < 1.5:
        #     return
        # for i in range(gt_cloth.shape[0]):
        #     print(input_batch['dataname'][i])
            # tmp = gt_cloth[i]-pred_cloth[i]
            # print(tmp.min(), tmp.max(), tmp.norm())
        with open('{}/{}_gt_cloth.obj'.format(self.options.summary_dir,self.step_count), 'w') as f:
            for p in gt_cloth[0].reshape(-1,3).cpu().numpy():
                f.write('v {} {} {}\n'.format(p[0],p[1],p[2]))
        with open('{}/{}_ours_cloth.obj'.format(self.options.summary_dir,self.step_count), 'w') as f:
            for p in pred_cloth[0].reshape(-1,3).cpu().numpy():
                f.write('v {} {} {}\n'.format(p[0],p[1],p[2]))
        with open('{}/{}_ours_body.obj'.format(self.options.summary_dir,self.step_count), 'w') as f:
            for p in pred_vertices[0].reshape(-1,3).cpu().numpy():
                f.write('v {} {} {}\n'.format(p[0],p[1],p[2]))
        with open('{}/{}_gt_body.obj'.format(self.options.summary_dir,self.step_count), 'w') as f:
            for p in opt_vertices[0].reshape(-1,3).cpu().numpy():
                f.write('v {} {} {}\n'.format(p[0],p[1],p[2]))
        import sys
        sys.exit(0)
