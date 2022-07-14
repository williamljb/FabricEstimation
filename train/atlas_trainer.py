import torch
import torch.nn as nn
import numpy as np
from torchgeometry import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis
import cv2

from datasets import AtlasDataset
from models import hmr, SMPL, Atlas
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
from fml.nn import SinkhornLoss
from torch_cluster import fps

class AtlasTrainer(BaseTrainer):
    
    def init_fn(self):
        self.train_ds = AtlasDataset(self.options, dataset='synthetic_CH', ignore_3d=self.options.ignore_3d, is_train=True)
        self.test_ds = AtlasDataset(self.options, dataset='synthetic_CH', ignore_3d=self.options.ignore_3d, is_train=False)

        self.smpl = SMPL(config.SMPL_MODEL_DIR,
                         batch_size=self.options.batch_size,
                         create_transl=False).to(self.device)
        self.model = Atlas(self.options.num_fps,
            self.options.num_sample, self.options.num_patches, self.options.dim)
        self.model = nn.DataParallel(self.model, device_ids=[0,1]).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.module.parameters(),
                                          lr=self.options.lr,
                                          weight_decay=0)
        # Per-vertex loss on the shape
        self.criterion_shape = nn.L1Loss().to(self.device)
        # Keypoint (2D and 3D) loss
        # No reduction because confidence weighting needs to be applied
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        # Loss for SMPL parameter regression
        self.criterion_regr = nn.MSELoss().to(self.device)
        self.criterion_img = nn.MSELoss().to(self.device)
        self.criterion_class = nn.BCEWithLogitsLoss().to(self.device)
        self.models_dict = {'model': self.model}
        self.optimizers_dict = {'optimizer': self.optimizer}
        self.focal_length = constants.FOCAL_LENGTH
        p = pkl.load(open(constants.SAMPLE_FILE, 'rb'))
        self.pvts = torch.tensor(p['vertices']).long()
        self.pws = torch.tensor(p['weights']).float().unsqueeze(1)
        self.geomloss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
        self.sinkhorn = SinkhornLoss()
        # Initialize SMPLify fitting module
        if self.options.pretrained_checkpoint is not None:
            self.load_pretrained(checkpoint_file=self.options.pretrained_checkpoint)

    def finalize(self):
        pass


    def get_body_pts(self, batch_size, vertices):
        flip_vertices = vertices + 0
        body_pts = flip_vertices[:,constants.VALID_PTS0,:].reshape([batch_size,-1,3])
        vts = torch.arange(batch_size).long().reshape([-1,1,1]).repeat(1,256*256,3)
        tmp = body_pts[vts,self.pvts.repeat(batch_size,1,1)]
        fin = torch.matmul(self.pws.cuda(), tmp)
        body_pts = fin.reshape([batch_size,256,256,3]).permute(0,3,1,2)
        return body_pts.roll(64,3)

    def cloth_loss_g(self, pred_cloth, gt_cloth):
        # return chamfer_distance(pred_cloth, gt_cloth)
        # return self.sinkhorn(pred_cloth,gt_cloth)
        # return chamfer_distance(pred_cloth.reshape(-1,1,3), gt_cloth.reshape(-1,1,3))[0]
        loss = 0
        for i in range(pred_cloth.shape[0]):
            loss = loss + self.geomloss(pred_cloth[i], gt_cloth[i])
        return loss / pred_cloth.shape[0]

    def cloth_loss_l(self, pred_cloth, gt_cloth):
        return chamfer_distance(pred_cloth, gt_cloth)[0]

    def train_step(self, input_batch, is_train=True):
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        # Get data from the batch
        gt_pose = input_batch['pose'] # SMPL pose parameters
        gt_betas = input_batch['betas'] # SMPL beta parameters
        gt_cloth = input_batch['map']
        gt_norm = input_batch['normmap']
        rot_angle = input_batch['rot_angle'] # rotation angle used for data augmentation
        dataset_name = input_batch['dataset_name'] # name of the dataset the image comes from
        indices = input_batch['sample_index'] # index of example inside its dataset
        batch_size = gt_cloth.shape[0]

        # Get GT vertices and model joints
        # Note that gt_model_joints is different from gt_joints as it comes from SMPL
        gt_out = self.smpl(betas=gt_betas, body_pose=gt_pose[:,3:], global_orient=gt_pose[:,:3])
        gt_vertices = gt_out.vertices
        if self.options.dim == 3:
            gt_abscloth0 = gt_cloth
        else:
            gt_abscloth0 = torch.cat([gt_cloth, gt_norm], dim=2)
        dim = self.options.dim
        num_sample = self.options.num_sample
        num_fps = self.options.num_fps

        #print(gt_abscloth.shape)
        # Feed images in the network to predict camera and SMPL parameters
        gt_rot = batch_rodrigues(gt_pose.reshape([-1,3])).reshape([batch_size,24*9])
        #print(gt_pose[:,:3])

        #align to global 0 rotation and origin mid-hip
        gt_abscloth0 = align(gt_abscloth0, gt_rot[:,:9].reshape(batch_size, 3, 3), gt_out.joints[:,8:9,:])
        gt_cores = sample_cores(gt_abscloth0, batch_size, self.options.num_fps, num_sample)

        gt_abscloth, pred_cloth, pred_cores, _,_ = self.model(gt_abscloth0,gt_cores,torch.cat([gt_rot[:,9:],gt_betas],dim=1))
        # print('model:',pred_cores.shape,gt_cores.shape)
        pred_cores = pred_cores.permute(0,2,3,1)
        gt_cores = gt_cores.permute(0,2,3,1)
        pred_cloth = pred_cloth.permute(0,2,3,1)
        gt_abscloth = gt_abscloth.permute(0,2,3,1)+pred_cores.permute(0,2,1,3)

        # Compute loss on cloth
        loss_cloth_g = self.cloth_loss_l(pred_cores.reshape(batch_size, num_fps, dim), 
            gt_abscloth0.reshape(batch_size, -1, dim))
        # loss_cloth_g = self.cloth_loss_g(pred_cores.reshape(batch_size, num_fps, dim), 
        #     gt_cores.reshape(batch_size, num_fps, dim))

        #commented only for testing!
        loss_cloth_l = self.cloth_loss_l(pred_cloth.reshape(batch_size*num_fps, num_sample, dim), 
            gt_abscloth.reshape(batch_size*num_fps, num_sample, dim))
        loss_cloth_l = loss_cloth_l + self.cloth_loss_l(pred_cloth.reshape(batch_size, num_fps*num_sample, dim), 
            gt_abscloth.reshape(batch_size, -1, dim))
        # loss_cloth_l = self.cloth_loss_l(pred_cloth.reshape(batch_size, num_fps*num_sample, dim), 
        #     gt_abscloth0.reshape(batch_size, -1, dim))
        # loss_cloth_geom = self.cloth_loss_g(pred_cloth.reshape(batch_size, num_fps*num_sample, dim), gt_abscloth0)
        loss_cloth = loss_cloth_g + loss_cloth_l
        # loss_region = self.region_loss(pred_cloth) * 0

        # Compute total loss
        # The last component is a loss that forces the network to predict positive depth values
        loss = self.options.cloth_loss_weight * loss_cloth
                # self.options.cloth_loss_weight * loss_region
        # loss = self.options.cloth_loss_weight * loss_cloth
        loss *= 60

        # Do backprop
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Pack output arguments for tensorboard logging
        output = {'pred_cloth': pred_cloth,#.detach(),
                  'gt_cloth': gt_abscloth,
                  'pred_cores': pred_cores.detach(),
                  'gt_cores': gt_cores}
        losses = {'losses/total': loss.detach().item(),
                  'losses/cloth_geom': loss_cloth_geom.detach().item(),
                  'losses/cloth_g': loss_cloth_g.detach().item(),
                  'losses/cloth_l': loss_cloth_l.detach().item()
                  }

        return output, losses

    def train_summaries(self, input_batch, output, losses, out_img=False, is_train=True):
        for loss_name, val in losses.items():
            if not is_train:
                loss_name = 'test_'+loss_name
            self.summary_writer.add_scalar(loss_name, val, self.step_count)
        self.summary_writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], self.step_count)
            # if losses['losses/cloth_g'] > 0.01:
            #     print(input_batch['dataname'])
            # for i in range(output['gt_cores'].shape[0]):
            #     with open('{}/{}_gt{}.obj'.format(self.options.summary_dir,self.step_count,i), 'w') as f:
            #         for p in output['gt_cores'][i,0]:
            #             f.write('v {} {} {}\n'.format(p[0],p[1],p[2]))
            #     with open('{}/{}_gtall{}.obj'.format(self.options.summary_dir,self.step_count,i), 'w') as f:
            #         for p in output['gt_cloth'][i].reshape(-1,3):
            #             f.write('v {} {} {}\n'.format(p[0],p[1],p[2]))
            #     with open('{}/{}_ours{}.obj'.format(self.options.summary_dir,self.step_count,i), 'w') as f:
            #         for p in output['pred_cores'][i,0]:
            #             f.write('v {} {} {}\n'.format(p[0],p[1],p[2]))
            #     with open('{}/{}_oursall{}.obj'.format(self.options.summary_dir,self.step_count,i), 'w') as f:
            #         for p in output['pred_cloth'][i].reshape(-1,3):
            #             f.write('v {} {} {}\n'.format(p[0],p[1],p[2]))
            # import sys
            # sys.exit(0)
