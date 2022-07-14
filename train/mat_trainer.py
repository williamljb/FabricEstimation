import torch
import torch.nn as nn
import numpy as np

from datasets import MatDataset
from models import hmr, SMPL, MATREG
from utils.renderer import Renderer
from utils import BaseTrainer
from utils.geometry import batch_rodrigues, perspective_projection, estimate_translation, sample_cores, align

import config
import constants
from .fits_dict import FitsDict
import time
import pickle as pkl

from pytorch3d.loss import chamfer_distance

from torchvision import models

class MatTrainer(BaseTrainer):

    def load_hmr(self, checkpoint_file):
        state_dict = torch.load(checkpoint_file)['model']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.smpl'):
                continue
            new_state_dict[k] = v
        # load params
        self.coder.load_state_dict(new_state_dict, strict=False)
        print('Decoder loaded')
    
    def init_fn(self):
        self.train_ds = MatDataset(self.options, dataset='synthetic_M_H01', ignore_3d=self.options.ignore_3d, is_train=True)
        self.test_ds = MatDataset(self.options, dataset='synthetic_M_1_H01', ignore_3d=self.options.ignore_3d, is_train=False)

        self.smpl = SMPL(config.SMPL_MODEL_DIR,
                         batch_size=self.options.batch_size*25,
                         create_transl=False).to(self.device)
        self.model = MATREG(self.options.feature_mode)
        self.model = nn.DataParallel(self.model).to(self.device)

        self.coder = hmr(config.SMPL_MEAN_PARAMS, pretrained=True, smpl=self.smpl)
        self.coder = nn.DataParallel(self.coder).to(self.device)
        self.coder.eval()
        self.load_hmr(self.options.pretrained_hmr)

        self.optimizer = torch.optim.Adam(self.model.module.parameters(),
                                          lr=self.options.lr,
                                          weight_decay=0)
        # Per-vertex loss on the shape
        self.criterion_mat = nn.CrossEntropyLoss().to(self.device)

        self.models_dict = {'model': self.model}
        self.optimizers_dict = {'optimizer': self.optimizer}
        self.focal_length = constants.FOCAL_LENGTH
        self.pers_rot = torch.FloatTensor([[1.,0,0],[0,-1,0],[0,0,-1]]).to(self.device).unsqueeze(0)

        # Initialize SMPLify fitting module
        if self.options.pretrained_checkpoint is not None:
            self.load_pretrained(checkpoint_file=self.options.pretrained_checkpoint)

        # Create renderer
        self.renderer = Renderer(focal_length=self.focal_length, img_res=self.options.img_res, faces=self.smpl.faces)
        self.l1 = nn.L1Loss()

    def finalize(self):
        pass

    def train_step(self, input_batch, is_train=True):
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        # Get data from the batch
        # images = input_batch['img'] # input image
        garf = input_batch['garf']
        imgf = input_batch['imgf']
        imgname = input_batch['dataname']
        gt_str = input_batch['stretch_mat']
        gt_ben = input_batch['bend_mat']
        gt_den = input_batch['density']
        dataset_name = input_batch['dataset_name'] # name of the dataset the image comes from
        indices = input_batch['sample_index'] # index of example inside its dataset
        batch_size = garf.shape[0]
        length = self.options.seq_len #25
        # print(imgf.shape,garf.shape)

        # with torch.no_grad():
        #     pred_rotmat, pred_betas, pred_camera, garf, imgf = self.coder(images.reshape(batch_size*length,
        #         images.shape[2],images.shape[3],images.shape[4]), output_xf=True)
        # print(imgname)
        # print(garf[0,:10])
        # print(imgf[0,:10])
        # print(images[0,0,:,0,0])
        # import sys;sys.exit(0)

        # Feed images in the network to predict camera and SMPL parameters
        # str_logit, ben_logit = self.model(images)
        str_logit, ben_logit, den = self.model(imgf.reshape(batch_size,length,-1), garf.reshape(batch_size,length,-1))

        # pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
        # pred_vertices = pred_output.vertices
        # pred_joints = pred_output.joints

        # # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
        # # This camera translation can be used in a full perspective projection
        # pred_cam_t = torch.stack([pred_camera[:,1],
        #                           pred_camera[:,2],
        #                           2*self.focal_length/(self.options.img_res * pred_camera[:,0] +1e-9)],dim=-1)

        # camera_center = torch.zeros(batch_size*length, 2, device=self.device)
        # pred_keypoints_2d = perspective_projection(pred_joints,
        #                                            rotation=self.pers_rot.expand(batch_size*length, -1, -1),
        #                                            translation=pred_cam_t,
        #                                            focal_length=self.focal_length,
        #                                            camera_center=camera_center)
        # # Normalize keypoints to [-1,1]
        # pred_keypoints_2d = pred_keypoints_2d / (self.options.img_res / 2.)

        # Compute loss on cloth
        if is_train:
            loss_str = 1*self.criterion_mat(str_logit, gt_str)
            loss_ben = 1*self.criterion_mat(ben_logit, gt_ben)
        else:
            loss_str = 1*(str_logit.max(dim=1)[1] == gt_str).float().mean()
            loss_ben = 1*(ben_logit.max(dim=1)[1] == gt_ben).float().mean()
        loss_den = 0*self.l1(den, gt_den.unsqueeze(1))
        # Compute total loss
        # The last component is a loss that forces the network to predict positive depth values
        if is_train:
            loss = loss_str + loss_ben + loss_den
        else:
            loss = ((str_logit.max(dim=1)[1] == gt_str) & (ben_logit.max(dim=1)[1] == gt_ben)).float().mean()
        # loss = self.options.cloth_loss_weight * loss_cloth
        # loss *= 60


        # Do backprop
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Pack output arguments for tensorboard logging
        output = {
                # 'pred_vertices': pred_vertices.detach(),
                #   'pred_kp': 0.5 * self.options.img_res * (pred_keypoints_2d.detach()+1),
                #   'pred_cam_t': pred_cam_t.detach()
                  }
        losses = {'losses/total': loss.detach().item(),
                  'losses/str': loss_str.detach().item(),
                  'losses/ben': loss_ben.detach().item(),
                  'losses/den': loss_den.detach().item()}

        return output, losses

    def train_summaries(self, input_batch, output, losses, out_img=False, is_train=True):
        # images = input_batch['img'][:,0]
        # images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
        # images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)

        # ind = torch.arange(0, output['pred_vertices'].shape[0], step=self.options.seq_len)
        # pred_vertices = output['pred_vertices'][ind]
        # pred_cam_t = output['pred_cam_t'][ind]
        # pred_kp = output['pred_kp'][ind]
        for loss_name, val in losses.items():
            if not is_train:
                loss_name = 'test_'+loss_name
            self.summary_writer.add_scalar(loss_name, val, self.step_count)
        self.summary_writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], self.step_count)
        if not out_img:
            return
        # images_pred = self.renderer.visualize_tb(pred_vertices, pred_cam_t, images, pred_kp)
        # self.summary_writer.add_image('pred_shape', images_pred, self.step_count)
