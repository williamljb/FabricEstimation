import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
import numpy as np
import math
import sys
from utils.geometry import rot6d_to_rotmat
from .smpl import SMPL
import trimesh
import pickle as pkl
import constants
import time

class Bottleneck(nn.Module):
    """ Redefinition of Bottleneck residual block
        Adapted from the official PyTorch implementation
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class HMR(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """

    def __init__(self, block, layers, smpl_mean_params, smpl):
        self.inplanes = 64
        super(HMR, self).__init__()
        npose = 24 * 6
        self.smpl = smpl
        self.dim_cloth = 512+8*256
        p = pkl.load(open(constants.SAMPLE_FILE, 'rb'))
        self.pvts = torch.tensor(p['vertices']).long()
        self.pws = torch.tensor(p['weights']).float().unsqueeze(1)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(512 * block.expansion + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        # self.fc_cloth00 = nn.Linear(npose + 13, 1024)
        # self.fc_cloth01 = nn.Linear(512 * block.expansion, 1024)
        self.fc_cloth00 = nn.Linear(npose + 13, 512 * block.expansion)
        self.fc_cloth01 = nn.Linear(512 * block.expansion, self.dim_cloth)
        self.fc_cloth02 = nn.Linear(self.dim_cloth, 1024)
        self.fc_cloth1 = nn.Linear(1024, 1024)
        self.drop3 = nn.Dropout()
        self.decpose1 = nn.Linear(1024, npose)
        self.decshape1 = nn.Linear(1024, 10)
        self.deccam1 = nn.Linear(1024, 3)
        self.deccloth = nn.Linear(1024, self.dim_cloth)
        # self.final_conv = nn.Sequential(nn.ConvTranspose2d(64, 4, kernel_size=1, stride=1),nn.Tanh())
        nn.init.xavier_uniform_(self.decpose1.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape1.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam1.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccloth.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)
        self.ori_m = ['conv1','bn1','relu','maxpool','layer1','layer2','layer3','layer4','avgpool',
            'fc1','drop1','fc2','drop2','decpose','decshape','deccam']
        self.new_m = ['fc_cloth00','fc_cloth01','fc_cloth02','fc_cloth1','drop3','decpose1','decshape1','deccam1','deccloth']

    def ori_param(self):
        ans = []
        for p,m in self.named_children():
            if p in self.ori_m:
                ans += m.parameters()
        return ans

    def new_param(self):
        ans = []
        for p,m in self.named_children():
            if p in self.new_m:
                ans += m.parameters()
        return ans


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def get_body_pts(self, pred_pose, batch_size, pred_shape):
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)
        flip_vertices = self.smpl(betas=pred_shape, body_pose=pred_rotmat[:,3:], global_orient=pred_rotmat[:,:3], pose2rot=False).vertices
        flip_vertices[:,:,1:3] = -flip_vertices[:,:,1:3]
        body_pts = flip_vertices[:,constants.VALID_PTS0,:].reshape([batch_size,-1,3])
        vts = torch.arange(batch_size).long().reshape([-1,1,1]).repeat(1,256*256,3)
        tmp = body_pts[vts,self.pvts.repeat(batch_size,1,1)]
        fin = torch.matmul(self.pws.cuda(), tmp)
        body_pts = fin.reshape([batch_size,256,256,3]).permute(0,3,1,2)
        return body_pts.roll(64,3)


    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, init_cloth=None, n_iter=3, output_xf=False, output_pose=False):

        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)
        if init_cloth is None:
            init_cloth = torch.zeros([batch_size,self.dim_cloth]).cuda()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        xf = self.avgpool(x4)
        xf = xf.view(xf.size(0), -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        pred_cloth = init_cloth
        pred_cloth[:,512:] = pred_cloth[:,512:] / 10
        for i in range(n_iter):
            xc = torch.cat([xf, pred_pose, pred_shape, pred_cam],1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        for i in range(n_iter):
            xc = torch.cat([pred_pose, pred_shape, pred_cam],1)
            # xc = self.lrelu(self.fc_cloth00(xc) +self.fc_cloth01(xf) +self.fc_cloth02(pred_cloth))
            xc = self.fc_cloth00(xc) + xf
            xc = self.fc_cloth01(xc) + pred_cloth
            xc = self.lrelu(self.fc_cloth02(xc))
            xc = self.drop3(xc)
            xc = self.lrelu(self.fc_cloth1(xc))
            xc = self.drop3(xc)
            pred_pose = self.decpose1(xc) + pred_pose
            pred_shape = self.decshape1(xc) + pred_shape
            pred_cam = self.deccam1(xc) + pred_cam
            pred_cloth = self.deccloth(xc) + pred_cloth
        
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)
        # pred_rotmat[:,0,1:3,:] = -pred_rotmat[:,0,1:3,:]
        pred_cloth[:,512:] = pred_cloth[:,512:] * 10

        if output_xf:
            return pred_rotmat, pred_shape, pred_cam, pred_cloth, xf
        elif output_pose:
            return pred_rotmat, pred_shape, pred_cam, pred_cloth, pred_pose
        else:
            return pred_rotmat, pred_shape, pred_cam, pred_cloth

def hmr(smpl_mean_params, pretrained=True, **kwargs):
    """ Constructs an HMR model with ResNet50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = HMR(Bottleneck, [3, 4, 6, 3],  smpl_mean_params, **kwargs)
    if pretrained:
        resnet_imagenet = resnet.resnet50(pretrained=True)
        model.load_state_dict(resnet_imagenet.state_dict(),strict=False)
    return model

