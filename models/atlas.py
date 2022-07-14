import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.geometry import sample_patches

class Mapping2Dto3D(nn.Module):
    """
    Core Atlasnet Function.
    Takes batched points as input and run them through an MLP.
    Note : the MLP is implemented as a torch.nn.Conv1d with kernels of size 1 for speed.
    Note : The latent vector is added as a bias after the first layer. Note that this is strictly identical
    as concatenating each input point with the latent vector but saves memory and speeed.
    Author : Thibault Groueix 01.11.2019
    input : 1*3*1*nsample, bs*latent_size*nfps*1
    output :bs*3*nfps*nsample
    """

    def __init__(self, latent_size, dim, hidden):
        self.bottleneck_size = latent_size
        self.dim_output = dim
        self.hidden_neurons = hidden
        self.num_layers = 4
        super(Mapping2Dto3D, self).__init__()
        print(
            f"New MLP decoder : hidden size {self.hidden_neurons}, num_layers {self.num_layers}")

        self.conv10 = torch.nn.Conv2d(dim, self.bottleneck_size, 1)
        self.conv11 = torch.nn.Conv2d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv2d(self.bottleneck_size, self.hidden_neurons, 1)

        self.conv_list = nn.ModuleList(
            [torch.nn.Conv2d(self.hidden_neurons, self.hidden_neurons, 1) for i in range(self.num_layers)])

        self.last_conv = torch.nn.Conv2d(self.hidden_neurons, self.dim_output, 1)

        #self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        #self.bn2 = torch.nn.BatchNorm1d(self.hidden_neurons)

        #self.bn_list = nn.ModuleList([torch.nn.BatchNorm1d(self.hidden_neurons) for i in range(self.num_layers)])

        self.activation = nn.LeakyReLU()
        self.th = nn.Tanh()

    def forward(self, x, latent):
        # print('2d23d:',x.shape,latent.shape)
        # x = torch.cat([x.repeat(latent.shape[0],1,latent.shape[2],1), 
            # latent.repeat(1,1,1,x.shape[3])], dim=1) # bs*(3+latent_size)*nfps*nsample
        x = self.activation(self.conv10(x)+self.conv11(latent))
        x = self.activation((self.conv2(x)))
        for i in range(self.num_layers):
            x = self.activation((self.conv_list[i](x)))
        ret = 2*self.th(self.last_conv(x))
        # print('2d23d1:',ret.shape)
        return ret

class Atlasnet(nn.Module):

    def __init__(self, np, nr, nf=1024, dim=3, hidden=512):
        """
        Core Atlasnet module : decoder to meshes and pointclouds.
        This network takes an embedding in the form of a latent vector and returns a pointcloud or a mesh
        Author : Thibault Groueix 01.11.2019
        input: bs*nf*nfps
        output:bs*3*nfps*nsample
        """
        super(Atlasnet, self).__init__()

        # Define number of points per primitives
        self.nb_primitives = nr
        self.nb_pts_in_primitive = np // self.nb_primitives
        self.np = np
        self.dim = dim

        # Intialize deformation networks
        self.decoder = nn.ModuleList([Mapping2Dto3D(nf, dim, hidden) for i in range(0, self.nb_primitives)])

        self.grid = []
        for patchIndex in range(self.nb_primitives):
            patch = torch.nn.Parameter(torch.FloatTensor(1,dim,1,self.nb_pts_in_primitive))
            patch.data.uniform_(-1,1)
            self.register_parameter("patch%d"%patchIndex,patch)
            self.grid.append(patch)

    def forward(self, latent_vector, train=True):
        """
        Deform points from self.template using the embedding latent_vector
        :param latent_vector: an bottleneck size vector encoding a 3D shape or an image. size : batch, bottleneck
        :return: A deformed pointcloud os size : batch, nb_prim, num_point, 3
        """
        # Deform each patch
        # print('atlasnet:',latent_vector.shape)
        output_points = torch.cat([self.decoder[i](self.grid[i].cuda(), latent_vector.unsqueeze(3)).unsqueeze(1) for i in range(0, self.nb_primitives)], dim=1)

        nfps = latent_vector.shape[2]
        # Return the deformed pointcloud
        ret = output_points.permute(0,2,3,1,4).reshape([-1,self.dim,nfps,self.np]).contiguous()
        # print('atlasnet1:',ret.shape)
        return ret

class PointNet(nn.Module):
    def __init__(self, nlatent=1024, nf=3, nc=3):
        """
        PointNet Encoder
        See : PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
                Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas
        input : bs*(nf,nc)*nfps*nsample
        output : bs*nlatent*nfps
        """

        super(PointNet, self).__init__()
        self.nf = nf
        self.conv00 = torch.nn.Conv2d(nf, nc, 1)
        self.conv01 = torch.nn.Conv2d(nc, nc, 1)
        self.conv1 = torch.nn.Conv2d(nc, 256, 1)
        self.conv2 = torch.nn.Conv2d(256, 256, 1)
        self.conv3 = torch.nn.Conv2d(256, nlatent, 1)
        self.conv4 = torch.nn.Conv2d(nlatent, nlatent, 1)
        self.lin1 = torch.nn.Conv1d(nlatent, nlatent, 1)
        self.lin2 = torch.nn.Conv1d(nlatent, nlatent, 1)
        self.lin3 = torch.nn.Conv1d(nlatent, nlatent, 1)
        self.nlatent = nlatent

    def forward(self, x, cond):
        # x = x[:,:self.nf]
        # print('pointnet:',x.shape)
        x = F.leaky_relu(self.conv00(x) + self.conv01(cond))
        x = F.leaky_relu((self.conv1(x)))
        x = F.leaky_relu((self.conv2(x)))
        x = F.leaky_relu((self.conv3(x)))
        x = (self.conv4(x))
        x, _ = torch.max(x, 3) # bs*nlatent*nfps
        x = F.leaky_relu((self.lin1(x)))
        x = F.leaky_relu((self.lin2(x)))
        x = (self.lin3(x))
        # print('pointnet1:',x.shape)
        return x

class Atlas(nn.Module):
    """
    Wrapper for a encoder and a decoder.
    Author : Thibault Groueix 01.11.2019
    """

    def __init__(self, nfps, nsample, nr, dim):
        super(Atlas, self).__init__()
        latent_g = 512
        latent_l = 8
        latent_body = 226-9
        self.dim = dim
        self.en_g = PointNet(nlatent=latent_g, nf=dim, nc=latent_body)
        self.en_l = PointNet(nlatent=latent_l, nf=dim, nc=dim)

        self.de_g = Atlasnet(nfps, nr, nf=latent_body+latent_g, dim=dim, hidden=512)
        self.de_l = Atlasnet(nsample, nr, nf=latent_g+latent_l, dim=dim, hidden=256)
        self.nsample = nsample
        self.nfps = nfps

        self.apply(weights_init)

    def decode(self, z_g, z_l, body):
        # z_g: bs*512*1
        # z_l: bs*512*nfps
        rec_g = self.de_g(torch.cat([z_g,body.unsqueeze(2)],dim=1)) # bs*3*1*nfps
        rec_l = self.de_l(torch.cat([z_g.repeat(1,1,self.nfps),z_l],dim=1)) # bs*3*nfps*nsample
        rec = rec_l + rec_g.permute(0,1,3,2) # bs*3*nfps*nsample
        return rec.reshape(z_g.shape[0],3,self.nfps,self.nsample), rec_g

    def forward(self, gt_abscloth, cores, body):
        # gt_abscloth: bs* nfps*nsample *3
        # cores: bs*3*1*nfps
        # body: bs*226
        # print('atlas:',pts.shape,cores.shape,body.shape)
        # bdy_rep = body.unsqueeze(2).repeat(1,1,self.nfps).unsqueeze(2) # bs*226*1*nfps
        z_g = self.en_g(gt_abscloth.permute(0,2,1).unsqueeze(2).contiguous(),body.reshape(body.shape[0],body.shape[1],1,1)) # bs*512*1
        # z_g = self.en_g(cores,body.reshape(body.shape[0],body.shape[1],1,1)) # bs*512*1
        rec_g = self.de_g(torch.cat([z_g,body.unsqueeze(2)],dim=1)) # bs*3*1*nfps
        # return None, None, rec_g, z_g, None
        pts = sample_patches(gt_abscloth, rec_g.detach(), gt_abscloth.shape[0], self.nfps, self.nsample)

        cores_l = rec_g.permute(0,1,3,2).repeat(1,1,1,self.nsample) # bs*3*nfps*nsample
        z_l = self.en_l(pts, cores_l) # bs*8*nfps
        rec_l = self.de_l(torch.cat([z_g.repeat(1,1,self.nfps),z_l],dim=1)) # bs*3*nfps*nsample
        rec = rec_l + rec_g.permute(0,1,3,2) # bs*3*nfps*nsample
        return pts, rec, rec_g, z_g, z_l
        # rec_g = rec_l[:,:,:,:1]
        # rec = rec_l[:,:,:,1:] + rec_g
        #print(p)
        # return rec, rec_g.permute(0,1,3,2), z_g, z_l

class Baseline(nn.Module):
    """
    Wrapper for a encoder and a decoder.
    Author : Thibault Groueix 01.11.2019
    """

    def __init__(self, nfps, nsample, nr, dim):
        super(Baseline, self).__init__()
        latent_g = 512
        latent_l = 8
        latent_body = 226-9
        self.dim = dim
        self.en = PointNet(nlatent=latent_g+latent_l*nfps, nf=dim, nc=latent_body)

        self.de = Atlasnet(nfps*nsample, nfps, nf=latent_body+latent_g+latent_l*nfps, dim=dim, hidden=512)
        self.nsample = nsample
        self.nfps = nfps

        self.apply(weights_init)

    def forward(self, gt_abscloth, cores, body):
        # gt_abscloth: bs* nfps*nsample *3
        # cores: bs*3*1*nfps
        # body: bs*226
        # print('atlas:',pts.shape,cores.shape,body.shape)
        # bdy_rep = body.unsqueeze(2).repeat(1,1,self.nfps).unsqueeze(2) # bs*226*1*nfps
        z = self.en(gt_abscloth.permute(0,2,1).unsqueeze(2).contiguous(),body.reshape(body.shape[0],body.shape[1],1,1)) # bs*512*1
        # z_g = self.en_g(cores,body.reshape(body.shape[0],body.shape[1],1,1)) # bs*512*1
        rec = self.de(torch.cat([z,body.unsqueeze(2)],dim=1)) # bs*3*1*nfps
        return rec, z

class AtlasEncoder(nn.Module):
    """
    Wrapper for a encoder and a decoder.
    Author : Thibault Groueix 01.11.2019
    """

    def __init__(self, nfps, nsample, nr, dim):
        super(AtlasEncoder, self).__init__()
        latent_g = 512
        latent_l = 8
        latent_body = 226-9
        self.dim = dim
        self.en_g = PointNet(nlatent=latent_g, nf=dim, nc=latent_body)
        self.en_l = PointNet(nlatent=latent_l, nf=dim, nc=dim)
        self.de_g = Atlasnet(nfps, nr, nf=latent_body+latent_g, dim=dim, hidden=512)
        self.nsample = nsample
        self.nfps = nfps

        self.apply(weights_init)

    def forward(self, gt_abscloth, body, train=True):
        z_g = self.en_g(gt_abscloth.permute(0,2,1).unsqueeze(2).contiguous(),body.reshape(body.shape[0],body.shape[1],1,1)) # bs*512*1
        rec_g = self.de_g(torch.cat([z_g,body.unsqueeze(2)],dim=1)) # bs*3*1*nfps
        pts = sample_patches(gt_abscloth, rec_g.detach(), gt_abscloth.shape[0], self.nfps, self.nsample)
        cores_l = rec_g.permute(0,1,3,2).repeat(1,1,1,self.nsample) # bs*3*nfps*nsample
        z_l = self.en_l(pts, cores_l) # bs*8*nfps
        return pts, z_g, z_l

class AtlasDecoder(nn.Module):
    """
    Wrapper for a encoder and a decoder.
    Author : Thibault Groueix 01.11.2019
    """

    def __init__(self, nfps, nsample, nr, dim):
        super(AtlasDecoder, self).__init__()
        latent_g = 512
        latent_l = 8
        latent_body = 226-9
        self.dim = dim
        self.de_g = Atlasnet(nfps, nr, nf=latent_body+latent_g, dim=dim, hidden=512)
        self.de_l = Atlasnet(nsample, nr, nf=latent_g+latent_l, dim=dim, hidden=256)
        self.nsample = nsample
        self.nfps = nfps

        self.apply(weights_init)

    def forward(self, z_g, z_l, body):
        # z_g: bs*512*1
        # z_l: bs*512*nfps
        rec_g = self.de_g(torch.cat([z_g,body.unsqueeze(2)],dim=1)) # bs*3*1*nfps
        rec_l = self.de_l(torch.cat([z_g.repeat(1,1,self.nfps),z_l],dim=1)) # bs*3*nfps*nsample
        rec = rec_l + rec_g.permute(0,1,3,2) # bs*3*nfps*nsample
        return rec.reshape(z_g.shape[0],3,self.nfps*self.nsample)
        # rec_l = self.de_l(torch.cat([z_g.repeat(1,1,self.nfps),z_l],dim=1)) # bs*3*nfps*nsample
        # rec_g = rec_l[:,:,:,:1]
        # rec = rec_l[:,:,:,1:] + rec_g
        # return rec.reshape(z_g.shape[0],3,self.nfps*self.nsample)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
