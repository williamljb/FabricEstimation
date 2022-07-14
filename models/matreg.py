import torch
import torch.nn as nn
import torchvision.models.resnet as resnet

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

class Rescnn(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(Rescnn, self).__init__()
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

    def forward(self, x):

        batch_size = x.shape[0]


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
        return xf

def rescnn():
    """ Constructs an HMR model with ResNet50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Rescnn(Bottleneck, [3, 4, 6, 3])
    resnet_imagenet = resnet.resnet50(pretrained=True)
    model.load_state_dict(resnet_imagenet.state_dict(),strict=False)
    return model

class tcn(nn.Module):
    def __init__(self, in_f):
        super(tcn, self).__init__()
        self.num_c = 4
        nf = [in_f, 1024, 1024, 1024, 1024]
        self.cns = nn.ModuleList([nn.Conv1d(nf[i], nf[i+1], kernel_size=2, stride=2) for i in range(self.num_c)])
        self.fc = nn.Linear(1024,1024)
        self.activation = nn.ReLU()
    def forward(self, x):
        #x: bs*time*in_f
        x=x.permute(0,2,1)
        for i in range(self.num_c):
            x = self.activation(self.cns[i](x))
        return self.fc(x.reshape(x.shape[0],x.shape[1]))

class MATREG(nn.Module):

    def __init__(self, feature_mode):
        super(MATREG, self).__init__()
        self.num_imgf = 2048
        self.num_garf = 512+8*256
        self.hidden = 1024
        self.num_str = 6
        self.num_ben = 9
        self.mode = feature_mode
        if self.mode == 'std':
            input_size = self.num_imgf+self.num_garf
        elif self.mode == 'garf':
            input_size = self.num_garf
        elif self.mode == 'imgf':
            input_size = self.num_imgf
        else:
            print('???')
            import sys;sys.exit(0)
        self.temporal = nn.LSTM(input_size=input_size,
            hidden_size=self.hidden, batch_first=True)
        self.tcn = tcn(self.hidden)
        # self.cnn = rescnn()
        self.num_fc = 0
        self.fcs = nn.ModuleList([nn.Linear(self.hidden, self.hidden) for i in range(self.num_fc)])
        self.fcn0 = nn.Linear(self.hidden, self.num_str)
        self.fcn1 = nn.Linear(self.hidden, self.num_ben)
        self.fcn2 = nn.Linear(self.hidden, 1)
        self.activation = nn.ReLU()

    def forward(self, imgf, garf):
        #imgf: bs*time*2048
        #garf: bs*time*2560
        #logits: bs*num
    # def forward(self, x):
    #     bs = x.shape[0]
    #     leng = x.shape[1]
    #     imgf = self.cnn(x.reshape(-1,x.shape[2],x.shape[3],x.shape[4])).reshape(bs,leng,self.num_imgf)
        self.temporal.flatten_parameters()
        #        tem_out,_ = self.temporal(imgf)
        if self.mode == 'std':
            x = torch.cat([imgf, garf], dim=2)
        elif self.mode == 'garf':
            x = garf
        else:
            x = imgf
        # x = torch.cat([x[:,0:1],x[:,1:]-x[:,:-1]], dim=1)
        x,_ = self.temporal(x)
        x = x[:,-1,:]
        # x = self.tcn(x)
        for i in range(self.num_fc):
            x = self.activation(self.fcs[i](x))
        str_logit = self.fcn0(x)
        ben_logit = self.fcn1(x)
        density = self.fcn2(x)
        return str_logit, ben_logit, density
