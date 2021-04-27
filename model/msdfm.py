
import torch
import torch.nn as nn

from .utils import cSE, Spacial_Relation_Module, conv2DBatchNormRelu, DUpsampling, inception_module_3


class MSDFM(nn.Module):
    def __init__(self, args):
        super(MSDFM, self).__init__()

        self.top_encoder = Encoder()
        self.dsm_encoder = Encoder()

        self.cSE_5_t = cSE(in_ch=2048)
        self.spacial_feature_5_t = Spacial_Relation_Module(in_ch=2048, h=16, w=16)
        self.cSE_5_d = cSE(in_ch=2048)
        self.spacial_feature_5_d = Spacial_Relation_Module(in_ch=2048, h=16, w=16)

        self.cSE_4_t = cSE(in_ch=1024)
        self.spacial_feature_4_t = Spacial_Relation_Module(in_ch=1024, h=16, w=16)
        self.cSE_4_d = cSE(in_ch=1024)
        self.spacial_feature_4_d = Spacial_Relation_Module(in_ch=1024, h=16, w=16)

        self.cSE_3_t = cSE(in_ch=512)
        self.spacial_feature_3_t = Spacial_Relation_Module(in_ch=512, h=32, w=32)
        self.cSE_3_d = cSE(in_ch=512)
        self.spacial_feature_3_d = Spacial_Relation_Module(in_ch=512, h=32, w=32)

        self.conv2_1 = conv2DBatchNormRelu(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv2_2 = conv2DBatchNormRelu(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.SP_x5 = DUpsampling(inplanes=2304, outplanes=144, scale=4)
        self.SP_dx5 = DUpsampling(inplanes=2304, outplanes=144, scale=4)

        self.SP_x4 = DUpsampling(inplanes=1280, outplanes=80, scale=4)
        self.SP_dx4 = DUpsampling(inplanes=1280, outplanes=80, scale=4)

        self.SP_x3 = DUpsampling(inplanes=1536, outplanes=384, scale=2)
        self.SP_dx3 = DUpsampling(inplanes=1536, outplanes=384, scale=2)

        self.IP = inception_module_3(in_ch=1216)

        self.conv4_1 = conv2DBatchNormRelu(in_channels=768, out_channels=256, kernel_size=3, stride=1, padding=1)

        if args.dataset == "Vaihingen":
            self.conv4_2 = conv2DBatchNormRelu(in_channels=256, out_channels=80, kernel_size=3, stride=1, padding=1)
            self.SP = DUpsampling(inplanes=80, outplanes=5, scale=4, residual=True)
        elif args.dataset == "Potsdam":
            self.conv4_2 = conv2DBatchNormRelu(in_channels=256, out_channels=96, kernel_size=3, stride=1, padding=1)
            self.SP = DUpsampling(inplanes=96, outplanes=6, scale=4, residual=True)

    def forward(self, x, dsm_x):
        x1, x2, x3, x4, x5 = self.top_encoder(x)

        dx1, dx2, dx3, dx4, dx5 = self.dsm_encoder(x)

        cx2 = self.conv2_2(self.conv2_1(torch.cat([x2, dx2], dim=1)))

        x5 = self.SP_x5(self.spacial_feature_5_t(self.cSE_5_t(x5)))
        x4 = self.SP_x4(self.spacial_feature_4_t(self.cSE_4_t(x4)))
        x3 = self.SP_x3(self.spacial_feature_3_t(self.cSE_3_t(x3)))
        dx5 = self.SP_dx5(self.spacial_feature_5_d(self.cSE_5_t(dx5)))
        dx4 = self.SP_dx4(self.spacial_feature_4_d(self.cSE_4_t(dx4)))
        dx3 = self.SP_dx3(self.spacial_feature_3_d(self.cSE_3_t(dx3)))

        out = torch.cat([x5, dx5, x4, dx4, x3, dx3], dim=1)

        out = self.IP(out)

        out = self.conv4_2(self.conv4_1(torch.cat([cx2, out], dim=1)))

        out = self.SP(out)

        return out



affine_par = True


def load_pretrained_model(net, state_dict, strict=True):
    """Copies parameters and buffers from :attr:`state_dict` into
    this module and its descendants. If :attr:`strict` is ``True`` then
    the keys of :attr:`state_dict` must exactly match the keys returned
    by this module's :func:`state_dict()` function.
    Arguments:
        state_dict (dict): A dict containing parameters and
            persistent buffers.
        strict (bool): Strictly enforce that the keys in :attr:`state_dict`
            match the keys returned by this module's `:func:`state_dict()`
            function.
    """
    own_state = net.state_dict()
    # print state_dict.keys()
    # print own_state.keys()
    for name, param in state_dict.items():
        if name in own_state:
            # print name, np.mean(param.numpy())
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            if strict:
                try:
                    own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError('While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, own_state[name].size(), param.size()))
            else:
                try:
                    own_state[name].copy_(param)
                except Exception:
                    print('Ignoring Error: While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, own_state[name].size(), param.size()))

        elif strict:
            raise KeyError('unexpected key "{}" in state_dict'
                           .format(name))
    if strict:
        missing = set(own_state.keys()) - set(state_dict.keys())
        if len(missing) > 0:
            raise KeyError('missing keys in state_dict: "{}"'.format(missing))


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                            padding=dilation * multi_grid, dilation=dilation * multi_grid, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
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

        out = out + residual
        out = self.relu_inplace(out)

        return out


class Encoder(nn.Module):
    def __init__(self, pretrain=True, model_path='/home/sugarchl1/Segv2/data/resnet101-5d3b4d8f.pth'):
        super(Encoder, self).__init__()
        self.model = ResNet(Bottleneck, [3, 4, 23, 3])
        if pretrain:
            load_pretrained_model(self.model, torch.load(model_path), strict=False)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.model(x)
        return x1, x2, x3, x4, x5


class ResNet(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)

        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2, multi_grid=(1, 1, 1))

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                                multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x):
        # input 528 * 528
        x = self.relu1(self.bn1(self.conv1(x)))  # 264 * 264

        x1 = self.maxpool(x)  # 66 * 66
        x2 = self.layer1(x1)  # 66 * 66
        x3 = self.layer2(x2)  # 33 * 33
        x4 = self.layer3(x3)  # 66 * 66
        x5 = self.layer4(x4)  # 33 * 33

        return x1, x2, x3, x4, x5
