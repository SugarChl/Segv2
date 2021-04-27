
import torch
import torch.nn as nn


class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, dilation=1):
        super(conv2DBatchNormRelu, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=bias, dilation=dilation)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class segnetDown2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(segnetDown2, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                         padding=1)
        self.conv2 = conv2DBatchNormRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                         padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.max_pool(x)
        return x


class cSE(nn.Module):
    def __init__(self, in_ch):
        super(cSE, self).__init__()
        self.in_ch = in_ch

        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc2 = nn.Linear(in_features=self.in_ch, out_features=int(self.in_ch/2), bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(in_features=int(self.in_ch/2), out_features=self.in_ch, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        average_pooling_down5 = self.gap(x).squeeze(3).squeeze(2)  # torch.Size([2, 512, 1, 1])

        z = self.fc2(average_pooling_down5)
        z = self.relu(z)
        z = self.fc1(z)
        z = self.sigmoid(z)
        #print(z)
        z = z.reshape([x.shape[0], x.shape[1], 1, 1])
        x = x*z
        return x


class Spacial_Relation_Module(nn.Module):
    def __init__(self, in_ch, h, w, n_classes = 5):
        super(Spacial_Relation_Module, self).__init__()
        self.in_ch = in_ch
        self.h = h
        self.w = w
        self.spacial_u_conv = conv2DBatchNormRelu(in_channels=self.in_ch,
                                                  out_channels=self.in_ch,
                                                  kernel_size=1,
                                                  padding=0,
                                                  stride=1)
        self.spacial_v_conv = conv2DBatchNormRelu(in_channels=self.in_ch,
                                                  out_channels=self.in_ch,
                                                  kernel_size=1,
                                                  padding=0,
                                                  stride=1)

    def forward(self, x):
        batchsize = x.shape[0]
        S_u = self.spacial_u_conv(x).reshape([batchsize, self.in_ch, self.h * self.w]).permute(0, 2, 1)  # torch.Size([1, 512, 256])
        S_v = self.spacial_v_conv(x).reshape([batchsize, self.in_ch, self.h * self.w])  # torch.Size([1, 512, 256])
        spacial_feature = torch.bmm(S_u, S_v)
        spacial_feature = spacial_feature.unsqueeze(0).reshape([batchsize, self.h * self.w, self.h, self.w])
        X_c = torch.cat([spacial_feature, x], dim=1)  # torch.Size([1, 768, 256, 256])


        return X_c


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False, scale=2):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1):
        x1 = self.up(x1)
        x = self.conv(x1)
        return x


class DeconvUpsample2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DeconvUpsample2, self).__init__()

        self.b1 = up(in_ch=in_ch, out_ch=out_ch, bilinear=False)

    def forward(self, x):
        res = self.b1(x)
        return res


class DeconvUpsample4(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DeconvUpsample4, self).__init__()

        self.b1 = up(in_ch=in_ch, out_ch=out_ch, bilinear=False)
        self.b2 = up(in_ch=out_ch, out_ch=out_ch, bilinear=False)

    def forward(self, x):
        x = self.b1(x)
        res = self.b2(x)
        return res


class BilinearUpsample2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(BilinearUpsample2, self).__init__()

        self.b1 = up(in_ch=in_ch, out_ch=out_ch, bilinear=True)

    def forward(self, x):
        res = self.b1(x)
        return res


class BilinearUpsample4(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(BilinearUpsample4, self).__init__()

        self.b1 = up(in_ch=in_ch, out_ch=out_ch, bilinear=True)
        self.b2 = up(in_ch=out_ch, out_ch=out_ch, bilinear=True)

    def forward(self, x):
        x = self.b1(x)
        res = self.b2(x)
        return res

class segnetDown3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(segnetDown3, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                         padding=1)
        self.conv2 = conv2DBatchNormRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                         padding=1)
        self.conv3 = conv2DBatchNormRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                         padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x_1)
        x_3 = self.conv3(x_2)
        p_4 = self.max_pool(x_3)
        return p_4, x_3


class inception_module_3(nn.Module):
    def __init__(self, in_ch):
        super(inception_module_3, self).__init__()
        self.ConvI1 = nn.Conv2d(in_channels=in_ch, out_channels=128, kernel_size=1, stride=1, padding=0)

        self.ConvI2_1 = nn.Conv2d(in_channels=in_ch, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.ConvI2_2 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.ConvI3_1 = nn.Conv2d(in_channels=in_ch, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.ConvI3_2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=5, stride=1, padding=2)

        self.ConvI4_1 = nn.Conv2d(in_channels=in_ch, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.ConvI4_2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        x1 = self.ConvI1(x)
        x2_1 = self.ConvI2_1(x)
        x2_2 = self.ConvI2_2(x2_1)

        x3_1 = self.ConvI3_1(x)
        x3_2 = self.ConvI3_2(x3_1)

        x4_1 = self.ConvI4_1(x)
        x4_2 = self.ConvI4_2(x4_1)

        x = torch.cat([x1, x2_2, x3_2, x4_2], dim=1)
        return x


class DUpsampling(nn.Module):
    def __init__(self, inplanes, scale, outplanes, pad=0, residual=False):
        super(DUpsampling, self).__init__()
        ## W matrix
        self.conv_w = nn.Conv2d(inplanes, outplanes * scale * scale, kernel_size=1, padding=pad, bias=False)
        ## P matrix
        self.conv_p = nn.Conv2d(outplanes * scale * scale, inplanes, kernel_size=1, padding=pad, bias=False)

        self.scale = scale
        self.residual = residual
        if self.residual is True:
            self.conv1 = nn.Conv2d(in_channels=outplanes, out_channels=outplanes, kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(num_features=outplanes)

            self.conv2 = nn.Conv2d(in_channels=outplanes, out_channels=outplanes, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(num_features=outplanes)

            self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.conv_w(x)
        N, C, H, W = x.size()

        # N, W, H, C
        x_permuted = x.permute(0, 3, 2, 1)

        # N, W, H*scale, C/scale
        x_permuted = x_permuted.contiguous().view((N, W, H * self.scale, int(C / (self.scale))))

        # N, H*scale, W, C/scale
        x_permuted = x_permuted.permute(0, 2, 1, 3)
        # N, H*scale, W*scale, C/(scale**2)
        x_permuted = x_permuted.contiguous().view(
            (N, W * self.scale, H * self.scale, int(C / (self.scale * self.scale))))

        # N, C/(scale**2), H*scale, W*scale
        x = x_permuted.permute(0, 3, 1, 2)

        if self.residual is True:
            residual = x

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.conv2(x)
            x = self.bn2(x)

            x = residual + x
            x = self.relu(x)

        return x
