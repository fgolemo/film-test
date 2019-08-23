import torch
import torch.nn as nn

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class FiLM(nn.Module):
    """
  A Feature-wise Linear Modulation Layer from
  'FiLM: Visual Reasoning with a General Conditioning Layer'
  """

    def forward(self, x, gammas, betas):
        gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
        betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)
        return (gammas * x) + betas


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None,
                 film_in=None):
        super(BasicBlock, self).__init__()
        self.filmed = False
        if film_in is not None:
            self.filmed = True
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.film = FiLM()

    def forward(self, x):
        if self.filmed:
            x, gammas, betas = x

        identity = x

        out = self.conv1(x)

        if not self.filmed:
            out = self.bn1(out)
        else:
            pass  # if FiLM, we don't apply anything here

        out = self.relu(out)
        out = self.conv2(out)

        if not self.filmed:
            out = self.bn2(out)
        else:
            out = self.film(out, gammas, betas)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 num_classes=1000,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None,
                 film_inputs=None,
                 film_init_gamma_one=False):
        super(ResNet, self).__init__()
        self.film_in = film_inputs

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(
                                 replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if self.film_in is not None:
            assert type(self.film_in) == type(42)  # check if int
            self.film1 = nn.Linear(self.film_in, 32)
            self.film2 = nn.Linear(
                32, 2 * (64 * 2 + 128 * 2 + 256 * 2 + 512 * 2))  # gamma+beta

        self.layer1_a, self.layer1_b = self._make_layer(block, 64, layers[0])
        self.layer2_a, self.layer2_b = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0])
        self.layer3_a, self.layer3_b = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1])
        self.layer4_a, self.layer4_b = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation, norm_layer, self.film_in))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    film_in=self.film_in))

        return layers

    def forward(self, x, film_embed=None):
        """

        :param x: image, (minibatch, 3, 32, 32)
        :param film_embed: one-hot vector question embedding, (minibatch, 2,)
        :return:
        """
        if film_embed is not None:
            # precompute all the gammas/betas, one for each
            film_x = self.relu(self.film1(film_embed))
            film_x = self.film2(film_x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if film_embed is not None:
            last_idx = 0

            #FIXME: This is suuuuper nasty and I hope there's a better way of doing that

            x = self.layer1_a((
                x,
                film_x[:, last_idx:last_idx + 64],  # gammas
                film_x[:, last_idx + 64:last_idx + 64 * 2]))  # betas
            last_idx += 64 * 2
            x = self.layer1_b((
                x,
                film_x[:, last_idx:last_idx + 64],  # gammas
                film_x[:, last_idx + 64:last_idx + 64 * 2]))  # betas
            last_idx += 64 * 2

            x = self.layer2_a((
                x,
                film_x[:, last_idx:last_idx + 128],  # gammas
                film_x[:, last_idx + 128:last_idx + 128 * 2]))  # betas
            last_idx += 128 * 2
            x = self.layer2_b((
                x,
                film_x[:, last_idx:last_idx + 128],  # gammas
                film_x[:, last_idx + 128:last_idx + 128 * 2]))  # betas
            last_idx += 128 * 2

            x = self.layer3_a((
                x,
                film_x[:, last_idx:last_idx + 256],  # gammas
                film_x[:, last_idx + 256:last_idx + 256 * 2]))  # betas
            last_idx += 256 * 2
            x = self.layer3_b((
                x,
                film_x[:, last_idx:last_idx + 256],  # gammas
                film_x[:, last_idx + 256:last_idx + 256 * 2]))  # betas
            last_idx += 256 * 2

            x = self.layer4_a((
                x,
                film_x[:, last_idx:last_idx + 512],  # gammas
                film_x[:, last_idx + 512:last_idx + 512 * 2]))  # betas
            last_idx += 512 * 2
            x = self.layer4_b((
                x,
                film_x[:, last_idx:last_idx + 512],  # gammas
                film_x[:, last_idx + 512:last_idx + 512 * 2]))  # betas

        else:
            #FIXME: yeah, this is nasty, too
            x = self.layer1_a(x)
            x = self.layer1_b(x)
            x = self.layer2_a(x)
            x = self.layer2_b(x)
            x = self.layer3_a(x)
            x = self.layer3_b(x)
            x = self.layer4_a(x)
            x = self.layer4_b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18(**kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    """

    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


if __name__ == '__main__':
    from torchsummary import summary

    # net = resnet18(num_classes=2)
    # summary(net, input_size=(3, 32, 32))
    net = resnet18(num_classes=2, film_inputs=2)
    # summary(net, input_size=[(3, 32, 32), (2,)]) # for some reason this doesn't work
    abc = net.forward(torch.zeros((2, 3, 32, 32)), torch.zeros((2, 2)))
