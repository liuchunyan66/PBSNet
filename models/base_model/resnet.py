import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as modelzoo

resnet18_url = 'D:/pretrain/resnet18-5c106cde.pth'

from torch.nn import BatchNorm2d


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_chan, out_chan, stride)
        self.bn1 = BatchNorm2d(out_chan)
        self.conv2 = conv3x3(out_chan, out_chan)
        self.bn2 = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(out_chan),
                )

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = shortcut + residual
        out = self.relu(out)
        return out


def create_layer_basic(in_chan, out_chan, bnum, stride=1):
    layers = [BasicBlock(in_chan, out_chan, stride=stride)]
    for i in range(bnum-1):
        layers.append(BasicBlock(out_chan, out_chan, stride=1))
    return nn.Sequential(*layers)


class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.conv0_0 = conv3x3(3, 32)
        self.bn0 = BatchNorm2d(32)
        self.relu0 = nn.ReLU(inplace=True)
        self.conv0_1 = conv3x3(32, 32)
        self.bn2 = BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv0_2 = conv3x3(32, 64)
        self.bn1 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = create_layer_basic(64, 64, bnum=2, stride=1)
        self.layer2 = create_layer_basic(64, 128, bnum=2, stride=2)
        self.layer3 = create_layer_basic(128, 256, bnum=2, stride=2)
        self.layer4 = create_layer_basic(256, 512, bnum=2, stride=2)
        self.init_weight()

    def forward(self, x):
        x = self.conv0_0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.conv0_1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv0_2(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        feat4 = self.layer1(x)
        feat8 = self.layer2(feat4) # 1/8
        feat16 = self.layer3(feat8) # 1/16
        feat32 = self.layer4(feat16) # 1/32
        return feat4, feat8, feat16, feat32

    def init_weight(self):
        pretrained_dict = torch.load(resnet18_url)
        self_state_dict = self.state_dict()
        state_dict = {k: v for k, v in pretrained_dict.items() if k in self_state_dict.keys()}  # filter out unnecessary keys
        self_state_dict.update(state_dict)
        self.load_state_dict(self_state_dict)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


if __name__ == "__main__":
    net = Resnet18()
    x = torch.randn(16, 3, 512, 1024)
    out = net(x)
    print(out[0].size())
    print(out[1].size())
    print(out[2].size())
    print(out[3].size())
    net.get_params()


if __name__ == "__main__":
    print('start')
    cuda1 = torch.device('cuda:2')
    input = torch.rand(3, 3, 768, 768).to(device=cuda1)
    model = resnet18()
    print(sum([p.numel() for p in model.parameters() if p.requires_grad]))
    model = model.to(device=cuda1)
    print('model: ', model)
    model.eval()
    output = model(input)
    print(output[-1].size())