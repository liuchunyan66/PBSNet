import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models.base_model.resnet import Resnet18

BatchNorm2d = nn.BatchNorm2d


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False)
        # self.bn = BatchNorm2d(out_chan)
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU()
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

class CorrModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(CorrModule, self).__init__()
        self.conv1 = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.sab1 = ConvBNReLU(out_chan, out_chan, ks=1, stride=1, padding=0)
        self.sab2 = ConvBNReLU(out_chan, out_chan, ks=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        self.cab1 = ConvBNReLU(out_chan, out_chan, ks=1, stride=1, padding=0)
        self.cab2 = ConvBNReLU(out_chan, out_chan, ks=1, stride=1, padding=0)

        self.alph = ConvBNReLU(out_chan, out_chan, ks=3, stride=1, padding=1)
        self.beta = ConvBNReLU(out_chan, out_chan, ks=3, stride=1, padding=1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        self.conv2 = ConvBNReLU(out_chan, out_chan, ks=1, stride=1, padding=0)

        self.init_weight()

    def forward(self, f4, f5, fg):

        f = torch.cat([f4, f5, fg], dim=1)
        f = self.conv1(f)
        Ms = self.sab1(f)
        Ms = self.sab2(Ms)

        avg = F.avg_pool2d(f, f.size()[2:])
        Mc = self.cab1(avg)
        Mc = self.cab2(Mc)

        alph = self.alph(f)
        alph = self.softmax(alph)

        beta = self.beta(f)
        beta = self.softmax(beta)

        Ms = torch.mul(Ms, alph)
        Mc = torch.mul(Mc, beta)

        Mf = Mc + Ms
        Mf = self.sigmoid(Mf)
        fs = torch.mul(f, Mf) + f
        out = self.conv2(fs)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

class AdaptiveSpatialAttentionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AdaptiveSpatialAttentionModule, self).__init__()
        self.conv_a1 = ConvBNReLU(2, out_chan, ks=1, stride=1, padding=0)
        self.conv_a2 = ConvBNReLU(out_chan, 1, ks=1, stride=1, padding=0)
        self.conv_t = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.sigmoid_atten = nn.Sigmoid()

        self.init_weight()

    def forward(self, x):
        feat = self.conv_t(x)
        # 空间注意力
        max_mask, _ = torch.max(x, dim=1, keepdim=True)
        avg_mask = torch.mean(x, dim=1, keepdim=True)
        mask = torch.cat([max_mask, avg_mask], dim=1)
        sp_atten = self.conv_a1(mask)
        sp_atten = self.conv_a2(sp_atten)

        atten = sp_atten * feat
        atten = self.sigmoid_atten(atten)
        out = atten * feat

        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class BiPropagateModule(nn.Module):
    def __init__(self, lchan, mchan, hchan, out_chan, *args, **kwargs):
        super(BiPropagateModule, self).__init__()

        self.conv_mid0 = ConvBNReLU(hchan + mchan, out_chan, ks=1, stride=1, padding=0)
        self.conv_low = ConvBNReLU(lchan + out_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv_mid = ConvBNReLU(out_chan * 2, out_chan, ks=1, stride=1, padding=0)
        self.conv_high = ConvBNReLU(out_chan * 2, out_chan, 1, 1, 0, bias=False)

        self.down_low = ConvBNReLU(out_chan, out_chan, 3, 2, 1)
        self.down_mid = ConvBNReLU(out_chan, out_chan, 3, 2, 1)

        self.init_weight()

    def forward(self, low, mid, high):
        high_up = F.interpolate(high, mid.size()[2:], mode='nearest')
        feat_mid0 = torch.cat([high_up, mid], dim=1)
        feat_mid0 = self.conv_mid0(feat_mid0)
        feat_mid_up = F.interpolate(feat_mid0, low.size()[2:], mode='nearest')
        feat_low = torch.cat([feat_mid_up, low], dim=1)
        feat_low = self.conv_low(feat_low)

        feat_low_down = self.down_low(feat_low)
        feat_mid = torch.cat([feat_mid0, feat_low_down], dim=1)
        feat_mid = self.conv_mid(feat_mid)

        feat_mid_down = self.down_mid(feat_mid)
        feat_high = torch.cat([feat_mid_down, high], dim=1)
        feat_high = self.conv_high(feat_high)

        return feat_low, feat_mid, feat_high

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

class ContextPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ContextPath, self).__init__()

        self.backbone = Resnet18()
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)
        self.conv32 = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)
        self.conv16 = ConvBNReLU(256, 128, ks=1, stride=1, padding=0)
        self.SEM = CorrModule(128*3, 128)
        self.BVP = BiPropagateModule(64, 128, 128, 128)
        self.ASA16 = AdaptiveSpatialAttentionModule(128, 128)
        self.ASA8 = AdaptiveSpatialAttentionModule(128, 128)
        self.ASA4 = AdaptiveSpatialAttentionModule(128, 128)

        self.init_weight()

    def forward(self, x):

        feat4, feat8, feat16, feat32 = self.backbone(x)
        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]

        avg = F.avg_pool2d(feat32, feat32.size()[2:])
        avg = self.conv_avg(avg)
        # print(avg.size())
        fg = F.interpolate(avg, (H16, W16), mode='nearest')
        f5 = self.conv32(feat32)
        f5 = F.interpolate(f5, (H16, W16), mode='nearest')
        f4 = self.conv16(feat16)
        fs = self.SEM(f4, f5, fg)
        A3, A2, A1 = self.BVP(feat4, feat8, fs)

        asa4 = self.ASA4(A3)
        asa8 = self.ASA8(A2)
        asa16 = self.ASA16(A1)

        return feat4, feat8, feat16, asa4, asa8, asa16

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

class FFM(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FFM, self).__init__()
        self.conv = ConvBNReLU(in_chan*3, out_chan, ks=1, stride=1, padding=0)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fcs = []
        for i in range(3):
            self.fcs.append(
                nn.Linear(out_chan, in_chan)
            )
        self.fcs = nn.ModuleList(self.fcs)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid_atten = nn.Sigmoid()

        self.init_weight()

    def forward(self, flp, fmp, fhp):
        H8, W8 = flp.size()[2:]
        fmp = F.interpolate(fmp, (H8, W8), mode='nearest')
        fhp = F.interpolate(fhp, (H8, W8), mode='nearest')
        feats = torch.cat((flp, fmp, fhp), dim=1)
        feats = self.conv(feats)
        feat_global = self.avg(feats).squeeze()
        for i, fc in enumerate(self.fcs):
            vector = fc(feat_global).unsqueeze(1)  # [B, 1, C]
            if i == 0:
                attention_vector = vector
            else:
                attention_vector = torch.cat([attention_vector, vector], dim=1)
        attention_vector = self.softmax(attention_vector)  # [B, 3, C]
        # [B, 1, C]
        vectors = attention_vector.chunk(3, dim=1)
        # [B, C, 1, 1]
        vector1 = vectors[0].squeeze().unsqueeze(-1).unsqueeze(-1)
        vector2 = vectors[1].squeeze().unsqueeze(-1).unsqueeze(-1)
        vector3 = vectors[2].squeeze().unsqueeze(-1).unsqueeze(-1)

        out1 = flp * vector1
        out2 = fmp * vector2
        out3 = fhp * vector3
        out = out1 + out2 + out3
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class PBSNet(nn.Module):
    def __init__(self, n_classes, *args, **kwargs):
        super(PBSNet, self).__init__()

        self.cp = ContextPath()
        self.afm = FFM(128, 128)

        self.seg_out = BiSeNetOutput(128, 128, n_classes)
        self.seg_out32 = BiSeNetOutput(128, 128, n_classes)
        self.seg_out16 = BiSeNetOutput(128, 128, n_classes)
        self.boundary_out8 = BiSeNetOutput(128, 128, 1)
        self.boundary_out4 = BiSeNetOutput(64, 128, 1)
        self.boundary_out16 = BiSeNetOutput(256, 128, 1)
        self.init_weight()

    def forward(self, x):
        feat4, feat8, feat16, ASA_low, ASA_mid, ASA_high = self.cp(x)

        H, W = x.size()[2:]
        out8 = self.afm(ASA_low, ASA_mid, ASA_high)
        seg_out = self.seg_out(out8)
        seg_out32 = self.seg_out32(ASA_high)
        seg_out16 = self.seg_out32(ASA_mid)

        feat_out = F.interpolate(seg_out, (H, W), mode='bilinear', align_corners=True)
        feat_out16 = F.interpolate(seg_out16, (H, W), mode='bilinear', align_corners=True)
        feat_out32 = F.interpolate(seg_out32, (H, W), mode='bilinear', align_corners=True)
        boundary_out8 = self.boundary_out8(feat8)
        boundary_out4 = self.boundary_out4(feat4)
        # boundary_out16 = self.boundary_out16(feat16)

        return feat_out, feat_out16, feat_out32, boundary_out4, boundary_out8

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, (FFM, BiSeNetOutput)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


if __name__ == "__main__":
    net = PBSNet(19)
    net.cuda()
    net.eval()
    in_ten = torch.randn(1, 3, 768, 1536).cuda()
    feat_out, feat_out16, feat_out32, boundary_out4, boundary_out8 = net(in_ten)
    print(feat_out.shape)