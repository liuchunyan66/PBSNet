import torch
from torch import nn
from torch.nn import functional as F
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt


def dice_loss_func(input, target):
    smooth = 1.
    n = input.size(0)
    iflat = input.view(n, -1)
    tflat = target.view(n, -1)
    intersection = (iflat * tflat).sum(1)
    loss = 1 - ((2. * intersection + smooth) /
                (iflat.sum(1) + tflat.sum(1) + smooth))
    return loss.mean()


# class ClassSegLoss(nn.Module):
#     def __init__(self, channel, *args, **kwargs):
#         super(ClassSegLoss, self).__init__()
#         self.laplacian_kernel = torch.tensor(
#             [-1, -1, -1, -1, 8, -1, -1, -1, -1],
#             dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).type(torch.cuda.FloatTensor)
#         # self.sobel_kernel = torch.tensor(
#         #     [1, 2, 1, 0, 0, 0, -1, -2, -1],
#         #     dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).type(torch.cuda.FloatTensor)
#
#         # self.fuse_kernel = torch.nn.Parameter(torch.tensor([[6. / 10], [3. / 10], [1. / 10]],
#         #                                                    dtype=torch.float32).reshape(1, 3, 1, 1).type(
#         #     torch.cuda.FloatTensor))
#
#         # 用高层语义特征产生一个注意力来指导底层语义特征
#         self.conv = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False).type(torch.cuda.FloatTensor),
#             # nn.BatchNorm2d(32),
#             nn.ReLU(True),
#             nn.Conv2d(32, 64, kernel_size=1, padding=0, bias=False).type(torch.cuda.FloatTensor),
#             # nn.BatchNorm2d(64),
#             nn.ReLU(True),
#             nn.Conv2d(64, channel, kernel_size=1, padding=0, bias=False).type(torch.cuda.FloatTensor)
#
#         )
#
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, feat, gtmasks):
#         # 产生边缘信息
#         boundary_targets = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, padding=1)
#         # 将输入张量每个元素的范围限制到区间[min, max]，返回结果到一个新张量
#         boundary_targets = boundary_targets.clamp(min=0)
#         boundary_targets[boundary_targets > 0.1] = 1
#         boundary_targets[boundary_targets <= 0.1] = 0
#
#         # classes_boudary = torch.zeros(gtmasks.size()).cuda()
#         # classes_boudary = classes_boudary.unsqueeze(1).type(torch.cuda.FloatTensor)
#         # classes_ids = np.unique(gtmasks.cpu().numpy())
#         # gtmasks = gtmasks.cpu().numpy()
#         #
#         # for idx in classes_ids:
#         #     if idx not in [0, 255]:
#         #         imgidx = np.zeros(gtmasks.shape)
#         #         imgidx[np.where(gtmasks == idx)] = 1
#         #
#         #         class_boundary = torch.from_numpy(imgidx).cuda()
#         #         boundary = F.conv2d(class_boundary.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, stride=1, padding=1)
#         #         # edge1 = boundary.squeeze().cpu().numpy()
#         #         # plt.subplot(1, 2, 1)
#         #         # plt.imshow(edge1, cmap='gray')
#         #         # plt.show()
#         #
#         #         classes_boudary += boundary
#
#         # boundary_target = boundary_targets[0]
#         # # boundary_target.squeeze(0)
#         # boundary_target = boundary_target.cpu().numpy()
#         # plt.imshow(boundary_target)
#         # plt.show()
#
#
#         boundary_atten = self.conv(boundary_targets)
#         boundary_atten = self.softmax(boundary_atten)
#
#         if boundary_atten.shape[-1] != feat.shape[-1]:
#             boundary_atten = F.interpolate(
#                 boundary_atten, feat.shape[2:], mode='bilinear', align_corners=True)
#         out = torch.mul(boundary_atten, feat)
#         # out = out + feat
#
#         return out
#
#
#     def get_params(self):
#         wd_params, nowd_params = [], []
#         for name, module in self.named_modules():
#             nowd_params += list(module.parameters())
#         return nowd_params

def gram_matrix(input):
    b, ch, h, w = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(b, ch, w*h)
    features_t = features.transpose(1, 2)
    G = torch.bmm(features, features_t) / (ch*h*w) # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G



class ClassSegLoss(nn.Module):
    def __init__(self, channuls, *args, **kwargs):
        super(ClassSegLoss, self).__init__()

        # 用高层语义特征产生一个注意力来指导底层语义特征
        self.conv = nn.Sequential(
            nn.Conv2d(1, channuls, kernel_size=3,padding=1, bias=False).type(torch.cuda.FloatTensor),
            nn.ReLU(True),
            nn.Conv2d(channuls, 1, 1, bias=False).type(torch.cuda.FloatTensor),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, channuls, kernel_size=3, padding=1, bias=False).type(torch.cuda.FloatTensor),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channuls, channuls // 2, 1, bias=False).type(torch.cuda.FloatTensor),
            nn.ReLU(True)
        )
        self.conv3 = nn.Conv2d(channuls // 2, 1, 3, 1, bias=False).type(torch.cuda.FloatTensor)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, feat_atten, gtmasks):
        gtmasks = gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor)

        seg_atten = self.conv(gtmasks)
        # seg_atten = self.softmax(seg_atten)
        # seg_atten = self.fc(seg_atten)

        if feat_atten.shape[-1] != seg_atten.shape[-1]:
            seg_atten = F.interpolate(
                seg_atten, feat_atten.shape[2:], mode='bilinear', align_corners=True)

        seg_conv1 = self.conv1(seg_atten)
        seg_conv2 = self.conv2(seg_conv1)
        seg_conv3 = self.conv3(seg_conv2)
        feat_conv1 = self.conv1(feat_atten)
        feat_conv2 = self.conv2(feat_conv1)
        feat_conv3 = self.conv3(feat_conv2)
        conv2_t = gram_matrix(seg_conv2)
        conv2_g = gram_matrix(feat_conv2)
        conv3_t = gram_matrix(seg_conv3)
        conv3_g = gram_matrix(feat_conv3)
        loss1 = F.mse_loss(conv2_g, conv2_t)
        loss2 = F.mse_loss(conv3_g, conv3_t)

        return loss1, loss2

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            nowd_params += list(module.parameters())
        return nowd_params


if __name__ == '__main__':
    torch.manual_seed(15)
    with open('../cityscapes_info.json', 'r') as fr:
        labels_info = json.load(fr)
    lb_map = {el['id']: el['trainId'] for el in labels_info}

    img_path = 'D:/data/gtFine_trainvaltest/gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png'
    img = cv2.imread(img_path, 0)

    label = np.zeros(img.shape, np.uint8)
    for k, v in lb_map.items():
        label[img == k] = v

    img_tensor = torch.from_numpy(label).cuda()
    img_tensor = torch.unsqueeze(img_tensor, 0).type(torch.cuda.FloatTensor)

    detailAggregateLoss = ClassSegLoss(19)
    for param in detailAggregateLoss.parameters():
        print(param)

    bce_loss = detailAggregateLoss(torch.unsqueeze(img_tensor, 0), img_tensor)
    print(bce_loss)