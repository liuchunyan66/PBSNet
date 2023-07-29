import torch
from torch import nn
from torch.nn import functional as F
import cv2
from PIL import Image, ImageFilter
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


def get_one_hot(label, N):
    size = list(label.size())
    label = label.view(-1)  # reshape 为向量
    ones = torch.sparse.torch.eye(N).cuda()
    ones = ones.index_select(0, label.long())  # 用上面的办法转为换one hot
    size.append(N)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸
    return ones.view(*size)


def get_boundary(gtmasks):
    laplacian_kernel = torch.tensor(
        [-1, -1, -1, -1, 8, -1, -1, -1, -1],
        dtype=torch.float32, device=gtmasks.device).reshape(1, 1, 3, 3).requires_grad_(False)
    # boundary_logits = boundary_logits.unsqueeze(1)
    boundary_targets = F.conv2d(gtmasks.unsqueeze(1), laplacian_kernel, padding=1)
    boundary_targets = boundary_targets.clamp(min=0)
    boundary_targets[boundary_targets > 0.1] = 1
    boundary_targets[boundary_targets <= 0.1] = 0
    return boundary_targets


class BoundaryLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BoundaryLoss, self).__init__()

        self.laplacian_kernel = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).type(torch.cuda.FloatTensor)

        self.fuse_kernel = torch.nn.Parameter(torch.tensor([[5. / 10], [3. / 10], [2. / 10]],
                                                           dtype=torch.float32).reshape(1, 3, 1, 1).type(
            torch.cuda.FloatTensor))

    def forward(self, prodict_seg, gtmasks):

        boundary_targets = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, padding=1)
        boundary_targets = boundary_targets.clamp(min=0)
        boundary_targets[boundary_targets > 0.1] = 1
        boundary_targets[boundary_targets <= 0.1] = 0

        prodict_seg = prodict_seg.cpu().detach().numpy()  # BxCxHxW -> BxCxHxW
        prodict_seg = prodict_seg.transpose(0, 2, 3, 1)  # BxCxHxW   -> BxHxWxC
        prodict_seg = np.asarray(np.argmax(prodict_seg, axis=3), dtype=np.uint8)  # BxHxWxC, 网络的分割图
        prodict_seg = torch.tensor(prodict_seg)

        if prodict_seg.shape[-1] != boundary_targets.shape[-1]:
            prodict_seg = F.interpolate(
                prodict_seg, boundary_targets.shape[2:], mode='bilinear', align_corners=True)
        boundary_logits = F.conv2d(prodict_seg.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel,
                                   padding=1)
        boundary_logits = boundary_logits.clamp(min=0)
        boundary_logits[boundary_logits > 0.1] = 1
        boundary_logits[boundary_logits <= 0.1] = 0


        bce_loss = F.binary_cross_entropy_with_logits(boundary_logits, boundary_targets)
        dice_loss = dice_loss_func(torch.sigmoid(boundary_logits), boundary_targets)

        return bce_loss, dice_loss

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            nowd_params += list(module.parameters())
        return nowd_params


if __name__ == '__main__':
    torch.manual_seed(15)
    # with open('../cityscapes_info.json', 'r') as fr:
    #         labels_info = json.load(fr)
    # lb_map = {el['id']: el['trainId'] for el in labels_info}

    img_path = 'D:/data/val_gt/val/frankfurt/frankfurt_000000_014480_gtFine_labelIds.png'
    # img_path = 'D:/data/Camvid/camvid/testannot/Seq05VD_f00930_L.png'
    label = cv2.imread(img_path, 0)
    # cv2.imshow("image", img)

    # print(type(img))

    # label = np.zeros(img.shape, np.uint8)
    # for k, v in lb_map.items():
    #     label[img == k] = v

    img_tensor = torch.from_numpy(label).cuda()
    img_tensor = torch.unsqueeze(img_tensor, 0).type(torch.cuda.FloatTensor)

    # detailAggregateLoss = DetailAggregateLoss()
    BoundaryLoss = BoundaryLoss()
    # for param in detailAggregateLoss.parameters():
    #     print(param)

    bce_loss, dice_loss, boudary_targets_pyramid, gtmask = BoundaryLoss(torch.unsqueeze(img_tensor, 0), img_tensor)
    # print(bce_loss,  dice_loss)
    gtmask = gtmask.squeeze()
    gtmask = gtmask.cpu().detach().numpy()
    # cv2.imshow("gtmask", gtmask)
    # print(boudary_targets_pyramid)
    boudary_targets_pyramid = boudary_targets_pyramid.squeeze()
    img_numpy = boudary_targets_pyramid.cpu().detach().numpy()
    # print(type(img_numpy))
    # print(img.shape)
    img = np.array(255 * img_numpy, np.uint8)
    # print(img)
    # print(img_numpy)
    cv2.imshow("bound", img_numpy)
    cv2.imwrite('C:/Users/ASUS/Desktop/boundary/boundary4_gt.jpg', img)
    cv2.waitKey(0)
