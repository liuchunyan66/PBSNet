import torch
from torch import nn
from torch.nn import functional as F
import cv2
import numpy as np
import json


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
    # Binary images of each class
    # 获取gtmask中的所有id值
    classes_ids = np.unique(gtmasks)
    print(classes_ids)
    # 循环取出每个类别id对应的像素图片
    imgIds_list = []

    for idx in classes_ids:
        if idx not in [0, 255]:
            # 获取二值图片
            imgidx = gtmasks
            imgidx[np.where(imgidx!=idx)] = 0
            imgidx[np.where(imgidx == idx)] = 1
            # 得到类边缘图
            cv2.Canny(imgidx, 30, 100)
            # print(imgidx)
            imgIds_list.append(imgidx)
    # print('imgIds_list:', len(imgIds_list))
    class_boudary = torch.add(imgIds_list)

    return class_boudary


class ClassesLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ClassesLoss, self).__init__()

        self.laplacian_kernel = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).type(torch.cuda.FloatTensor)

        self.fuse_kernel = torch.nn.Parameter(torch.tensor([[6. / 10], [3. / 10], [1. / 10]],
                                                           dtype=torch.float32).reshape(1, 3, 1, 1).type(
            torch.cuda.FloatTensor))

    def forward(self, boundary_logits, gtmasks):
        # print('boundary:', boundary_logits.size())
        # print('gtmask:', gtmasks.size())
        classes_boudary = torch.zeros(gtmasks.size()).cuda()
        gtmasks = gtmasks.cpu().numpy()
        classes_ids = np.unique(gtmasks)

        for idx in classes_ids:
            if idx not in [0, 255]:
                imgidx = gtmasks
                imgidx[np.where(imgidx != idx)] = 0
                imgidx[np.where(imgidx == idx)] = 1
                imgidx = imgidx.astype(np.uint8)
                cv2.Canny(imgidx, 30, 100)
                class_boudary = torch.from_numpy(imgidx).cuda()
                # print(class_boudary.size())
                classes_boudary += class_boudary
                # print(classes_boudary.size())

        classes_boudary = classes_boudary.unsqueeze(1).type(torch.cuda.FloatTensor)
        # print('fanish',classes_boudary.size())


        if boundary_logits.shape[-1] != classes_boudary.shape[-1]:
            boundary_logits = F.interpolate(
                boundary_logits, classes_boudary.shape[2:], mode='bilinear', align_corners=True)

        bce_loss = F.binary_cross_entropy_with_logits(boundary_logits, classes_boudary)
        dice_loss = dice_loss_func(torch.sigmoid(boundary_logits), classes_boudary)
        return bce_loss, dice_loss

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
    print('img_tensor:', img_tensor.size())

    detailAggregateLoss = ClassesLoss()
    for param in detailAggregateLoss.parameters():
        print(param)

    bce_loss, dice_loss = detailAggregateLoss(torch.unsqueeze(img_tensor, 0), img_tensor)
    print(bce_loss,  dice_loss)
