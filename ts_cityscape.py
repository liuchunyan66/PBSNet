from logger import setup_logger
from models.step_to_step.fusion import BiSeNet
from models.SABA.saba_dense import BiSeNet
from models.Res.two_stream import ResNet
# from models.TestModel.saba import BiSeNet
from models.TestModel.bisenetv2 import BiSeNetV2
from cityscapes import CityscapesTDataSet
from PIL import Image
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist

import os
import os.path as osp
import logging
import time
import numpy as np
from tqdm import tqdm
import math
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# seg_im_path = 'D:/STDC-Seg-master/visual/cityscape/val/PBSNet/seg'
# output_im_path = 'D:/STDC-Seg-master/visual/cityscape/val/PBSNet/color'

seg_im_path = 'F:/SABA/seg'
output_im_path = 'F:/SABA/color'

# seg_im_path = 'D:/STDC-Seg-master/visual/compare/saba/seg'
# output_im_path = 'D:/STDC-Seg-master/visual/compare/saba/color'

# seg_im_path = 'D:/STDC-Seg-master/visual/compare/bisenet/seg'
# output_im_path = 'D:/STDC-Seg-master/visual/compare/bisenet/color'


def decode_segmap(temp):
    colors = [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]
    label_colours = dict(zip(range(19), colors))
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, 19):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]
    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return rgb


def id2trainId(label):
    label_copy = label.copy()
    id_to_trained ={
        7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8,
        22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17,
        33: 18
    }
    for k, v in id_to_trained.items():
        label_copy[label == v] = k
    return label_copy


def tstSegmenttion(respth='./pretrained', dspth='./data', backbone='CatNetSmall'):
    ## dataset
    batchsize = 1
    n_workers = 1
    dsval = CityscapesTDataSet(dspth)
    dl = DataLoader(dsval,
                    batch_size=batchsize,
                    shuffle=False,
                    num_workers=n_workers,
                    drop_last=False)

    n_classes = 19
    print("backbone:", backbone)
    net = BiSeNet(backbone=backbone, n_classes=n_classes)
    # net = ResNet(n_classes)
    pretrained_dict = torch.load(respth)
    model_dict = net.state_dict()
    state_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}  # filter out unnecessary keys
    model_dict.update(state_dict)
    net.load_state_dict(model_dict)
    # net.load_state_dict(torch.load(respth))
    net.cuda()
    net.eval()

    with torch.no_grad():
        data_time = []
        for step, batch in enumerate(dl):
            images, name = batch

            N, C, H, W = images.size()

            # 缩放图片大小进行预测
            # new_hw = [int(H * 0.5), int(W * 0.5)]

            # images = F.interpolate(images, new_hw, mode='bilinear', align_corners=True)
            #
            images = Variable(images).cuda()
            start_time = time.time()
            outputs = net(images)[0]

            # 计算一次测试时间
            end_time = time.time()
            data_time.append(end_time - start_time)

            # _, pred = torch.max(output, 1)
            # pred = pred.cpu().data.numpy()      # 将Tensor --> numpy
            # img = img.cpu().data.numpy()
            # save seg image
            outputs = outputs.cpu().data[0].numpy()   # 1xCxHxW -> CxHxW
            outputs = outputs.transpose(1, 2, 0)      # CxHxW   -> HxWxC
            outputs = np.asarray(np.argmax(outputs, axis=2), dtype=np.uint8)    # HxW, 网络的分割图

            seg_pred = id2trainId(outputs)  # 反转由标记变为类别号
            seg_pred = Image.fromarray(seg_pred)
            output_color = decode_segmap(outputs)  # 输出颜色值
            # output_color = Image.fromarray(output_color)
            output_color = Image.fromarray(np.uint8(output_color))
            seg_pred.save('{0}/{1}.png'.format(seg_im_path, name[0]))
            output_color.save('{0}/{1}.png'.format(output_im_path, name[0]))

        time_sum = 0
        for step, i in enumerate (data_time):
            print(i)
            if step > 500:
                time_sum += i
        print('test finish!!!')
        print("FPS为: %f"%(1.0/(time_sum / (len(data_time)-500))))




if __name__ == '__main__':
    tstSegmenttion('D:/STDC-Seg-master/checkpoints/SABA/saba_dense/model_final.pth',
                   dspth = 'D:/STDC-Seg-master/data',
                   backbone='STDCNet1446')


