from logger import setup_logger
from models.PBSNet import PBSNet
# from models.Chart.resnet_arm_ffm import BiSeNet
from cityscapes import CityScapes
# from loss.loss import WeightedOhemCELoss
from loss.loss import OhemCELoss
from loss.detail_loss import DetailAggregateLoss
from loss.boundary_loss import BoundaryLoss
from evaluation import MscEvalV0
from optimizer_loss import Optimizer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F


import os
import os.path as osp
import logging
import time
import datetime
import argparse

logger = logging.getLogger()

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        '--n_workers_train',
        dest='n_workers_train',
        type=int,
        default=8,
    )
    parse.add_argument(
        '--n_workers_val',
        dest='n_workers_val',
        type=int,
        default=1,
    )
    parse.add_argument(
        '--n_img_per_gpu',
        dest='n_img_per_gpu',
        type=int,
        default=2,
    )
    parse.add_argument(
        '--max_iter',
        dest='max_iter',
        type=int,
        default=80000,
    )
    parse.add_argument(
        '--save_iter_sep',
        dest='save_iter_sep',
        type=int,
        default=1000,
    )
    parse.add_argument(
        '--warmup_steps',
        dest='warmup_steps',
        type=int,
        default=1000,
    )
    parse.add_argument(
        '--mode',
        dest='mode',
        type=str,
        default='train',
    )
    parse.add_argument(
        '--ckpt',
        dest='ckpt',
        type=str,
        default=None,
    )
    parse.add_argument(
        '--respath',
        dest='respath',
        type=str,
        default=None,
    )
    parse.add_argument(
        '--backbone',
        dest='backbone',
        type=str,
        default='STDCNet1446',
    )
    parse.add_argument(
        '--pretrain_path',
        dest='pretrain_path',
        type=str,
        default='D:/STDC-Seg-master/data/Pretrained_model/STDCNet1446_76.47.tar',
        # default='D:/pretrain/resnet18-5c106cde.pth'  D:/STDC-Seg-master/data/Pretrained_model/STDCNet1446_76.47.tar
    )
    parse.add_argument(
        '--use_conv_last',
        dest='use_conv_last',
        type=str2bool,
        default=False,
    )
    parse.add_argument(
        '--use_boundary_2',
        dest='use_boundary_2',
        type=str2bool,
        default=False,
    )
    parse.add_argument(
        '--use_boundary_4',
        dest='use_boundary_4',
        type=str2bool,
        default=True,
    )
    parse.add_argument(
        '--use_boundary_8',
        dest='use_boundary_8',
        type=str2bool,
        default=True,
    )
    parse.add_argument(
        '--use_boundary_16',
        dest='use_boundary_16',
        type=str2bool,
        default=True,
    )
    return parse.parse_args()

def train():
    args = parse_args()
    save_root = 'D:/STDC-Seg-master/checkpoints/SABA'
    save_pth_path = os.path.join(save_root, 'saba_dense')
    dspth = 'D:/STDC-Seg-master/data'

    if not osp.exists(save_pth_path):
        os.makedirs(save_pth_path)

    setup_logger(save_root)
    ## dataset
    n_classes = 19
    n_img_per_gpu = args.n_img_per_gpu
    n_workers_train = args.n_workers_train
    n_workers_val = args.n_workers_val
    use_boundary_16 = args.use_boundary_16
    use_boundary_8 = args.use_boundary_8
    use_boundary_4 = args.use_boundary_4
    use_boundary_2 = args.use_boundary_2

    mode = args.mode
    cropsize = [1024, 512]
    randomscale = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)


    ds = CityScapes(dspth, cropsize=cropsize, mode= mode, randomscale=randomscale)
    dl = DataLoader(ds,
                    batch_size=n_img_per_gpu,
                    shuffle=True,
                    num_workers=n_workers_train,
                    pin_memory=False,
                    drop_last=True)
    # exit(0)
    dsval = CityScapes(dspth, mode='val', randomscale=randomscale)
    dlval = DataLoader(dsval,
                       batch_size=1,
                       shuffle=False,
                       num_workers=n_workers_val,
                       drop_last=False)

    ## model
    ignore_idx = 255
    # net = BiSeNet(n_classes=n_classes,
    #               # pretrain_model=args.pretrain_path,
    #               use_boundary_32=use_boundary_32, use_boundary_4=use_boundary_4, use_boundary_8=use_boundary_8,
    #               use_boundary_16=use_boundary_16)
    net = BiSeNet(backbone=args.backbone, n_classes=n_classes, pretrain_model=args.pretrain_path,
                  use_boundary_2=use_boundary_2, use_boundary_4=use_boundary_4, use_boundary_8=use_boundary_8,
                  use_boundary_16=use_boundary_16, use_conv_last=args.use_conv_last)
    model_path = 'D:/STDC-Seg-master/checkpoints/SABA/saba_dense/model_final.pth'
    # model_path = 'D:/STDC-Seg-master/data/Pretrained_model/model_maxmIOU75.pth'
    pretrained_dict = torch.load(model_path)
    model_dict = net.state_dict()
    state_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}  # filter out unnecessary keys
    model_dict.update(state_dict)
    net.load_state_dict(model_dict)

    # # 冻结context path的权重
    # for k, v in net.named_parameters():
    #     if k.startswith('cp.backbone'):
    #         v.requires_grad = False
            # print(k)
            # print(v.requires_grad)
    print(net)

    if not args.ckpt is None:
        net.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
    # net = nn.DataParallel(net, device_ids=[0, 1, 2, 3])
    net.cuda()
    net.train()

    ## criterion，交叉熵损失、拉普拉斯算子的细节损失
    score_thres = 0.7
    n_min = n_img_per_gpu * cropsize[0] * cropsize[1] // 16
    criteria_p = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_16 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_8 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    # criteria_p = WeightedOhemCELoss(thresh=score_thres, n_min=n_min, num_classes=n_classes, ignore_lb=ignore_idx)
    # criteria_8 = WeightedOhemCELoss(thresh=score_thres, n_min=n_min, num_classes=n_classes, ignore_lb=ignore_idx)
    # criteria_16 = WeightedOhemCELoss(thresh=score_thres, n_min=n_min, num_classes=n_classes, ignore_lb=ignore_idx)
    # boundary_loss_func = DetailAggregateLoss()
    boundary_loss_func = BoundaryLoss()

    ## optimizer:SGD
    maxmIOU50 = 0.
    maxmIOU75 = 0.
    momentum = 0.9
    weight_decay = 5e-4
    lr_start = 0.001
    max_iter = args.max_iter
    save_iter_sep = args.save_iter_sep
    power = 0.9
    warmup_steps = args.warmup_steps
    warmup_start_lr = 1e-5

    optim = Optimizer(
        model=net,
        loss=boundary_loss_func,
        lr0=lr_start,
        momentum=momentum,
        wd=weight_decay,
        warmup_steps=warmup_steps,
        warmup_start_lr=warmup_start_lr,
        max_iter=max_iter,
        power=power)

    ## train loop
    msg_iter = 50
    loss_avg = []
    loss_boundery_bce = []
    loss_boundery_dice = []
    loss_s4 = []
    loss_s8 = []
    loss_s16 = []
    boundary_loss4 = []
    boundary_loss8 = []
    boundary_loss16 = []
    st = glob_st = time.time()  # 开始时间
    diter = iter(dl)
    epoch = 0
    for it in range(max_iter):
        try:
            im, lb, name = next(diter)
            if not im.size()[0] == n_img_per_gpu: raise StopIteration
        except StopIteration:
            epoch += 1
            # sampler.set_epoch(epoch)
            diter = iter(dl)
            im, lb, name = next(diter)
        im = im.cuda()
        lb = lb.cuda()
        H, W = im.size()[2:]
        # 去除维数为1的维度
        lb = torch.squeeze(lb, 1)

        optim.zero_grad()

        # 根据使用不同阶段产生的边界轮廓产生输出
        if use_boundary_4 and use_boundary_8 and use_boundary_16:
            out, out8, out16 = net(im)

        # if (not use_boundary_2) and use_boundary_4 and use_boundary_8:
        #     out, out16, out32, detail8, detail16 = net(im)
        #
        # if (not use_boundary_2) and (not use_boundary_4) and use_boundary_8:
        #     out, out16, out32, detail16 = net(im)
        #
        # if (not use_boundary_2) and use_boundary_4 and (not use_boundary_8):
        #     out, out16, out32, detail8= net(im)

        # loss：根据不同的输出计算损失，二元交叉熵损失和dice损失
        lossp = criteria_p(out, lb)
        loss2 = criteria_8(out8, lb)
        loss3 = criteria_16(out16, lb)

        boundery_bce_loss = 0.
        boundery_dice_loss = 0.

        # 使用不同尺度输出的边界轮廓所产生的细节损失：boundery_bce_loss + boundery_dice_loss
        # if use_boundary_2:
        #     # if dist.get_rank()==0:
        #     #     print('use_boundary_2')
        #     boundery_bce_loss4, boundery_dice_loss4 = boundary_loss_func(detail4, lb)
        #     boundery_bce_loss += boundery_bce_loss4
        #     boundery_dice_loss += boundery_dice_loss4
        #
        # if use_boundary_8:
        #     boundery_bce_loss8, boundery_dice_loss8 = boundary_loss_func(detail8, lb)
        #     boundery_bce_loss += boundery_bce_loss8
        #     boundery_dice_loss += boundery_dice_loss8
        #
        # if use_boundary_16:
        #     boundery_bce_loss16, boundery_dice_loss16 = boundary_loss_func(detail16, lb)
        #     boundery_bce_loss += boundery_bce_loss16
        #     boundery_dice_loss += boundery_dice_loss16

        if use_boundary_4:
            boundery_bce_loss4, boundery_dice_loss4 = boundary_loss_func(out, lb)
            boundery_bce_loss += boundery_bce_loss4
            boundery_dice_loss += boundery_dice_loss4
            b4 = boundery_bce_loss4 + boundery_dice_loss4
            boundary_loss4.append(b4.item())

        if use_boundary_8:
            boundery_bce_loss8, boundery_dice_loss8 = boundary_loss_func(out8, lb)
            boundery_bce_loss += boundery_bce_loss8
            boundery_dice_loss += boundery_dice_loss8
            b8 = boundery_bce_loss8 + boundery_dice_loss8
            boundary_loss8.append(b8.item())

        if use_boundary_16:
            boundery_bce_loss16, boundery_dice_loss16 = boundary_loss_func(out16, lb)
            boundery_bce_loss += boundery_bce_loss16
            boundery_dice_loss += boundery_dice_loss16
            b16 = boundery_bce_loss16 + boundery_dice_loss16
            boundary_loss16.append(b16.item())

        loss_s4.append(lossp.item())
        loss_s8.append(loss2.item())
        loss_s16.append(loss3.item())
        loss = lossp + loss2 + loss3 + 0.8 * (boundery_bce_loss + boundery_dice_loss)

        loss.backward()
        optim.step()

        loss_avg.append(loss.item())

        loss_boundery_bce.append(boundery_bce_loss.item())
        loss_boundery_dice.append(boundery_dice_loss.item())

        ## print training log message，训练日志
        if (it + 1) % msg_iter == 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            lr = optim.lr
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int((max_iter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds=eta))

            loss_boundery_bce_avg = sum(loss_boundery_bce) / len(loss_boundery_bce)
            loss_boundery_dice_avg = sum(loss_boundery_dice) / len(loss_boundery_dice)
            msg = ', '.join([
                'it: {it}/{max_it}',
                'lr: {lr:4f}',
                'loss: {loss:.4f}',
                'boundery_bce_loss: {boundery_bce_loss:.4f}',
                'boundery_dice_loss: {boundery_dice_loss:.4f}',
                'eta: {eta}',
                'time: {time:.4f}',
            ]).format(
                it=it + 1,
                max_it=max_iter,
                lr=lr,
                loss=loss_avg,
                boundery_bce_loss=loss_boundery_bce_avg,
                boundery_dice_loss=loss_boundery_dice_avg,
                time=t_intv,
                eta=eta
            )

            logger.info(msg)
            loss_avg = []
            loss_boundery_bce = []
            loss_boundery_dice = []
            st = ed
            # print(boundary_loss_func.get_params())
        # 训练多代使用验证集进行验证
        if (it + 1) % save_iter_sep == 0:  # and it != 0:

            ## model
            logger.info('evaluating the model ...')
            logger.info('setup and restore model')

            net.eval()

            # ## evaluator
            logger.info('compute the mIOU')
            with torch.no_grad():
                # 计算miou
                single_scale1 = MscEvalV0()
                mIOU50 = single_scale1(net, dlval, n_classes)

                single_scale2 = MscEvalV0(scale=0.75)
                mIOU75 = single_scale2(net, dlval, n_classes)

            # 保存模型路径
            # save_pth = osp.join(save_pth_path, 'model_iter{}_mIOU50_{}_mIOU75_{}.pth'
            #                     .format(it + 1, str(round(mIOU50, 4)), str(round(mIOU75, 4))))
            #
            # state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
            #
            # torch.save(state, save_pth)
            #
            # logger.info('training iteration {}, model saved to: {}'.format(it + 1, save_pth))

            # 如果miou>maxmiou，保存模型
            if mIOU50 > maxmIOU50:
                maxmIOU50 = mIOU50
                save_pth = osp.join(save_pth_path, 'model_iter{}_maxmIOU50_{}.pth'.format(it + 1, str(round(maxmIOU50, 4))))
                state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()

                torch.save(state, save_pth)

                logger.info('max mIOU model saved to: {}'.format(save_pth))

            if mIOU75 > maxmIOU75:
                maxmIOU75 = mIOU75
                save_pth = osp.join(save_pth_path, 'model_iter{}_maxmIOU75_{}.pth'.format(it + 1, str(round(maxmIOU75, 4))))
                state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
                torch.save(state, save_pth)
                logger.info('max mIOU model saved to: {}'.format(save_pth))

            logger.info('mIOU50 is: {}, mIOU75 is: {}'.format(mIOU50, mIOU75))
            logger.info('maxmIOU50 is: {}, maxmIOU75 is: {}.'.format(maxmIOU50, maxmIOU75))

            net.train()

        # write segmentation loss into txt
        with open("D:/STDC-Seg-master/loss/learn_para_oi/loss_seg4.txt", 'w') as seg_loss4:
            seg_loss4.write(str(loss_s4))
        with open("D:/STDC-Seg-master/loss/learn_para_oi/loss_seg8.txt", 'w') as seg_loss8:
            seg_loss8.write(str(loss_s8))
        with open("D:/STDC-Seg-master/loss/learn_para_oi/loss_seg16.txt", 'w') as seg_loss16:
            seg_loss16.write(str(loss_s16))

        # write boundary loss into txt
        with open("D:/STDC-Seg-master/loss/learn_para_oi/loss_boundary4.txt", 'w') as bound_loss4:
            bound_loss4.write(str(boundary_loss4))
        with open("D:/STDC-Seg-master/loss/learn_para_oi/loss_boundary8.txt", 'w') as bound_loss8:
            bound_loss8.write(str(boundary_loss8))
        with open("D:/STDC-Seg-master/loss/learn_para_oi/loss_boundary16.txt", 'w') as bound_loss16:
            bound_loss16.write(str(boundary_loss16))

        with open("D:/STDC-Seg-master/loss/learn_para_oi/loss_total.txt", 'w') as loss_total:
            loss_total.write(str(loss_avg))

    ## dump the final model
    save_pth = osp.join(save_pth_path, 'model_final.pth')
    net.cpu()
    state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    torch.save(state, save_pth)
    logger.info('training done, model saved to: {}'.format(save_pth))
    print('epoch: ', epoch)

if __name__ == "__main__":
    # 配置GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = '2, 3'
    train()
