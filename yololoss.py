import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class Loss(nn.Module):
    def __init__(self, lambda_coord, lambda_noobj):
        super(Loss, self).__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, pred_tensor, target_tensor):
        """
            输入两个变量分别为，通过网络预测的张量和实际标签张量。两个张量的尺寸均为[batch_size，s，s，95]
        batch_size为批量处理的图像个数，s为网格尺寸，95就是5个box参数加90类，前5个参数为box属性。
        """
        """
            计算网格是否包含有目标，应从实际标签张量的box属性第5各参数来判定，该值表征某网格某box的预测概率为1
        逻辑mask应与原tensor尺寸相同，只包含0-1两个值，表示原tensor对应位置是否满足条件。
        """
        # 具有目标的标签逻辑索引
        coo_mask = target_tensor[:, :, :, 4] > 0
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)
        # print("coomask", coo_mask)
        # 没有目标的标签逻辑索引
        noo_mask = target_tensor[:, :, :, 4] == 0
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)
        """
            计算每张图像中，每个目标对应的，最大IOU的预测box的定位误差、confidence误差、类别误差
            及每个不含目标的box的confidence误差。
        """
        xy_loss = 0
        wh_loss = 0
        con_obj_loss = 0
        nocon_obj_loss = 0
        for i in range(pred_tensor.size()[0]):
            # 提取真实box属性
            # print("size", target_tensor[i][coo_mask[i]].shape)
            coo_targ = target_tensor[i][coo_mask[i]].view(-1, 30)
            box_targ_s = coo_targ[:, :5].contiguous().view(-1, 5)  # 单独提取出目标数组

            # 提取预测box属性
            box_targ = target_tensor[i, :, :, :5].view(-1, 5)
            box_pred = pred_tensor[i, :, :, :5].view(-1, 5)
            # 计算IOU张量，尺寸为N×M。
            if box_targ.size()[0] != 0:
                # print("box_pred", box_pred)
                # print("mask",  coo_mask[i, :, :, 1])
                iou = self.cal_iou(box_targ_s, box_pred, coo_mask[i, :, :, 1])  # box_targ  大小为 实际框个数*5，box_pred为 预测框大小49*5，coomask为7*7的真假值
                # print("iou", iou)
                #iou tensor([[0.3177],
                    # [0.1949],
                    # [0.3190],
                    # [0.3465],
                    # [0.6593],
                    # [0.3211],
                    # [0.1587],
        
                # 找到每列的最大值及对应行，即对应的真实box的最大IOU及box序号
                max_iou, max_sort = torch.max(iou, dim=0)
                # print("maxiou",  max_iou)   # maxiou tensor([0.7318], device='cuda:0', grad_fn=<MaxBackward0>)
                # print("maxsort0", max_sort)  # maxsort tensor([46], device='cuda:0')

                # 计算定位误差
                xy_loss += F.mse_loss(box_pred[max_sort, :2], box_targ[max_sort, :2], reduction='sum')  # 前两个值为坐标误差
                wh_loss += F.mse_loss(box_pred[max_sort, 2:4].sqrt(), box_targ[max_sort, 2:4].sqrt(), reduction='sum')  # 前三四个值为宽高误差
                # print("wh_loss", wh_loss)

                # 计算confidence误差
                """
                    confidence误差，应为每一个网格内的每一个box的置信概率乘以该box的IOU值，该误差包括两个部分，一个是对于
                包含目标的box，上面已经计算出IOU值，可以直接进行计算，但对于另一部分，也就是不包含目标的box，由于其不包含
                box属性，所以真实confidence应该取0。对于预测的IOU可直接设为1。在计算损失函数时，为计算方便实际可分别设置
                为ones张量和zeros张量。
                """

                # 包含目标的box confidence误差
                con_obj_c = box_pred[max_sort][:, 4] * max_iou
                con_obj_loss += F.mse_loss(con_obj_c, torch.ones_like(con_obj_c), reduction='sum')

                # 不含目标的box confidence误差
                no_sort = torch.ones(box_pred.size()[0]).byte()
                no_sort[max_sort] = 0
                no_sort = no_sort.bool()  # 太狗了
                nocon_obj_c = box_pred[no_sort][:, 4]
                nocon_obj_loss += F.mse_loss(nocon_obj_c, torch.zeros_like(nocon_obj_c), reduction='sum')

        # 计算类别误差
        """
            由于类别是通过网格来确定的，每一个网格无论有几个box，一个所属类概率。
            在计算类别误差时，只对目标中心落在该其中的网格进行计算。
        """
        # coo_mask 表示在整个张量中，包含目标的网格点索引，所以可以不对每一个bitch进行分别计算，直接整体求和
        con_pre_class = pred_tensor[coo_mask].view(-1, 30)[:, 10:]
        con_tar_class = target_tensor[coo_mask].view(-1, 30)[:, 10:]
        con_class_loss = F.mse_loss(con_pre_class, con_tar_class, reduction='sum')
        
        import numpy as np
        # 总损失函数求和
        loss_total = (self.lambda_coord * (xy_loss + wh_loss) + con_obj_loss
                      + self.lambda_noobj  * nocon_obj_loss + con_class_loss)/pred_tensor.size()[0]
        return loss_total

    def cal_iou(self, box_targ, box_pred, mask):
        # 计算box数量
        M = box_targ.size()[0]
        N = box_pred.size()[0]
        # 转化box参数，转化为统一坐标
        # row = torch.arange(7, dtype=torch.float).unsqueeze(-1).expand_as(mask)[mask].cuda()
        # col = torch.arange(7, dtype=torch.float).unsqueeze(0).expand_as(mask)[mask].cuda()
        # box_targ[:, 0] = col / 7 + box_targ[:, 0] * 1 /7
        # box_targ[:, 1] = row / 7 + box_targ[:, 1] * 1 / 7

        exboxM = box_targ.unsqueeze(0).expand(N, M, 5)
        exboxN = box_pred.unsqueeze(1).expand(N, M, 5)
        dxy = (exboxM[:, :, :2] - exboxN[:, :, :2])
        swh = (exboxM[:, :, 2:4] + exboxN[:, :, 2:4])
        s_inter = swh / 2 - dxy.abs()
        s_inter = (s_inter[:, :, 0] * s_inter[:, :, 1]).clamp(min=0)
        s_union = exboxM[:, :, 2] * exboxM[:, :, 3] + exboxN[:, :, 2] * exboxN[:, :, 3] - s_inter
        iou = s_inter / s_union
        # print("s_inter", s_inter)
        # print("s_union", s_union)
        return iou




