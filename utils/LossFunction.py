import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        probas = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probas, 1 - probas)
        at = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = -at * (1 - pt) ** self.gamma * torch.log(pt + 1e-8)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def bce_loss(prediction, target):
    """
    计算 BCE 损失（使用 sigmoid 激活后的概率）
    """
    bce = nn.BCELoss()
    return bce(prediction, target)


def dice_loss(prediction, target, smooth=1e-6):
    intersection = torch.sum(prediction * target, dim=(1, 2, 3))
    pred_sum = torch.sum(prediction, dim=(1, 2, 3))
    target_sum = torch.sum(target, dim=(1, 2, 3))
    d_loss = 1 - (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
    return d_loss.mean()

def combined_loss(predict_logits, target,
                  lambda_focal, lambda_dice,
                  #lambda_iou,
                  alpha=0.25, gamma=2.0):
    """
    综合损失函数：Focal + Dice + IoU，加权组合
    """
    prob = torch.sigmoid(predict_logits)

    loss_focal = FocalLoss(alpha=alpha, gamma=gamma)(predict_logits, target)
    loss_dice = dice_loss(prob, target)

    loss_total = (
        lambda_focal * loss_focal +
        lambda_dice * loss_dice 
    )

    loss_dict = {
        'focal': loss_focal.item(),
        'dice': loss_dice.item(),
        'total': loss_total.item()
    }

    return loss_total, loss_dict

def compute_road_iou(pred_list, target_list, threshold=0.35, eps=1e-6):
    """
    对整个验证集的预测图像和真实图像求整体 TP/FP/FN 后计算一次总体 IoU。

    pred_list: list of torch.Tensor (每个元素形状为 NCHW 或 HW)
    target_list: list of torch.Tensor
    """
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0

    for pred, target in zip(pred_list, target_list):
        pred_bin = (torch.sigmoid(pred) > threshold).float()
        target_bin = (target > threshold).float()

        tp = (pred_bin * target_bin).sum()
        fp = (pred_bin * (1 - target_bin)).sum()
        fn = ((1 - pred_bin) * target_bin).sum()

        total_tp += tp
        total_fp += fp
        total_fn += fn

    iou = total_tp / (total_tp + total_fp + total_fn + eps)
    return iou.item()