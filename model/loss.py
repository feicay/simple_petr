import torch
from torch import nn
from torch.nn import functional as F 
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.reduction = reduction

    def focal_loss(self, x, y, eps=1e-9):
        alpha = 0.25
        gamma = 2.0
        p = x.sigmoid()
        weight = (p - y).pow(gamma)
        alpha_t = alpha * y + (1 - alpha) * (1 - y)
        weight *= alpha_t
        p = p.to(torch.float32)
        eps = 1e-9
        loss = y * (0 - torch.log(p + eps)) + \
               (1.0 - y) * (0 - torch.log(1.0 - p + eps))
        loss *= weight
        num_pos = y.sum().clamp(min=1.0)
        loss = loss.sum()
        if self.reduction == 'mean':
            loss = loss / num_pos
        return loss

    def forward(self, keypoint_pred, keypoint_truth):
        loss = self.focal_loss(keypoint_pred, keypoint_truth)
        return loss

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha=0.25, gamma=2):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.sum() / num_boxes
