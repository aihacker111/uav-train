# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from torch.nn import Parameter
from typing import Tuple

from .utils import _tranpose_and_gather_feat
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.functional import mse_loss
from functools import partial
import torch.distributed as dist
from ..utils.utils import bbox_iou


def _slow_neg_loss(pred, gt):
    '''focal loss from CornerNet'''
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    loss = 0
    pos_pred = pred[pos_inds]
    neg_pred = pred[neg_inds]

    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
    neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if pos_pred.nelement() == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _neg_loss(pred, gt):
    """ Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    """
    pos_inds = gt.eq(1).float()  # ground truth为1的表示正样本像素点
    neg_inds = gt.lt(1).float()  # ground truth小于1的表示负样本点

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos

    return loss


def _not_faster_neg_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    num_pos = pos_inds.float().sum()
    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    trans_pred = pred * neg_inds + (1 - pred) * pos_inds
    weight = neg_weights * neg_inds + pos_inds
    all_loss = torch.log(1 - trans_pred) * torch.pow(trans_pred, 2) * weight
    all_loss = all_loss.sum()

    if num_pos > 0:
        all_loss /= num_pos
    loss -= all_loss
    return loss


def _slow_reg_loss(regr, gt_regr, mask):
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr = regr[mask]
    gt_regr = gt_regr[mask]

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, reduction='sum')
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


def _reg_loss(regr, gt_regr, mask):
    ''' L1 regression loss
      Arguments:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects)
    '''
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()

    regr = regr * mask
    gt_regr = gt_regr * mask

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, reduction='sum')
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


class FocalLoss(nn.Module):
    """
    nn.Module warpper for focal loss
    """

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)


# 自定义用于多类分类的Focal loss
class McFocalLoss(nn.Module):
    def __init__(self,
                 num_classes,
                 device,
                 use_alpha=False,
                 alpha=None,
                 gamma=1.5,
                 size_average=True):
        """
        :param num_classes:
        :param device:
        :param alpha:
        :param gamma:
        :param use_alpha:
        :param size_average:
        """
        super(McFocalLoss, self).__init__()

        self.num_classes = num_classes
        self.dev = device
        self.alpha = alpha
        self.gamma = gamma

        if use_alpha:
            self.alpha = torch.tensor(alpha).to(self.dev)

        self.softmax = nn.Softmax(dim=1)
        self.use_alpha = use_alpha
        self.size_average = size_average

    def forward(self, pred, target):
        prob = self.softmax(pred.view(-1, self.num_classes))
        prob = prob.clamp(min=0.0001, max=1.0)

        target_ = torch.zeros(target.size(0), self.num_classes).to(self.dev)
        target_.scatter_(1, target.view(-1, 1).long(), 1.0)

        if self.use_alpha:
            batch_loss = - self.alpha.double() * torch.pow(1 - prob, self.gamma).double() \
                         * prob.log().double() * target_.double()
        else:
            batch_loss = - torch.pow(1 - prob, self.gamma).double() * prob.log().double() * target_.double()

        batch_loss = batch_loss.sum(dim=1)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss


# Arc loss的FC layer用于细粒度分类或Re-ID
class ArcMarginFc(nn.Module):
    r"""
    Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """

    def __init__(self,
                 in_features,
                 out_features,
                 device,
                 s=30.0,
                 m=0.50,
                 easy_margin=False):
        """
        ArcMargin
        :type in_features: int
        :type out_features: int
        :param in_features:
        :param out_features:
        :param s:
        :param m:
        :param easy_margin:
        """
        super(ArcMarginFc, self).__init__()

        self.device = device
        self.in_dim = in_features
        self.out_dim = out_features
        print('=> in dim: %d, out dim: %d' % (self.in_dim, self.out_dim))

        self.s = s
        self.m = m

        # 根据输入输出dim确定初始化权重
        self.weight = Parameter(torch.FloatTensor(self.out_dim, self.in_dim))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # L2 normalize and calculate cosine
        cosine = F.linear(F.normalize(input, p=2), F.normalize(self.weight, p=2))

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # phi: cos(θ+m)
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=self.device)  # device='cuda'
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        # you can use torch.where if your torch.__version__ >= 0.4
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        # print(output)

        return output


# -----Circle loss
def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss


class Registry(object):
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def _register_module(self, module_class):
        """Register a module.
        Args:
            module (:obj:`nn.Module`): Module to be registered.
        """
        if not issubclass(module_class, nn.Module):
            raise TypeError(
                'module must be a child of nn.Module, but got {}'.format(
                    module_class))
        module_name = module_class.__name__
        if module_name in self._module_dict:
            raise KeyError('{} is already registered in {}'.format(
                module_name, self.name))
        self._module_dict[module_name] = module_class

    def register_module(self, cls):
        self._register_module(cls)
        return cls


BACKBONES = Registry('backbone')
NECKS = Registry('neck')
ROI_EXTRACTORS = Registry('roi_extractor')
SHARED_HEADS = Registry('shared_head')
HEADS = Registry('head')
LOSSES = Registry('loss')
DETECTORS = Registry('detector')


def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    bin_label_weights = label_weights.view(-1, 1).expand(
        label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


@LOSSES.register_module
class GHMC(nn.Module):
    """GHM Classification Loss.
    Details of the theorem can be viewed in the paper
    "Gradient Harmonized Single-stage Detector".
    https://arxiv.org/abs/1811.05181
    Args:
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        use_sigmoid (bool): Can only be true for BCE based loss now.
        loss_weight (float): The weight of the total GHM-C loss.
    """

    def __init__(
            self,
            bins=10,
            momentum=0,
            use_sigmoid=True,
            loss_weight=1.0):
        """
        :param bins:
        :param momentum:
        :param use_sigmoid:
        :param loss_weight:
        """
        super(GHMC, self).__init__()

        self.bins = bins
        self.momentum = momentum
        self.edges = torch.arange(bins + 1).float().cuda() / bins
        self.edges[-1] += 1e-6

        if momentum > 0:
            self.acc_sum = torch.zeros(bins).cuda()

        self.use_sigmoid = use_sigmoid
        if not self.use_sigmoid:
            raise NotImplementedError
        self.loss_weight = loss_weight

    def forward(self, pred, target, label_weight, *args, **kwargs):
        """Calculate the GHM-C loss.
        Args:
            pred (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            target (float tensor of size [batch_num, class_num]):
                Binary class target for each sample.
            label_weight (float tensor of size [batch_num, class_num]):
                the value is 1 if the sample is valid and 0 if ignored.
        Returns:
            The gradient harmonized loss.
        """
        # the target should be binary class label
        if pred.dim() != target.dim():
            target, label_weight = _expand_binary_labels(
                target, label_weight, pred.size(-1))

        target, label_weight = target.float(), label_weight.float()
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred)

        # gradient length
        g = torch.abs(pred.sigmoid().detach() - target)

        valid = label_weight > 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                                      + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        loss = F.binary_cross_entropy_with_logits(pred, target, weights, reduction='sum') / tot

        return loss * self.loss_weight


@LOSSES.register_module
class GHMR(nn.Module):
    """GHM Regression Loss.
    Details of the theorem can be viewed in the paper
    "Gradient Harmonized Single-stage Detector"
    https://arxiv.org/abs/1811.05181
    Args:
        mu (float): The parameter for the Authentic Smooth L1 loss.
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        loss_weight (float): The weight of the total GHM-R loss.
    """

    def __init__(
            self,
            mu=0.02,
            bins=10,
            momentum=0,
            loss_weight=1.0):
        super(GHMR, self).__init__()
        self.mu = mu
        self.bins = bins
        self.edges = torch.arange(bins + 1).float().cuda() / bins
        self.edges[-1] = 1e3
        self.momentum = momentum
        if momentum > 0:
            self.acc_sum = torch.zeros(bins).cuda()
        self.loss_weight = loss_weight

    def forward(self, pred, target, label_weight, avg_factor=None):
        """Calculate the GHM-R loss.
        Args:
            pred (float tensor of size [batch_num, 4 (* class_num)]):
                The prediction of box regression layer. Channel number can be 4
                or 4 * class_num depending on whether it is class-agnostic.
            target (float tensor of size [batch_num, 4 (* class_num)]):
                The target regression values with the same size of pred.
            label_weight (float tensor of size [batch_num, 4 (* class_num)]):
                The weight of each sample, 0 if ignored.
        Returns:
            The gradient harmonized loss.
        """
        mu = self.mu
        edges = self.edges
        mmt = self.momentum

        # ASL1 loss
        diff = pred - target
        loss = torch.sqrt(diff * diff + mu * mu) - mu

        # gradient length
        g = torch.abs(diff / torch.sqrt(mu * mu + diff * diff)).detach()
        weights = torch.zeros_like(g)

        valid = label_weight > 0
        tot = max(label_weight.float().sum().item(), 1.0)
        n = 0  # n: valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                n += 1
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                                      + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
        if n > 0:
            weights /= n

        loss = loss * weights
        loss = loss.sum() / tot
        return loss * self.loss_weight


class RegLoss(nn.Module):
    '''Regression loss for an output tensor
    Smooth L1 loss
      Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    '''

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        loss = _reg_loss(pred, target, mask)

        return loss


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        """
        :param output:
        :param mask:
        :param ind:
        :param target:
        :return:
        """
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)  # 计算平均值
        return loss


class NormRegL1Loss(nn.Module):
    def __init__(self):
        super(NormRegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        """
        :param output:
        :param mask:
        :param ind:
        :param target:
        :return:
        """
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        pred = pred / (target + 1e-4)
        target = target * 0 + 1
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss


class RegWeightedL1Loss(nn.Module):
    def __init__(self):
        super(RegWeightedL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        # ind = ind.cpu().numpy()
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        return loss


class BinRotLoss(nn.Module):
    def __init__(self):
        super(BinRotLoss, self).__init__()

    def forward(self, output, mask, ind, rotbin, rotres):
        pred = _tranpose_and_gather_feat(output, ind)
        loss = compute_rot_loss(pred, rotbin, rotres, mask)
        return loss


def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='elementwise_mean')


# TODO: weight
def compute_bin_loss(output, target, mask):
    mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction='elementwise_mean')


def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    # import pdb; pdb.set_trace()
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
            valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
            valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
            valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
            valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
        margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3, mutual_flag=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        # dist.addmm_(1, -2, inputs, inputs.t())
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][1 + (-1)*mask[i]].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        if self.mutual:
            return loss, dist
        return loss

def dice_loss(input, target):
    # xxx = target.detach().cpu().numpy()[0]
    # #
    # plt.imshow(xxx, cmap='viridis')  # cmap??????????
    # plt.show()
    
    smooth = 1.
    iflat = input.contiguous().view(-1)
    tflat = target.contiguous().view(-1)

    # xxx = iflat.detach().cpu().numpy()
    # is_all_zeros = np.all(xxx == 0)
    # xxx = np.isnan(xxx)
    # # 使用 np.sum() 计算 NaN 的数量
    # nan_count = np.sum(xxx)
    # 
    # if nan_count > 0 or is_all_zeros:
    #     print('!!!!!')  # 输出：1

    # xxx = tflat.detach().cpu().numpy()
    # is_all_zeros = np.all(xxx == 0)
    # xxx = np.isnan(xxx)
    # # 使用 np.sum() 计算 NaN 的数量
    # nan_count = np.sum(xxx)

    # if nan_count > 0 or is_all_zeros:
    #     print('????')  # 输出：1
    # count = np.count_nonzero(xxx == 1)
    # print(iflat)
    # if count == 0:
    #     print(count, '!!!!!')

    intersection = (iflat * tflat).sum()
    # print('\n')
    # print('inter:',intersection)
    # print('if:', (iflat * iflat).sum())
    # print('tf:', (tflat * tflat).sum())
    result = 1. - ((2. * intersection + smooth) / ((iflat * iflat).sum() + (tflat * tflat).sum() + smooth)) + 1e-5
    # print('result:', result)
    return result

class DiceLoss(nn.Module):
    def __init__(self,feat_channel):
        super(DiceLoss, self).__init__()
        self.feat_channel=feat_channel

    def forward(self, seg_feat, conv_weight, mask, ind, target, batch_num_obj):

        # xx = mask.cpu().numpy()
        # xxx = target[1][0].cpu().numpy()
        #
        # plt.imshow(xxx, cmap='viridis')  # cmap参数用于指定颜色映射
        # plt.show()

        mask_loss= 0.
        batch_size = seg_feat.size(0)

        # 根据id 选择权重
        weight = _tranpose_and_gather_feat(conv_weight, ind) # (2, 256, 169) ￥ (2,50,169) k=50?

        # 生成网格
        h,w = seg_feat.size(-2),seg_feat.size(-1)
        x,y = ind%w , ind/w
        x_range = torch.arange(w).float().to(device=seg_feat.device)
        y_range = torch.arange(h).float().to(device=seg_feat.device)
        y_grid, x_grid = torch.meshgrid([y_range, x_range])         # h,w的网格

        processed_batches = 0

        batch_mask_feat = []

        for i in range(batch_size):
            # 提取当前对象的数量
            num_obj = batch_num_obj.detach().cpu().numpy()
            num_obj = int(num_obj[i])

            # print('\n')
            # print('num_obj:', num_obj)

            if num_obj == 0:
                batch_mask_feat.append(torch.zeros(1,1,h,w))
                print("No objects in this batch, skipping.")

                continue

            conv1w,conv1b,conv2w,conv2b,conv3w,conv3b= \
                torch.split(weight[i,:num_obj],
                            [(self.feat_channel+2)*self.feat_channel,
                                                self.feat_channel,
                                                self.feat_channel**2,
                                                self.feat_channel,
                                                self.feat_channel,1] ,dim=-1)
            # 计算偏移 ind为目标位置索引
            y_rel_coord = (y_grid[None,None] - y[i,:num_obj].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float())/128.
            x_rel_coord = (x_grid[None,None] - x[i,:num_obj].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float())/128.

            feat = seg_feat[i][None].repeat([num_obj,1,1,1])
            feat = torch.cat([feat,x_rel_coord, y_rel_coord],dim=1).view(1,-1,h,w) #1,10,88,160

            conv1w=conv1w.contiguous().view(-1,self.feat_channel+2,1,1)
            conv1b=conv1b.contiguous().flatten()

            feat = F.conv2d(feat,conv1w,conv1b,groups=num_obj).relu()

            conv2w=conv2w.contiguous().view(-1,self.feat_channel,1,1)
            conv2b=conv2b.contiguous().flatten()
            feat = F.conv2d(feat,conv2w,conv2b,groups=num_obj).relu()

            conv3w=conv3w.contiguous().view(-1,self.feat_channel,1,1)
            conv3b=conv3b.contiguous().flatten()
            feat = F.conv2d(feat,conv3w,conv3b,groups=num_obj).sigmoid().squeeze()

            # print(feat.size())

            true_mask = mask[i,:num_obj,None,None].float()
            mask_loss += dice_loss(feat*true_mask,target[i][:num_obj]*true_mask)

            batch_mask_feat.append(feat*true_mask)

            processed_batches += 1

        if processed_batches > 0:
            average_mask_loss = mask_loss / processed_batches
        else:
            average_mask_loss = 0.0

        return average_mask_loss, batch_mask_feat


def extract_patch_(tensor, center_x, center_y, patch_size=16):
    """
    ???????????????? 16x16 ????

    ??:
    - tensor: ???????? (2, H, W)
    - center_x: ???? x ??
    - center_y: ???? y ??
    - patch_size: ??????????? 16

    ??:
    - ??? 16x16 ?????? (2, 16, 16)
    """
    # ?????????
    center_x = torch.round(center_x).int().item()
    center_y = torch.round(center_y).int().item()
    half_size = int(patch_size // 2)

    left = max(0, center_x - half_size)
    top = max(0, center_y - half_size)
    right = min(tensor.size(2), center_x + half_size)
    bottom = min(tensor.size(1), center_y + half_size)

    # ????
    patch = tensor[:, top:bottom, left:right]

    # ????????? 16x16?????
    pad_left = max(0, (half_size - center_x))
    pad_right = max(0, center_x + half_size - tensor.size(2))
    pad_top = max(0, half_size - center_y)
    pad_bottom = max(0, center_y + half_size - tensor.size(1))

    patch = torch.nn.functional.pad(patch, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0)

    if patch.shape[1] != patch_size or patch.shape[2] != patch_size:
        raise ValueError(f"require tensor size is {patch_size}, actual tensor size is {tensor.dim()}")

    return patch

def extract_patch(tensor, center_x_batch, center_y_batch, patch_size=16):

    patches = []
    for i in range(len(center_x_batch)):
        center_x = center_x_batch[i]
        center_y = center_y_batch[i]

        center_x = torch.round(center_x).int().item()
        center_y = torch.round(center_y).int().item()
        half_size = int(patch_size // 2)

        left = max(0, center_x - half_size)
        top = max(0, center_y - half_size)
        right = min(tensor.size(2), center_x + half_size)
        bottom = min(tensor.size(1), center_y + half_size)

        # ????
        patch = tensor[:, top:bottom, left:right]

        # ????????? 16x16?????
        pad_left = max(0, (half_size - center_x))
        pad_right = max(0, center_x + half_size - tensor.size(2))
        pad_top = max(0, half_size - center_y)
        pad_bottom = max(0, center_y + half_size - tensor.size(1))
        if pad_left != 0 or pad_right != 0 or pad_top != 0 or  pad_bottom != 0:
            patch = torch.nn.functional.pad(patch, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0)

        if patch.shape[1] != patch_size or patch.shape[2] != patch_size:
            raise ValueError(f"张量维度要求为{patch_size}，实际维度为{tensor.dim()}")
        patches.append(patch)

    patches = torch.stack(patches, dim=0)

    return patches
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

class Clip_loss_class(nn.Module):
    def __init__(self, device):
        super(Clip_loss_class, self).__init__()

        self.classes = [
            'pedestrian',
            'people',
            'bicycle',
            'car',
            'van',
            'truck',
            'tricycle',
            'awning-tricycle',
            'bus',
            'motor']
        # self.extra_classes = ['other']

        self.device = device
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        self.clip_text_fc = nn.Sequential(nn.Linear(512, 256, bias=False), nn.LayerNorm(256), nn.SiLU(True),
                                           nn.Linear(256, 256, bias=False), nn.LayerNorm(256), nn.SiLU(True),
                                           nn.Dropout(),
                                           nn.Linear(256, 256, bias=False)
                                           )

        self.feature_head = nn.Sequential(
            nn.Linear(256, 256, bias=False), nn.LayerNorm(256), nn.SiLU(True),
            nn.Dropout(),
            nn.Linear(256, 256, bias=False))

        self.clip_preprocess_mean = torch.tensor(self.preprocess.transforms[-1].mean).view(1, 3, 1, 1).to(self.device)
        self.clip_preprocess_std = torch.tensor(self.preprocess.transforms[-1].std).view(1, 3, 1, 1).to(self.device)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.text_inputs = torch.cat([clip.tokenize(f'a photo of the {c}') for c in self.classes]).to(self.device)

    def normalize_batch_inputs(self, batch_inputs):
        batch_inputs = (batch_inputs - self.clip_preprocess_mean) / self.clip_preprocess_std.clamp_min(1e-8)
        return batch_inputs

    def forward(self, images, num_obj, head_features, cls_id_ids, location_class):

        batch_size = images.size(0)

        all_num_obj = 0.

        clip_text_features_ = []
        head_feature_patch_ = []

        with torch.no_grad():
            clip_text_inputs = self.clip_model.encode_text(self.text_inputs).float()
        for i in range(batch_size):

            all_num_obj += float(num_obj[i])
            if int(num_obj[i]) == 0:
                print("\n")
                print("No objects in this batch, skipping clip loss.")
                continue

            head_feature = head_features[i]
            feature_w = location_class[i][0:num_obj[i]][:, 0].long()
            feature_h = location_class[i][0:num_obj[i]][:, 1].long()
            head_feature_patch_.append(head_feature[:, feature_h, feature_w].T)

            valid_text_inputs = clip_text_inputs[cls_id_ids[i][0:num_obj[i]]]
            clip_text_features_.append(valid_text_inputs)

        clip_text_features_ = torch.cat(clip_text_features_, dim=0)
        clip_text_features_ = self.clip_text_fc(clip_text_features_)

        head_feature_patch_ = torch.cat(head_feature_patch_, dim=0)
        head_feature_patch_ = self.feature_head(head_feature_patch_)

        clip_text_features_ = clip_text_features_ / clip_text_features_.norm(dim=1, keepdim=True)
        head_feature_patch_ = head_feature_patch_ / head_feature_patch_.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * head_feature_patch_ @ clip_text_features_.t()
        logits_per_text = logits_per_image.t()

        image_loss = contrastive_loss(logits_per_image)
        text_loss = contrastive_loss(logits_per_text)

        # clip_loss = (image_loss + text_loss) / all_num_obj * 0.5
        clip_loss = (text_loss) / all_num_obj * 0.5
        return clip_loss

class Clip_loss_class_patch(nn.Module):
    def __init__(self, device):
        super(Clip_loss_class_patch, self).__init__()

        self.classes = [
            'pedestrian',
            'people',
            'bicycle',
            'car',
            'van',
            'truck',
            'tricycle',
            'awning-tricycle',
            'bus',
            'motor']
        # self.extra_classes = ['other']

        self.device = device
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        self.clip_text_fc = nn.Sequential(nn.Linear(512, 256, bias=False), nn.LayerNorm(256), nn.SiLU(True),
                                           nn.Linear(256, 256, bias=False), nn.LayerNorm(256), nn.SiLU(True),
                                           nn.Dropout(),
                                           nn.Linear(256, 256, bias=False)
                                           )

        self.feature_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten())

        self.clip_preprocess_mean = torch.tensor(self.preprocess.transforms[-1].mean).view(1, 3, 1, 1).to(self.device)
        self.clip_preprocess_std = torch.tensor(self.preprocess.transforms[-1].std).view(1, 3, 1, 1).to(self.device)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.text_inputs = torch.cat([clip.tokenize(f'a photo of the {c}') for c in self.classes]).to(self.device)

    def normalize_batch_inputs(self, batch_inputs):
        batch_inputs = (batch_inputs - self.clip_preprocess_mean) / self.clip_preprocess_std.clamp_min(1e-8)
        return batch_inputs

    def forward(self, images, num_obj, head_features, cls_id_ids, location_class):

        batch_size = images.size(0)
        patch_size = 8

        all_num_obj = 0.

        clip_text_features_ = []
        head_feature_patch_ = []

        with torch.no_grad():
            clip_text_inputs = self.clip_model.encode_text(self.text_inputs).float()
        for i in range(batch_size):

            all_num_obj += float(num_obj[i])
            if int(num_obj[i]) == 0:
                print("\n")
                print("No objects in this batch, skipping clip loss.")
                continue

            valid_head_features = head_features[i]
            feature_x = location_class[i][0:num_obj[i]][:, 0]
            feature_y = location_class[i][0:num_obj[i]][:, 1]
            head_feature_patches = extract_patch(valid_head_features, feature_x, feature_y, patch_size=patch_size)

            valid_text_inputs = clip_text_inputs[cls_id_ids[i][0:num_obj[i]]]

            clip_text_features_.append(valid_text_inputs)
            head_feature_patch_.append(head_feature_patches)

        clip_text_features_ = torch.cat(clip_text_features_, dim=0)
        clip_text_features_ = self.clip_text_fc(clip_text_features_)

        head_feature_patch_ = torch.cat(head_feature_patch_, dim=0)
        head_feature_patch_ = self.feature_head(head_feature_patch_)

        clip_text_features_ = clip_text_features_ / clip_text_features_.norm(dim=1, keepdim=True)
        head_feature_patch_ = head_feature_patch_ / head_feature_patch_.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * head_feature_patch_ @ clip_text_features_.t()
        logits_per_text = logits_per_image.t()

        image_loss = contrastive_loss(logits_per_image)
        text_loss = contrastive_loss(logits_per_text)

        clip_loss = (image_loss + text_loss) / all_num_obj * 0.5

        return clip_loss

class Clip_loss(nn.Module):
    def __init__(self, device):
        super(Clip_loss, self).__init__()

        self.classes = ['car']
        # self.extra_classes = ['other']

        self.device = device
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        self.clip_text_fc = nn.Sequential(nn.Linear(512, 256, bias=False), nn.LayerNorm(256), nn.SiLU(True),
                                           nn.Linear(256, 256, bias=False), nn.LayerNorm(256), nn.SiLU(True),
                                           nn.Dropout(),
                                           nn.Linear(256, 256, bias=False)
                                           )


        self.clip_image_fc = nn.Sequential(nn.Linear(512, 256, bias=False), nn.LayerNorm(256), nn.SiLU(True),
                                           nn.Linear(256, 256, bias=False), nn.LayerNorm(256), nn.SiLU(True),
                                           nn.Dropout(),
                                           nn.Linear(256, 256, bias=False)
                                           )

        self.feature_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten())
            # # nn.Linear(256, 512, bias=False), nn.LayerNorm(512), nn.SiLU(True),
            # # nn.Linear(512, 512, bias=False), nn.LayerNorm(512), nn.SiLU(True),
            # # nn.Dropout(),
            # nn.Linear(512, 512, bias=False))

        self.clip_preprocess_mean = torch.tensor(self.preprocess.transforms[-1].mean).view(1, 3, 1, 1).to(self.device)
        self.clip_preprocess_std = torch.tensor(self.preprocess.transforms[-1].std).view(1, 3, 1, 1).to(self.device)

        self.text_inputs = torch.cat([clip.tokenize(f'a photo of the {c}') for c in self.classes]).to(self.device)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def normalize_batch_inputs(self, batch_inputs):
        batch_inputs = (batch_inputs - self.clip_preprocess_mean) / self.clip_preprocess_std.clamp_min(1e-8)
        return batch_inputs

    def forward(self, images, ids, num_obj, bbox_cxcy, head_features):
        batch_size = images.size(0)
        img_h, img_w = images.size(-2), images.size(-1)
        h, w = head_features.size(-2), head_features.size(-1)

        down_ratio = img_h / h
        patch_size = 8

        clip_text_features_ = []
        clip_local_image_feats_ = []
        head_feature_patch_ = []
        all_num_obj = 0.
        for i in range(batch_size):
            image_patches = []  # ????? batch ? image_patches
            head_feature_patches = []

            if int(num_obj[i]) == 0:
                print("\n")
                print("No objects in this batch, skipping clip loss.")
                continue

            all_num_obj += float(num_obj[i])

            image = images[i]
            head_feature = head_features[i]

            text_inputs = torch.cat([clip.tokenize(f'a photo of the {c}') for c in self.classes]).to(
                self.device).repeat(num_obj[i], 1)

            for j in range(num_obj[i]):

                feature_x = torch.clamp(bbox_cxcy[i][j][0] * w, 0, w - 1)
                feature_y = torch.clamp(bbox_cxcy[i][j][1] * h, 0, h - 1)

                image_x = torch.clamp(bbox_cxcy[i][j][0] * img_w, 0, img_w - 1)
                image_y = torch.clamp(bbox_cxcy[i][j][1] * img_h, 0, img_h - 1)

                image_patches.append(extract_patch(image, image_x, image_y, patch_size=patch_size * down_ratio))
                head_feature_patches.append(extract_patch(head_feature, feature_x, feature_y, patch_size=patch_size))

            image_patch = torch.stack(image_patches, dim=0)
            image_patch = F.interpolate(image_patch, size=(225, 225), mode='bicubic', align_corners=False)
            head_feature_patch = torch.stack(head_feature_patches, dim=0)

            image_patch = self.normalize_batch_inputs(image_patch)

            with torch.no_grad():
                clip_text_features = self.clip_model.encode_text(text_inputs).float()
                clip_local_image_feats = self.clip_model.encode_image(image_patch).float()



            clip_text_features_.append(clip_text_features)
            clip_local_image_feats_.append(clip_local_image_feats)
            head_feature_patch_.append(head_feature_patch)

        clip_text_features_ = torch.cat(clip_text_features_, dim=0)
        clip_local_image_feats_ = torch.cat(clip_local_image_feats_, dim=0)
        head_feature_patch_ = torch.cat(head_feature_patch_, dim=0)

        head_feature_patch_ = self.feature_head(head_feature_patch_)
        clip_local_image_feats_ = self.clip_image_fc(clip_local_image_feats_)
        clip_text_features_ = self.clip_text_fc(clip_text_features_)

        clip_text_features_ = clip_text_features_ / clip_text_features_.norm(dim=1, keepdim=True)

        clip_local_image_feats_ = clip_local_image_feats_ / clip_local_image_feats_.norm(dim=1, keepdim=True)
        head_feature_patch_ = head_feature_patch_ / head_feature_patch_.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * head_feature_patch_ @ clip_text_features_.t()
        logits_per_text = logits_per_image.t()

        image_loss = contrastive_loss(logits_per_image)
        text_loss = contrastive_loss(logits_per_text)

        clip_loss = (image_loss + text_loss) / all_num_obj * 0.5

        # img_loss = mse_loss(clip_local_image_feats_, head_feature_patch_)

        # all_loss = clip_loss * 0.5 + img_loss * 0.5

        return clip_loss



class HmLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 num_classes=5,
                 gamma=12,
                 mu=0.8,
                 alpha=4.0,
                 vis_grad=False):
        super().__init__()
        self.use_sigmoid = True
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.num_classes = num_classes
        self.group = True

        self.vis_grad = vis_grad
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha

        # initial variables
        self._pos_grad = None
        self._neg_grad = None
        self.pos_neg = None

        def _func(x, gamma, mu):
            return 1 / (1 + torch.exp(-gamma * (x - mu)))

        self.map_func = partial(_func, gamma=self.gamma, mu=self.mu)

    def forward(self,
                cls_score,
                label,
                size,
                location_class,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):

        self.batch, self.n_c, self.h, self.w = cls_score.size()
        self.gt_num = label.eq(1).float().sum()
        self.gt_classes = label
        self.pred_class_logits = cls_score

        target = label
        pos_w, neg_w = self.get_weight(cls_score)

        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()

        weight = (pos_w * target.permute(0, 2, 3, 1) + neg_w * (
                1 - target.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)

        cls_loss = F.binary_cross_entropy_with_logits(cls_score, target, reduction='none')
        if self.gt_num == 0:
            cls_loss = torch.sum(cls_loss * weight)
        else:
            cls_loss = torch.sum(cls_loss * weight) / self.gt_num

        self.collect_grad(cls_score.detach(), target.detach(), weight.detach())

        return self.loss_weight * cls_loss

    def get_channel_num(self, num_classes):
        num_channel = num_classes + 1
        return num_channel

    def collect_grad(self, cls_score, target, weight):
        prob = torch.sigmoid(cls_score)
        grad = target * (prob - 1) + (1 - target) * prob

        grad = torch.abs(grad)

        pos_grad = torch.sum(grad * target * weight, dim=(0, 2, 3))
        neg_grad = torch.sum(grad * (1 - target) * weight, dim=(0, 2, 3))

        # dist.all_reduce(pos_grad)
        # dist.all_reduce(neg_grad)

        self._pos_grad += pos_grad
        self._neg_grad += neg_grad
        self.pos_neg = self._pos_grad / (self._neg_grad + 1e-10)

    def get_weight(self, cls_score):
        # we do not have information about pos grad and neg grad at beginning
        if self._pos_grad is None:
            self._pos_grad = cls_score.new_zeros(self.num_classes)
            self._neg_grad = cls_score.new_zeros(self.num_classes)
            neg_w = cls_score.new_ones(self.n_c)
            pos_w = cls_score.new_ones(self.n_c)
        else:
            neg_w = self.map_func(self.pos_neg)
            pos_w = 1 + self.alpha * (1 - neg_w)

        return pos_w, neg_w

    def reset_grads(self):
        self._pos_grad = None
        self._neg_grad = None
        self.pos_neg = None

class GBF(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 num_classes=10,
                 gamma=12,
                 mu=0.8,
                 alpha=4.0,
                 vis_grad=False):
        super().__init__()
        self.use_sigmoid = True
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.num_classes = num_classes
        self.group = True


        self.vis_grad = vis_grad
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha

        # initial variables
        self._pos_grad = None
        self._neg_grad = None
        self.pos_neg = None

        def _func(x, gamma, mu):
            return 1 / (1 + torch.exp(-gamma * (x - mu)))
        self.map_func = partial(_func, gamma=self.gamma, mu=self.mu)


    def forward(self,
                cls_score,
                label,
                size,
                location_class,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        # self.n_i, self.n_c = cls_score.size()
        self.batch, self.n_c, self.h, self.w = cls_score.size()
        self.gt_num = label.eq(1).float().sum()
        self.gt_classes = label
        self.pred_class_logits = cls_score

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target

        # target = expand_label(cls_score, label)
        target = label
        pos_w, neg_w = self.get_weight(cls_score)

        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()
        # weight = (pos_w * (target*pos_inds).permute(0,2,3,1)  + neg_w * (1 - (target*neg_inds).permute(0,2,3,1))).permute(0,3,1,2)
        weight = (pos_w * target.permute(0, 2, 3, 1)  + neg_w * (
                    1 - target.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        size_weight = torch.ones(cls_score.size()).cuda()
        for i in range(200):
            x_y_cls = location_class[:,i,:].type(torch.long)
            w=torch.exp(-(size[:,i,0]*size[:,i,1]-5))+1
            size_weight[:,x_y_cls[:, 2], x_y_cls[:, 1], x_y_cls[:, 0]] = w

        # cls_loss = design_focal_loss(cls_score, target)
        cls_loss = F.binary_cross_entropy(cls_score, target,reduction='none')
        if self.gt_num == 0:
            cls_loss = torch.sum(cls_loss * weight *size_weight)
        else:
            cls_loss = torch.sum(cls_loss * weight*size_weight) / self.gt_num



        self.collect_grad(cls_score.detach(), target.detach(), weight.detach())

        return self.loss_weight * cls_loss

    def get_channel_num(self, num_classes):
        num_channel = num_classes + 1
        return num_channel


    def collect_grad(self, cls_score, target, weight):
        # prob = torch.sigmoid(cls_score)
        prob = cls_score
        # pos_inds = target.eq(1).float()
        # neg_inds = target.lt(1).float()
        grad = target * (prob - 1) + (1 - target) * prob
        # grad = target * (torch.pow(prob-1,3) - 2 * torch.pow(prob-1,2) * prob + prob * torch.log(prob))*pos_inds + \
        #        (1 - target) * (2*torch.pow(prob,2)*(prob-1) +torch.log(1-prob)+torch.pow(prob,3)) * neg_inds
        grad = torch.abs(grad)

        pos_grad = torch.sum(grad * target * weight, dim=(0,2,3))
        neg_grad = torch.sum(grad * (1 - target) * weight, dim=(0,2,3))


        dist.all_reduce(pos_grad)
        dist.all_reduce(neg_grad)

        self._pos_grad += pos_grad
        self._neg_grad += neg_grad
        self.pos_neg = self._pos_grad / (self._neg_grad + 1e-10)

    def get_weight(self, cls_score):
        # we do not have information about pos grad and neg grad at beginning
        if self._pos_grad is None:
            self._pos_grad = cls_score.new_zeros(self.num_classes)
            self._neg_grad = cls_score.new_zeros(self.num_classes)
            neg_w = cls_score.new_ones(self.n_c)
            pos_w = cls_score.new_ones(self.n_c)
        else:
            neg_w = self.map_func(self.pos_neg)
            pos_w = 1 + self.alpha * (1 - neg_w)

        return pos_w, neg_w

def ciou_loss(pred_boxes: Tensor, gt_boxes: Tensor, eps: float = 1e-7) -> Tensor:
    """
    Paired CIoU loss for N matched (pred, gt) box pairs.

    Args:
        pred_boxes: (N, 4) xyxy in feature-map pixel coords
        gt_boxes  : (N, 4) xyxy in feature-map pixel coords
    Returns:
        scalar mean CIoU loss over N pairs
    """
    px1, py1, px2, py2 = pred_boxes.unbind(-1)
    gx1, gy1, gx2, gy2 = gt_boxes.unbind(-1)

    ix1 = torch.max(px1, gx1);  iy1 = torch.max(py1, gy1)
    ix2 = torch.min(px2, gx2);  iy2 = torch.min(py2, gy2)
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)

    pw = (px2 - px1).clamp(0);  ph = (py2 - py1).clamp(0)
    gw = (gx2 - gx1).clamp(0);  gh = (gy2 - gy1).clamp(0)
    union = pw * ph + gw * gh - inter + eps
    iou   = inter / union

    ex1 = torch.min(px1, gx1);  ey1 = torch.min(py1, gy1)
    ex2 = torch.max(px2, gx2);  ey2 = torch.max(py2, gy2)
    c2  = (ex2 - ex1).pow(2) + (ey2 - ey1).pow(2) + eps

    pcx = (px1 + px2) * 0.5;  pcy = (py1 + py2) * 0.5
    gcx = (gx1 + gx2) * 0.5;  gcy = (gy1 + gy2) * 0.5
    d2  = (pcx - gcx).pow(2) + (pcy - gcy).pow(2)

    v = (4.0 / math.pi ** 2) * (
        torch.atan(gw / (gh + eps)) - torch.atan(pw / (ph + eps))
    ).pow(2)
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)

    return (1.0 - (iou - d2 / c2 - alpha * v)).mean()


class VarifocalLoss(nn.Module):
    """
    Varifocal Loss (VFL) for heatmap supervision.

    For positive pixels: target = IoU(pred_box, gt_box) * gt_hm_value
    For negative pixels: standard focal suppression α * pred^γ * BCE

    This aligns heatmap score with actual localization quality — a detection
    that predicts the right location but wrong size gets a lower target score
    than one that predicts both correctly.

    Reference: Zhang et al. "VarifocalNet" CVPR 2021.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: Tensor, gt_hm: Tensor,
                iou_map: Tensor | None = None) -> Tensor:
        """
        Args:
            pred    : (B, C, H, W) raw logits
            gt_hm   : (B, C, H, W) Gaussian-rendered heatmap targets in [0,1]
            iou_map : (B, C, H, W) IoU quality at positive locations (optional).
                      When provided, positive targets = iou_map * (gt_hm == 1).
                      When None, falls back to standard focal loss.
        """
        pred_sigmoid = pred.sigmoid()

        if iou_map is not None:
            # positive target = iou quality; negative target = 0
            pos_mask = (gt_hm == 1.0).float()
            target   = iou_map * pos_mask + gt_hm * (1.0 - pos_mask)
        else:
            target = gt_hm

        # VFL: positives weighted by |q - p|^γ, negatives weighted by α * p^γ
        pos_mask = (target > 0).float()
        weight = (
            pos_mask * (target - pred_sigmoid).abs().pow(self.gamma)
            + (1.0 - pos_mask) * self.alpha * pred_sigmoid.pow(self.gamma)
        )

        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        return (bce * weight).sum() / (pos_mask.sum().clamp(min=1.0))


class LogWHLoss(nn.Module):
    """
    Scale-invariant WH loss: supervise in log space.

    The WH head predicts log(w), log(h).  GT targets are log(gt_w), log(gt_h).
    This makes a 2× error on a 5px object identical in loss magnitude to a
    2× error on a 50px object — critical for UAV where objects span 10–200px.

    At inference, apply exp() to the head output before using box coordinates.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, output: Tensor, mask: Tensor,
                ind: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            output : (B, 2, H, W)     raw wh head output (predicts log-space)
            mask   : (B, max_obj)     valid-object binary mask
            ind    : (B, max_obj)     flat spatial peak indices
            target : (B, max_obj, 2)  GT wh in pixel coords (will be log-ified)
        """
        pred = _tranpose_and_gather_feat(output, ind)   # (B, max_obj, 2)
        mask = mask.unsqueeze(2).expand_as(pred).float()

        log_target = torch.log(target.clamp(min=1e-4))  # log(gt_wh)
        loss = F.l1_loss(pred * mask, log_target * mask, reduction='sum')
        return loss / (mask.sum() + 1e-4)


# ── Repulsion Loss ─────────────────────────────────────────────────────────────
# Wang et al., "Repulsion Loss: Repulsing Proposals in Object Detection",
# CVPR 2018.  Penalises predicted boxes that overlap the GT of a neighbouring
# object, preventing box drift in dense / closely-packed scenes.

def _pairwise_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """(n, m) IoU matrix between two sets of xyxy boxes."""
    a1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(0)
    a2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(0)
    ix1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    iy1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    ix2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    iy2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    return inter / (a1[:, None] + a2[None, :] - inter + 1e-7)


def repulsion_loss(
    pred_boxes: Tensor,
    gt_boxes:   Tensor,
    mask:       Tensor,
    sigma:      float = 0.0,
) -> Tensor:
    """
    Repulsion Loss — penalises each predicted box for overlapping the GT of
    any *neighbouring* object (i.e. every GT except its own matched one).

    Args:
        pred_boxes: (B, max_obj, 4)  xyxy, feature-map coordinates.
        gt_boxes:   (B, max_obj, 4)  xyxy, feature-map coordinates.
        mask:       (B, max_obj)     bool — True for real (non-padded) objects.
        sigma:      Smooth-ln threshold.  0.0 → plain −log(1 − IoU).

    Returns:
        Scalar averaged over images that have ≥ 2 valid objects.
    """
    total, n_img = pred_boxes.new_zeros(()), 0

    for b in range(pred_boxes.shape[0]):
        valid = mask[b]
        n = int(valid.sum())
        if n < 2:
            continue

        p = pred_boxes[b][valid]                         # (n, 4)
        g = gt_boxes[b][valid]                           # (n, 4)

        iou = _pairwise_iou(p, g).fill_diagonal_(0.0)   # zero out matched pairs
        repul_iou, _ = iou.max(dim=1)                   # nearest non-matched GT

        if sigma > 0.0:
            penalty = torch.where(
                repul_iou < sigma,
                -torch.log(1.0 - repul_iou + 1e-7),
                (repul_iou - sigma) / (1.0 - sigma + 1e-7)
                + math.log(1.0 / (1.0 - sigma + 1e-7)),
            )
        else:
            penalty = -torch.log(1.0 - repul_iou + 1e-7)

        total  = total + penalty.mean()
        n_img += 1

    return total / max(n_img, 1)
