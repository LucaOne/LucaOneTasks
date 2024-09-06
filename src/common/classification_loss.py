#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/5/3 20:35
@project: LucaOneTasks
@file: loss.py
@desc: loss
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../src")
try:
    from masked_loss import _MaskedLoss
except ImportError:
    from src.common.masked_loss import _MaskedLoss


class MaskedFocalLossBinaryClass(_MaskedLoss):
    """Masked FocalLoss"""
    def __init__(self, alpha=1, gamma=2, normalization=True, reduction='mean', ignore_nans=True, ignore_value=-100):
        super().__init__(reduction=reduction, ignore_nans=ignore_nans, ignore_value=ignore_value)
        self.criterion = FocalLossBinaryClass(alpha=alpha, gamma=gamma, normalization=normalization, reduction='none', ignore_value=ignore_value)


class FocalLossBinaryClass(nn.Module):
    '''
    Focal loss
    '''
    def __init__(self, alpha=1, gamma=2, normalization=True, reduction="mean", ignore_value=-100):
        super(FocalLossBinaryClass, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.normalization = normalization
        self.reduction = reduction
        self.ignore_value = ignore_value

    def forward(self, inputs, targets):
        '''
        binary-classification:
            seq-level
            inputs: (N,), outputs: (N, )
            token-level
            inputs: (N * max_len,), outputs: (N * max_len, )
            or
            inputs: (N, max_len, 1), outputs: (N, max_len)
        multi-label-classification:
            seq-level
            inputs: (N, label_size), outputs: (N, label_size)
            token-level
            inputs: (N * max_len, label_size), outputs: (N * max_len, label_size)
            or
            inputs: (N, max_len, label_size), outputs: (N, max_len, label_size)
        :param inputs:
        :param targets:
        :return:
        '''
        if inputs.shape[-1] == 1:
            inputs = torch.squeeze(inputs, dim=-1)
        if targets.shape[-1] == 1:
            targets = torch.squeeze(targets, dim=-1)
        '''
        print("inputs:")
        print(inputs)
        print("targets:")
        print(targets)
        print(len(targets))
        unignored_mask = targets != -100
        targets = targets[unignored_mask]
        print(len(targets))
        if len(targets) == 0:
            return torch.tensor(0., device=inputs.device)
        inputs = inputs[unignored_mask]
        '''
        mask = targets != self.ignore_value
        targets[~mask] = 0.0
        inputs[~mask] = 0.0
        assert inputs.ndim == targets.ndim
        if self.normalization:
            '''
             reduction: the operation on the output loss, which can be set to 'none', 'mean', and 'sum'; 
            'none' will not perform any processing on the loss, 
            'mean' will calculate the mean of the loss, 
            'sum' will sum the loss, and the default is 'mean'
            '''
            bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            probs = torch.sigmoid(inputs)
        else:
            bce = F.binary_cross_entropy(inputs, targets, reduction='none')
            probs = inputs
        bce[~mask] = 0.0
        pt = targets * probs + (1 - targets) * (1 - probs)

        modulate = 1 if self.gamma is None else (1 - pt) ** self.gamma

        focal_loss = modulate * bce

        if self.alpha is not None:
            assert 0 <= self.alpha <= 1
            alpha_weights = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            focal_loss *= alpha_weights
        focal_loss[~mask] = 0.0

        if self.reduction == 'none':
            return focal_loss
        if self.reduction == 'sum':
            return focal_loss.sum()
        if self.reduction == 'mean':
            return focal_loss.sum() / (mask.to(focal_loss.dtype).sum() + 1e-12)
        if self.reduction == 'meanmean':
            if mask.ndim == 3:
                mask_sum = mask.to(focal_loss.dtype).sum(dim=-1)
                '''
                print("mask:")
                print(mask_sum)
                '''
                full_loss = focal_loss.sum(dim=-1) / (mask_sum + 1e-12)
                mask_sum = mask_sum.to(torch.bool).sum(dim=-1)
                # print(mask_sum)
                full_loss = full_loss.sum(dim=-1) / (mask_sum + 1e-12)
                mask_sum = mask_sum.to(torch.bool).sum()
                # print(mask_sum)
                loss = full_loss.sum() / (mask_sum + 1e-12)
            else:
                mask_sum = mask.to(focal_loss.dtype).sum(dim=-1)
                '''
                print("mask:")
                print(mask_sum)
                print(mask_sum.to(torch.bool).sum())
                '''
                loss = torch.sum(focal_loss.sum(dim=-1) / (mask_sum + 1e-12)) / (mask_sum.to(torch.bool).sum() + 1e-12)
            # print(full_loss.sum() / (mask.to(full_loss.dtype).sum() + 1e-12), loss)
            return loss
        if self.reduction in ['summean', 'meansum']:
            if mask.ndim == 3:
                mask_sum = mask.to(focal_loss.dtype).sum(dim=-1)
                '''
                print("mask:")
                print(mask_sum)
                '''
                full_loss = focal_loss.sum(dim=-1)
                mask_sum = mask_sum.to(torch.bool).sum(dim=-1)
                # print(mask_sum)
                full_loss = full_loss.sum(dim=-1) / (mask_sum + 1e-12)
                mask_sum = mask_sum.to(torch.bool).sum()
                # print(mask_sum)
                loss = full_loss.sum() / (mask_sum + 1e-12)
            else:
                mask_sum = mask.to(focal_loss.dtype).sum(dim=-1)
                '''
                print("mask:")
                print(mask_sum)
                print(mask_sum.to(torch.bool).sum())
                '''
                loss = focal_loss.sum() / (mask_sum.to(torch.bool).sum() + 1e-12)
            return loss
        return focal_loss


class MaskedFocalLossMultiClass(_MaskedLoss):
    """Masked FocalLoss"""
    def __init__(self, alpha=None, gamma=2, normalization=True, reduction='mean', ignore_nans=True, ignore_value=-100):
        super().__init__(reduction=reduction, ignore_nans=ignore_nans, ignore_value=ignore_value)
        self.criterion = FocalLossBinaryClass(alpha=alpha, gamma=gamma, normalization=normalization, reduction='none')


class FocalLossMultiClass(nn.Module):
    '''Multi-class Focal loss implementation'''
    def __init__(self, alpha=None, gamma=2, normalization=True, reduction='mean'):
        super(FocalLossMultiClass, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.normalization = normalization
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        multi-classification:
            seq-level
            inputs: (N, label_size), outputs: (N, 1)
            token-level
            inputs: (N * max_len, label_size), outputs: (N * max_len, 1)
            or
            inputs: (N, max_len, label_size), outputs: (N, max_len)
        """
        '''
        if self.normalization:
            logpt = F.log_softmax(inputs, dim=-1)
        else:
            logpt = inputs
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, targets, self.weight, ignore_index=self.ignore_index)
        '''
        if targets.shape[-1] == 1:
            targets = torch.squeeze(targets, dim=-1)
        assert inputs.ndim == targets.ndim + 1
        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(inputs, dim=-1)
        ce = self.nll_loss(log_p, targets)
        # get true class column from each row
        all_rows = torch.arange(len(inputs))
        log_pt = log_p[all_rows, targets]
        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma
        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == "meanmean":
            # mean of all samples and calc the mean value
            return torch.mean(torch.mean(loss, dim=-1))
        elif self.reduction in ['summean', 'meansum']:
            # sum of all samples and calc the mean value
            return torch.mean(torch.sum(loss, dim=-1))
        return loss


class MaskedMultiLabelCCE(_MaskedLoss):
    """Masked MultiLabel CCE"""
    def __init__(self, normalization=False, reduction='mean', ignore_nans=True, ignore_value=-100):
        super().__init__(reduction=reduction, ignore_nans=ignore_nans, ignore_value=ignore_value)
        self.criterion = MultiLabelCCE(normalization=normalization, reduction='none')


class MultiLabelCCE(nn.Module):
    '''
    Multi Label CCE
    '''
    def __init__(self, normalization=True, reduction='mean'):
        super(MultiLabelCCE, self).__init__()
        self.normalization = normalization
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Cross entropy of multi-label classification
        Note：The shapes of y_true and y_pred are consistent, and the elements of y_true are either 0 or 1. 1 indicates
        that the corresponding class is a target class, and 0 indicates that the corresponding class is a non-target class.
        """
        if self.normalization:
            y_pred = torch.sigmoid(inputs)
        else:
            y_pred = inputs

        y_true = targets
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat((y_pred_neg, zeros), axis=-1)
        y_pred_pos = torch.cat((y_pred_pos, zeros), axis=-1)
        neg_loss = torch.logsumexp(y_pred_neg,  axis=-1)
        pos_loss = torch.logsumexp(y_pred_pos,  axis=-1)
        if self.reduction == 'mean':
            return torch.mean(neg_loss + pos_loss)
        elif self.reduction == 'sum':
            return torch.sum(neg_loss + pos_loss)
        else:
            return neg_loss + pos_loss


class MaskedAsymmetricLoss(_MaskedLoss):
    """Masked AsymmetricLoss"""
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False, reduction='mean', ignore_nans=True, ignore_value=-100):
        super().__init__(reduction=reduction, ignore_nans=ignore_nans, ignore_value=ignore_value)
        self.criterion = AsymmetricLoss(gamma_neg, gamma_pos, clip, eps, disable_torch_grad_focal_loss)


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


class MaskedAsymmetricLossOptimized(_MaskedLoss):
    """Masked ASLSingleLabel loss"""
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False, reduction='mean', ignore_nans=True, ignore_value=-100):
        super().__init__(reduction=reduction, ignore_nans=ignore_nans, ignore_value=ignore_value)
        self.criterion = AsymmetricLossOptimized(gamma_neg, gamma_pos, clip, eps, disable_torch_grad_focal_loss)


class AsymmetricLossOptimized(nn.Module):
    '''
    Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations
    '''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()


class MaskedASLSingleLabel(_MaskedLoss):
    """Masked ASLSingleLabel loss"""
    def __init__(self, gamma_pos=0, gamma_neg=4, eps: float = 0.1, reduction='mean', ignore_nans=True, ignore_value=-100):
        super().__init__(reduction=reduction, ignore_nans=ignore_nans, ignore_value=ignore_value)
        self.criterion = ASLSingleLabel(gamma_pos, gamma_neg, eps, reduction='none')


class ASLSingleLabel(nn.Module):
    '''
    This loss is intended for single-label classification problems（multi-class）
    '''
    def __init__(self, gamma_pos=0, gamma_neg=4, eps: float = 0.1, reduction='mean'):
        super(ASLSingleLabel, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target):
        '''
        "input" dimensions: - (batch_size, number_classes)
        "target" dimensions: - (batch_size)
        '''
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg, self.gamma_pos * targets + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:
            # label smoothing
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(self.eps / num_classes)

        # loss calculation
        loss = - self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss


'''
class MaskedBCEWithLogitsLoss(BCEWithLogitsLoss):
    __name__ = "MaskedBCEWithLogitsLoss"

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction2="none", ignore_index=-100,
                 pos_weight: Optional[Tensor] = None) -> None:
        super(MaskedBCEWithLogitsLoss, self).__init__(weight=weight, size_average=size_average, reduce=reduce, reduction="none", pos_weight=pos_weight)
        self.reduction2 = reduction2
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        loss_val = super().forward(input, target)
        if mask is None and self.ignore_index is None:
            if self.reduction2 == "mean":
                return torch.mean(loss_val)
            elif self.reduction2 == "sum":
                return torch.sum(loss_val)
            else:
                return loss_val
        elif mask is None:
            mask = (target != self.ignore_index)
        if self.reduction2 == "mean":
            return torch.sum(loss_val * mask) / torch.sum(mask)
        elif self.reduction2 == "sum":
            return torch.sum(loss_val * mask)
        else:
            return loss_val * mask
'''


class MaskedBCEWithLogitsLoss(_MaskedLoss):
    """Masked MSE loss"""
    def __init__(self, pos_weight=None, weight=None, reduction='mean', ignore_nans=True, ignore_value=-100):
        super().__init__(reduction=reduction, ignore_nans=ignore_nans, ignore_value=ignore_value)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, weight=weight, reduction='none')


class MaskedCrossEntropyLoss(_MaskedLoss):
    """Masked MSE loss"""
    def __init__(self, weight=None, reduction='mean', ignore_nans=True, ignore_value=-100):
        super().__init__(reduction=reduction, ignore_nans=ignore_nans, ignore_value=ignore_value)
        self.criterion = nn.CrossEntropyLoss(weight=weight, reduction='none', ignore_index=ignore_value)


if __name__ == "__main__":
    '''
    loss_fct = nn.BCEWithLogitsLoss(reduction="none")
    batch_size = prediction_scores.size()[0]
    seq_len = prediction_scores.size()[1]
    tmp_loss = loss_fct(prediction_scores.view(batch_size * seq_len, 1),
                        labels.view(batch_size * seq_len, 1).float())
    tmp_mask = (labels != -100)
    # 忽视类别标签为-100
    real_loss = tmp_loss.view(batch_size, seq_len) * tmp_mask
    masked_lm_loss = torch.sum(real_loss) / torch.sum(tmp_mask)
    '''
    import torch
    label = torch.Tensor([[[1], [1], [-100]], [[1], [-100], [0]]])
    pred = torch.Tensor([[[2], [1], [3]], [[2], [1], [3]]])
    loss = MaskedBCEWithLogitsLoss(reduction="mean")
    print(loss(pred, label))


