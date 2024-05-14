#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/8/14 22:45
@project: LucaOne
@file: pairwise_metrics.py
@desc: pairwise metrics
'''
import torch


def compute_precision_at_l(targets, probs, seq_lens, ignore_index):
    with torch.no_grad():
        valid_mask = targets != ignore_index
        seq_pos = torch.arange(valid_mask.size(1), device=seq_lens.device)
        x_ind, y_ind = torch.meshgrid(seq_pos, seq_pos)
        valid_mask &= ((y_ind - x_ind) >= 24).unsqueeze(0)
        valid_mask = valid_mask.type_as(probs)
        correct = 0
        total = 0
        for length, prob, target, mask in zip(seq_lens, probs, targets, valid_mask):
            masked_prob = (prob * mask).view(-1)
            most_likely = masked_prob.topk(length, sorted=False)
            selected = target.view(-1).gather(0, most_likely.indices)
            correct += selected.sum().float()
            total += selected.numel()
        return correct / total


def compute_precision_at_l2(targets, probs, seq_lens, ignore_index):
    with torch.no_grad():
        valid_mask = targets != ignore_index
        seq_pos = torch.arange(valid_mask.size(1), device=seq_lens.device)
        x_ind, y_ind = torch.meshgrid(seq_pos, seq_pos)
        valid_mask &= ((y_ind - x_ind) >= 24).unsqueeze(0)
        valid_mask = valid_mask.type_as(probs)
        correct = 0
        total = 0
        for length, prob, target, mask in zip(seq_lens, probs, targets, valid_mask):
            masked_prob = (prob * mask).view(-1)
            most_likely = masked_prob.topk(length // 2, sorted=False)
            selected = target.view(-1).gather(0, most_likely.indices)
            correct += selected.sum().float()
            total += selected.numel()
        return correct / total


def compute_precision_at_l5(targets, probs, seq_lens, ignore_index):
    with torch.no_grad():
        valid_mask = targets != ignore_index
        seq_pos = torch.arange(valid_mask.size(1), device=seq_lens.device)
        x_ind, y_ind = torch.meshgrid(seq_pos, seq_pos)
        valid_mask &= ((y_ind - x_ind) >= 24).unsqueeze(0)
        valid_mask = valid_mask.type_as(probs)
        correct = 0
        total = 0
        for length, prob, target, mask in zip(seq_lens, probs, targets, valid_mask):
            masked_prob = (prob * mask).view(-1)
            most_likely = masked_prob.topk(length // 5, sorted=False)
            selected = target.view(-1).gather(0, most_likely.indices)
            correct += selected.sum().float()
            total += selected.numel()
        return correct / total


def metrics_pairwise(targets, probs, seq_lens, ignore_index):
    if seq_lens is None:
        mask = ~torch.all(targets == ignore_index, dim=-1)
        seq_lens = mask.long().sum(dim=-1)
    metrics = {
        'precision_at_l': compute_precision_at_l(targets, probs, seq_lens, ignore_index=ignore_index),
        'precision_at_l2': compute_precision_at_l2(targets, probs, seq_lens, ignore_index=ignore_index),
        'precision_at_l5': compute_precision_at_l5(targets, probs, seq_lens, ignore_index=ignore_index)
    }
    print("PairwiseLoss metrics:")
    print(metrics)
    return metrics


if __name__ == "__main__":
    a = torch.tensor([[[0, 1, 3], [1, 2, -100], [-100, -100, -100]],[[0, 1, 3], [-100, -100, -100], [-100, -100, -100]]], dtype=torch.int32)
    print(a)
    mask = a == -100
    print(mask)
    mask = ~torch.all(a == -100, dim=-1)
    seq_lens = mask.long().sum(dim=-1)
    print(seq_lens)