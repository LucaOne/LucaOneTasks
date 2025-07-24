#!/usr/bin/env python
# encoding: utf-8
"""
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2025/7/18 16:36
@project: LucaOneTasks
@file: test
@desc: xxxx
"""
import sys
import torch
import logging
sys.path.append(".")
sys.path.append("..")
sys.path.append("../src")
try:
    from file_operator import *
except ImportError:
    from src.file_operator import *

filepath = "../../predicts/ATAC_GE_loc_pair_30knn_3x/prediction_results.csv"

total = 0
right = 0
metrics = {
    "fp": 0,
    "tp": 0,
    "fn": 0,
    "tn": 0
}
for row in csv_reader(filepath):
    seq_id_a,seq_id_b,matrix_filename_a,matrix_filename_b,prob,label,ground_truth = row
    label = int(float(prob)>=0.5)
    ground_truth = int(ground_truth)
    total += 1
    if label == ground_truth and label == 1:
        right += 1
        metrics["tp"] += 1
    elif label == ground_truth and label == 0:
        right += 1
        metrics["tn"] += 1
    elif label != ground_truth and label == 1:
        metrics["fp"] += 1
    else:
        metrics["fn"] += 1
print("total: %d, right: %d" % (total, right))
print("metrics:")
print(metrics)
