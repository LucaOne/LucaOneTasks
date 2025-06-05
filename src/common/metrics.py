#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.**@**.com
@tel: 137****6540
@datetime: 2022/11/26 21:05
@project: LucaOneTasks
@file: metrics
@desc: metrics for binary classification or multi-class classification
'''
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams.update({'font.size': 18})
plt.rcParams['axes.unicode_minus'] = False
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,  \
    average_precision_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score, matthews_corrcoef


def topk_accuracy_score(targets, probs, k=3):
    '''
    topk accuracy
    :param targets:
    :param probs:
    :param k:
    :return:
    '''
    # obtain top-k label
    max_k_preds = probs.argsort(axis=1)[:, -k:][:, ::-1]
    a_real = np.resize(targets, (targets.shape[0], 1))
    # obtain the match result
    match_array = np.logical_or.reduce(max_k_preds == a_real, axis=1)
    topk_acc_score = match_array.sum() / match_array.shape[0]
    return topk_acc_score


def multi_class_acc(targets, probs, threshold=0.5):
    if targets.ndim == 2:
        targets = np.argmax(targets, axis=1)
    preds = np.argmax(probs, axis=1)
    return accuracy_score(targets, preds)


def multi_class_precision(targets, probs, average='macro'):
    if targets.ndim == 2:
        targets = np.argmax(targets, axis=1)
    preds = np.argmax(probs, axis=1)
    return precision_score(targets, preds, average=average)


def multi_class_recall(targets, probs, average='macro'):
    if targets.ndim == 2:
        targets = np.argmax(targets, axis=1)
    preds = np.argmax(probs, axis=1)
    return recall_score(targets, preds, average=average)


def multi_class_f1(targets, probs, average='macro'):
    if targets.ndim == 2:
        targets = np.argmax(targets, axis=1)
    preds = np.argmax(probs, axis=1)
    return f1_score(targets, preds, average=average)


def multi_class_roc_auc(targets, probs, average='macro'):
    if targets.ndim == 2:
        targets = np.argmax(targets, axis=1)
    return roc_auc_score(targets, probs, average=average, multi_class='ovr')


def multi_class_pr_auc(targets, probs, average='macro'):
    if targets.ndim == 2:
        targets = np.argmax(targets, axis=1)
    z = probs.shape[1]
    new_targets = np.eye(z)[targets]
    pr_auc = average_precision_score(new_targets, probs, average=average)
    return pr_auc


def metrics_multi_class(targets, probs, average="macro"):
    '''
    metrics of multi-class classification
    :param targets: 1d-array class index (n_samples, )
    :param probs:  2d-array probability (n_samples, m_classes)
    :return:
    '''
    if targets.ndim == 2 and targets.shape[1] > 1:
        targets = np.argmax(targets, axis=1)
    elif targets.ndim == 2 and targets.shape[1] == 1:
        targets = np.squeeze(targets, axis=1)

    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds, average=average)
    recall = recall_score(targets, preds, average=average)
    f1 = f1_score(targets, preds, average=average)
    result = {
        "acc": round(float(acc), 6),
        "prec": round(float(prec), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6)
    }
    result.update({
        "top2_acc": round(float(topk_accuracy_score(targets, probs, k=2)), 6),
        "top3_acc": round(float(topk_accuracy_score(targets, probs, k=3)), 6),
        "top4_acc": round(float(topk_accuracy_score(targets, probs, k=4)), 6),
        "top5_acc": round(float(topk_accuracy_score(targets, probs, k=5)), 6),
        "top10_acc": round(float(topk_accuracy_score(targets, probs, k=10)), 6)
    })
    try:
        roc_auc = roc_auc_score(targets, probs, average=average, multi_class='ovr')
        result.update({
            "roc_auc": round(float(roc_auc), 6)
        })
    except Exception as e:
        pass
    try:
        z = probs.shape[1]
        new_targets = np.eye(z)[targets]
        pr_auc = average_precision_score(new_targets, probs, average=average)
        result.update({
            "pr_auc": round(float(pr_auc), 6),
        })
    except Exception as e:
        pass
    try:
        mcc = matthews_corrcoef(targets, preds)
        result.update({
            "mcc": round(mcc, 6)
        })
        sn_avg = metrics_multi_class_sn(targets, preds)
        result.update({
            "sn": sn_avg
        })
        sp_avg = metrics_multi_class_sp(targets, preds)
        result.update({
            "sp": sp_avg
        })
    except Exception as e:
        pass
    try:
        cm = confusion_matrix(targets, preds)
        cm = cm.tolist()
        result.update({
            "confusion_matrix": cm
        })
    except Exception as e:
        pass
    return result


def metrics_multi_class_sn(targets, preds):
    n_classes = np.unique(targets).size
    con_mat = confusion_matrix(targets, preds)
    sn_list = []
    for class_idx in range(n_classes):
        tp = con_mat[class_idx][class_idx]
        fn = np.sum(con_mat[class_idx, :]) - tp
        sn = tp / (tp + fn)
        sn_list.append(sn)
    sn_avg = round(sum(sn_list)/len(sn_list), 6)
    return sn_avg


def metrics_multi_class_sp(targets, preds):
    n_classes = np.unique(targets).size
    con_mat = confusion_matrix(targets, preds)
    sp_list = []
    n = targets.size
    for class_idx in range(n_classes):
        tp = con_mat[class_idx][class_idx]
        fn = np.sum(con_mat[class_idx, :]) - tp
        fp = np.sum(con_mat[:, class_idx]) - tp
        tn = n - tp - fn - fp
        sp = tn / (tn + fp)
        sp_list.append(sp)
    sp_avg = round(sum(sp_list)/len(sp_list), 6)
    return sp_avg


def metrics_multi_class_for_pred(targets, preds, probs=None, average="macro", savepath=None):
    '''
    metrcis for multi-class classification
    :param targets: 1d-array class index (n_samples, )
    :param preds:  1d-array class index (n_samples, )
    :return:
    '''
    if targets.ndim == 2 and targets.shape[1] > 1:
        targets = np.argmax(targets, axis=1)
    elif targets.ndim == 2 and targets.shape[1] == 1:
        targets = np.squeeze(targets, axis=1)

    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds, average=average)
    recall = recall_score(targets, preds, average=average)
    f1 = f1_score(y_true=targets, y_pred=preds, average=average)
    result = {
        "acc": round(float(acc), 6),
        "prec": round(float(prec), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6)
    }
    try:
        roc_auc = roc_auc_score(targets, probs, average=average, multi_class='ovr')
        result.update({
            "roc_auc": round(float(roc_auc), 6)
        })
    except Exception as e:
        pass
    try:
        z = probs.shape[1]
        new_targets = np.eye(z)[targets]
        pr_auc = average_precision_score(new_targets, probs, average=average)
        result.update({
            "pr_auc": round(float(pr_auc), 6),
        })
    except Exception as e:
        pass
    try:
        mcc = matthews_corrcoef(targets, preds)
        result.update({
            "mcc": round(mcc, 6)
        })
        sn_avg = metrics_multi_class_sn(targets, preds)
        result.update({
            "sn": sn_avg
        })
        sp_avg = metrics_multi_class_sp(targets, preds)
        result.update({
            "sp": sp_avg
        })
    except Exception as e:
        pass
    try:
        cm = confusion_matrix(targets, preds)
        cm = cm.tolist()
        result.update({
            "confusion_matrix": cm
        })
    except Exception as e:
        pass
    return result


def metrics_regression(targets, preds):
    '''
    metrcis for regression
    :param targets: 1d-array class index (n_samples, )
    :param preds:  1d-array class index (n_samples, )
    :return:
    '''
    mae = mean_absolute_error(targets, preds)
    mse = mean_squared_error(targets, preds)
    r2 = r2_score(targets, preds)
    spearmanr = stats.spearmanr(targets, preds)
    pearsonr = stats.pearsonr(targets, preds)
    return {
        "mae": round(float(mae), 6),
        "mse": round(float(mse), 6),
        "r2": round(float(r2), 6),
        "sp_statistic": round(float(spearmanr.correlation), 6),
        "sp_pvalue": round(float(spearmanr.pvalue), 6),
        "ps_statistic": round(float(pearsonr.statistic), 6),
        "ps_pvalue": round(float(pearsonr.pvalue), 6)
    }


def metrics_contactmap(targets, probs, threshold=0.5, ranges=[1, 5]):
    '''
    metrcis for regression
    :param targets: 3d-array class index (n_samples, seq_len, seq_len)
    :param probs:  3d-array class index (n_samples, seq_len, seq_len)
    :return:
    '''
    max_seq_len = targets.shape[1]

    seqpos = np.arange(max_seq_len)
    y_ind, x_ind = np.meshgrid(seqpos, seqpos)
    print("y_ind:")
    print(y_ind)
    print("x_ind:")
    print(x_ind)
    if not isinstance(ranges, list):
        ranges = [ranges]
    res = {}
    for v in ranges:
        valid_mask = targets != -100
        print("valid_mask1:")
        print(valid_mask)
        valid_mask &= np.squeeze(((y_ind - x_ind) >= v + 1))
        print("valid_mask2:")
        print(valid_mask)
        valid_mask = valid_mask.astype(int)
        print("valid_mask3:")
        print(valid_mask)
        preds = (probs >= threshold).astype(int).reshape(-1)
        masked_targets = targets * valid_mask
        print("masked_targets:")
        print(masked_targets)
        masked_targets = masked_targets.astype(int).reshape(-1)
        correct = np.sum((preds == masked_targets).astype(int) * valid_mask.reshape(-1))
        print("correct:", correct)
        total = np.sum(valid_mask)
        print("total:", total)
        res["P_%d" % v] = round(correct / total, 6)
    return res


def transform(targets, probs, threshold):
    '''
    metrics of binary classification
    :param targets: 1d-array class index (n_samples, )
    :param probs: 1d-array larger class probability (n_samples, )
    :param threshold: 0-1 prob threshokd
    :return:
    '''
    if targets.ndim == 2:
        if targets.shape[1] == 2: # [[0, 1], [1, 0]]
            targets = np.argmax(targets, axis=1)
        else: # [[1], [0]]
            targets = targets.flatten()
    if probs.ndim == 2:
        if probs.shape[1] == 2: # [[0.1, 0.9], [0.9, 0.1]]
            preds = np.argmax(probs, axis=1)
            probs = probs[:, 1].flatten()
        else: # [[0.9], [0.1]]
            preds = (probs >= threshold).astype(int).flatten()
            probs = probs.flatten()
    else:
        preds = (probs >= threshold).astype(int)
    return targets, probs, preds


def binary_acc(targets, probs, threshold=0.5):
    targets, probs, preds = transform(targets, probs, threshold)
    return accuracy_score(targets, preds)


def binary_precision(targets, probs, threshold=0.5, average='binary'):
    targets, probs, preds = transform(targets, probs, threshold)
    return precision_score(targets, preds, average=average)


def binary_recall(targets, probs, threshold=0.5, average='binary'):
    targets, probs, preds = transform(targets, probs, threshold)
    return recall_score(targets, preds, average=average)


def binary_f1(targets, probs, threshold=0.5, average='binary'):
    targets, probs, preds = transform(targets, probs, threshold)
    return f1_score(targets, preds, average=average)


def binary_roc_auc(targets, probs, threshold=0.5, average='macro'):
    targets, probs, preds = transform(targets, probs, threshold)
    return roc_auc_score(targets, probs, average=average)


def binary_pr_auc(targets, probs, threshold=0.5, average='macro'):
    targets, probs, preds = transform(targets, probs, threshold)
    return average_precision_score(targets, probs, average=average)


def binary_confusion_matrix(targets, probs, threshold=0.5, savepath=None):
    targets, probs, preds = transform(targets, probs, threshold)
    cm_obj = confusion_matrix(targets, preds, labels=[0, 1])
    plot_confusion_matrix_for_binary_class(targets, preds, cm=cm_obj, savepath=savepath)
    tn, fp, fn, tp = cm_obj.ravel()
    cm = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return cm


def metrics_binary(targets, probs, threshold=0.5, average="binary", savepath=None):
    '''
    metrics for binary classification
    :param targets: 1d-array class index (n_samples, )
    :param probs: 1d-array larger class probability (n_samples, )
    :param threshold: 0-1 prob threshold
    :return:
    '''
    if targets.ndim == 2:
        if targets.shape[1] == 2: # [[0, 1], [1, 0]]
            targets = np.argmax(targets, axis=1)
        else: # [[1], [0]]
            targets = targets.flatten()
    if probs.ndim == 2:
        if probs.shape[1] == 2: # [[0.1, 0.9], [0.9, 0.1]]
            preds = np.argmax(probs, axis=1)
            probs = probs[:, 1].flatten()
        else: # [[0.9], [0.1]]
            preds = (probs >= threshold).astype(int).flatten()
            probs = probs.flatten()
    else:
        preds = (probs >= threshold).astype(int)
    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds, average=average)
    recall = recall_score(targets, preds, average=average)
    f1 = f1_score(targets, preds, average=average)
    result = {
        "acc": round(float(acc), 6),
        "prec": round(float(prec), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6)
    }
    try:
        roc_auc = roc_auc_score(targets, probs, average="macro")
        result.update({
            "roc_auc": round(float(roc_auc), 6)
        })
    except Exception as e:
        pass
    try:
        pr_auc = average_precision_score(targets, probs, average="macro")
        result.update({
            "pr_auc": round(float(pr_auc), 6)
        })
    except Exception as e:
        pass
    try:
        cm_obj = confusion_matrix(targets, preds, labels=[0, 1])
        plot_confusion_matrix_for_binary_class(targets, preds, cm=cm_obj, savepath=savepath)
        tn, fp, fn, tp = cm_obj.ravel()
        cm = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
        result.update({
            "confusion_matrix": cm
        })
    except Exception as e:
        pass
    # add mcc
    try:
        mcc2 = matthews_corrcoef(targets, preds)
        result.update({
            "mcc2": round(mcc2, 6)
        })
    except Exception as e:
        pass
    try:
        tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
        mcc = (tn*tp - fp*fn) / (((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ** 0.5)
        result.update({
            "mcc": round(mcc, 6)
        })
        result.update({
            "sn": round(tp/(tp + fn), 6)
        })
        result.update({
            "sp": round(tn/(tn + fp), 6)
        })
    except Exception as e:
        pass
    return result


def metrics_binary_for_pred(targets, preds, probs=None, average="binary", savepath=None):
    '''
    metrics for binary classification
    :param targets: 1d-array class index (n_samples, )
    :param preds: 1d-array larger class index (n_samples, )
    :return:
    '''
    if targets.ndim == 2:
        if targets.shape[1] == 2: # [[1, 0], [0, 1]
            targets = np.argmax(targets, axis=1)
        else: # [[1], [0]]
            targets = targets.flatten()
    if preds.ndim == 2:
        if preds.shape[1] == 2: # [[0.9, 0.1], [0.1, 0.9]]
            preds = np.argmax(preds, axis=1)
        else: # [[0], [1]]
            preds = preds.flatten()
    cm_obj = confusion_matrix(targets, preds, labels=[0, 1])
    plot_confusion_matrix_for_binary_class(targets, preds, cm=cm_obj, savepath=savepath)
    tn, fp, fn, tp = cm_obj.ravel()
    cm = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    if len(np.unique(targets)) > 1:
        acc = accuracy_score(targets, preds)
        prec = precision_score(targets, preds, average=average)
        recall = recall_score(targets, preds, average=average)
        f1 = f1_score(y_true=targets, y_pred=preds, average=average)
        result = {
            "acc": round(float(acc), 6),
            "prec": round(float(prec), 6),
            "recall": round(float(recall), 6),
            "f1": round(float(f1), 6)
        }
    else:
        result = {
            "acc": round(float((cm["tp"] + cm["tn"]) / (cm["tp"] + cm["tn"] + cm["fp"] + cm["fn"])), 6),
            "prec": round(float(cm["tp"]/(cm["tp"] + cm["fp"]) if cm["tp"] + cm["fp"] > 0 else 1.0), 6),
            "recall": round(float(cm["tp"]/(cm["tp"] + cm["fn"]) if cm["tp"] + cm["fn"] > 0 else 1.0), 6),
        }
        result["f1"] = 2 * result["prec"] * result["recall"] / (result["prec"] + result["recall"])
    try:
        pr_auc = average_precision_score(targets, probs, average="macro")
        result.update({
            "pr_auc": round(float(pr_auc), 6)
        })
    except Exception as e:
        pass
    try:
        roc_auc = roc_auc_score(targets, probs, average="macro")
        result.update({
            "roc_auc": round(float(roc_auc), 6)
        })
    except Exception as e:
        pass
    try:
        tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
        mcc = (tn*tp - fp*fn) / (((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ** 0.5)
        result.update({
            "mcc": round(mcc, 6)
        })
        result.update({
            "sn": round(tp/(tp + fn), 6)
        })
        result.update({
            "sp": round(tn/(tn + fp), 6)
        })
    except Exception as e:
        pass
    # add mcc
    try:
        mcc2 = matthews_corrcoef(targets, preds)
        result.update({
            "mcc2": round(mcc2, 6)
        })
    except Exception as e:
        pass
    result.update({
        "confusion_matrix": cm
    })
    return result


def write_error_samples_multi_class(filepath, samples, input_indexs, input_id_2_names, output_id_2_name, targets, probs,
                                    use_other_diags=False, use_other_operas=False, use_checkin_department=False):
    '''
    write the bad cases of multi-class classification
    :param filepath:
    :param samples:
    :param input_indexs:
    :param input_id_2_names:
    :param output_id_2_name:
    :param targets:
    :param probs:
    :param use_other_diags:
    :param use_other_operas:
    :param use_checkin_department:
    :return:
    '''
    targets = np.argmax(targets, axis=1)
    preds = np.argmax(probs, axis=1)
    with open(filepath, "w") as fp:
        writer = csv.writer(fp)
        writer.writerow(["score", "y_true", "y_pred", "inputs"])
        for i in range(len(targets)):
            target = targets[i]
            pred = preds[i]
            score = 1
            if target != pred:
                score = 0
            if output_id_2_name:
                target_label = output_id_2_name[target]
                pred_label = output_id_2_name[pred]
            else:
                target_label = target
                pred_label = pred
            sample = samples[i]
            if input_id_2_names:
                new_sample = []
                for idx, input_index in enumerate(input_indexs):
                    if input_index == 3 and not use_checkin_department:
                        input_index = 12
                    new_sample.append([input_id_2_names[idx][v] for v in sample[input_index]])
                    if (input_index == 6 and use_other_diags) or (input_index == 8 and use_other_operas) or (input_index == 10 and use_other_diags):
                        new_sample.append([input_id_2_names[idx][v] for v in sample[input_index + 1]])
            else:
                new_sample = sample
            row = [score, target_label, pred_label, new_sample]
            writer.writerow(row)


def write_error_samples_binary(filepath, samples, input_indexs, input_id_2_names, targets, probs, threshold=0.5,
                               use_other_diags=False, use_other_operas=False, use_checkin_department=False):
    '''
    write bad cases of binary classification
    :param filepath:
    :param samples:
    :param input_indexs:
    :param input_id_2_names:
    :param targets:
    :param probs:
    :param threshold:
    :param use_other_diags:
    :param use_other_operas:
    :param use_checkin_department:
    :return:
    '''
    with open(filepath, "w") as fp:
        writer = csv.writer(fp)
        writer.writerow(["score", "y_true", "y_pred", "inputs"])
        for i in range(len(targets)):
            target = targets[i][0]
            if target != 1:
                target = 1
            prob = probs[i][0]
            if prob >= threshold:
                pred = 1
            else:
                pred = 0
            score = 1
            if target != pred:
                score = 0
            target_label = "True" if target == 1 else "False"
            pred_label = "True" if target == 1 else "False"
            sample = samples[i]
            if input_id_2_names:
                new_sample = []
                for idx, input_index in enumerate(input_indexs):
                    if input_index == 3 and not use_checkin_department:
                        input_index = 12
                    new_sample.append([input_id_2_names[idx][v] for v in sample[input_index]])
                    if (input_index == 6 and use_other_diags) or (input_index == 8 and use_other_operas) or (input_index == 10 and use_other_diags):
                        new_sample.append([input_id_2_names[idx][v] for v in sample[input_index + 1]])
            else:
                new_sample = sample
            row = [score, target_label, pred_label, new_sample]
            writer.writerow(row)


def plot_confusion_matrix_for_binary_class(targets, preds, cm=None, savepath=None):
    '''
    :param targets: ground truth
    :param preds: prediction probs
    :param cm: confusion matrix
    :param savepath: confusion matrix picture savepth
    '''

    plt.figure(figsize=(40, 20), dpi=100)
    if cm is None:
        cm = confusion_matrix(targets, preds, labels=[0, 1])

    plt.matshow(cm, cmap=plt.cm.Oranges)
    plt.colorbar()

    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(y, x), verticalalignment='center', horizontalalignment='center')
    plt.ylabel('True')
    plt.xlabel('Prediction')
    if savepath:
        plt.savefig(savepath, dpi=100)
    else:
        plt.show()
    plt.close("all")


if __name__ == "__main__1":
    '''multi_class'''
    targets = np.array([0, 1, 2, 1, 3])
    probs = np.array([[0.9, 0.05, 0.05, 0], [0.5, 0.45, 0.05, 0], [0.4, 0.05, 0.55, 0], [0.1, 0.55, 0.25, 0.1], [0.4, 0.25, 0.35, 0]])
    print(metrics_multi_class(targets, probs))

    targets = np.array([0, 1, 2, 3, 3])
    probs = np.array([[0.9, 0.05, 0.05, 0], [0.5, 0.45, 0.05, 0], [0.4, 0.05, 0.55, 0], [0.1, 0.25, 0.25, 0.4], [0.1, 0.25, 0.25, 0.4]])
    print(metrics_multi_class(targets, probs))
    targets = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1]])
    probs = np.array([[0.9, 0.05, 0.05, 0], [0.5, 0.45, 0.05, 0], [0.4, 0.05, 0.55, 0], [0.1, 0.25, 0.25, 0.4], [0.1, 0.25, 0.25, 0.4]])
    print(metrics_multi_class(targets, probs))

    '''binary'''
    targets = np.array([0, 0, 1, 1])
    probs = np.array([[0.1], [0.1], [0.1], [0.9]])
    print(metrics_binary(targets, probs))

    targets = np.array([[0], [0], [1], [1]])
    probs = np.array([[0.1], [0.1], [0.1], [0.9]])
    print(metrics_binary(targets, probs))

    targets = np.array([0, 0, 1, 1])
    probs = np.array([[0.1, 0.1, 0.1, 0.9]])
    print(metrics_binary(targets, probs))

    targets = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.1, 0.1, 0.9])
    print(metrics_binary(targets, probs))

    targets = np.array([0, 1, 2, 1, 3])
    probs = np.array([[0.9, 0.05, 0.05, 0], [0.5, 0.45, 0.05, 0], [0.4, 0.05, 0.55, 0], [0.1, 0.55, 0.25, 0.1], [0.4, 0.25, 0.25, 0.1]])
    z = probs.shape[1]
    # print(z)
    print(np.eye(z))
    new_targets = np.eye(z)[targets]
    print(new_targets)


if __name__ == "__main__":
    import numpy as np
    import math
    # total collagen (mg/g dry weight of liver)
    x = np.array([7.1, 7.1, 7.2, 8.3, 9.4, 10.5, 11.4])
    # free proline (μ mole/g dry weight of liver)
    y = np.array([2.8, 2.9, 2.8, 2.6, 3.5, 4.6, 5.0])
    print(metrics_regression(x, y))
    sum = 0
    for idx in range(len(y)):
        sum += (y[idx] - x[idx]) * (y[idx] - x[idx])
    print(math.sqrt(sum/len(y)))

    import random
    n = 4
    seq_lens = [7, 8, 9, 10]
    targets = []
    probs = []
    for i in range(n):
        targets.append([])
        probs.append([])
        for j in range(max(seq_lens)):
            targets[i].append([])
            probs[i].append([])
            for k in range(max(seq_lens)):
                if j >= seq_lens[i] or k >= seq_lens[i]:
                    t_v = -100
                elif random.random() > 0.5:
                    t_v = 1
                else:
                    t_v = 0
                targets[i][j].append(t_v)
                p = random.random()
                probs[i][j].append(p)
    targets = np.array(targets)
    probs = np.array(probs)
    print(targets)
    print(probs)

    res = metrics_contactmap(targets, probs, threshold=0.5, ranges=[1, 5])
    print(res)