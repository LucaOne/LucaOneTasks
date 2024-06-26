#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.**@**.com
@tel: 137****6540
@datetime: 2022/12/10 20:18
@project: LucaOneTasks
@file: predict
@desc: predict or inference for trained downstream models
'''

import csv
import json
import os, sys
import torch
import codecs
import time
import numpy as np
import argparse
from collections import OrderedDict
from subword_nmt.apply_bpe import BPE
from transformers import BertConfig
sys.path.append(".")
sys.path.append("..")
sys.path.append("../src")
try:
    from .utils import to_device, device_memory, available_gpu_id, load_labels
    from .common.multi_label_metrics import relevant_indexes
    from .encoder import Encoder
    from .batch_converter import BatchConverter
    from .common.alphabet import Alphabet
    from .file_operator import csv_reader, fasta_reader, csv_writer, file_reader
    from .common.luca_base import LucaBase
    from .ppi.models.LucaPPI import LucaPPI
    from .ppi.models.LucaPPI2 import LucaPPI2
except ImportError:
    from src.utils import to_device, device_memory, available_gpu_id, load_labels
    from src.common.multi_label_metrics import relevant_indexes
    from src.encoder import Encoder
    from src.batch_converter import BatchConverter
    from src.common.alphabet import Alphabet
    from src.file_operator import csv_reader, fasta_reader, csv_writer, file_reader
    from src.common.luca_base import LucaBase
    from src.ppi.models.LucaPPI import LucaPPI
    from src.ppi.models.LucaPPI2 import LucaPPI2


def transform_one_sample_2_feature(device,
                                   input_mode,
                                   encoder,
                                   batch_convecter,
                                   row):
    batch_info = []
    if input_mode in ["pair"]:
        en = encoder.encode_pair(row[0],
                                 row[1],
                                 row[2],
                                 row[3],
                                 row[4],
                                 row[5],
                                 vector_filename_a=None,
                                 vector_filename_b=None,
                                 matrix_filename_a=None,
                                 matrix_filename_b=None,
                                 label=None
                                 )
        en_list = en
        batch_info.append([row[0], row[1], row[4], row[5]])
        seq_lens = [len(row[4]), len(row[5])]
    else:
        seq_lens = []
        en_list = []
        cur_seq = row[2]
        if batch_convecter.task_level_type not in ["seq_level", "seq-level"]:
            split_seqs = []
            max_len = 1024 - int(batch_convecter.prepend_bos) - int(batch_convecter.append_eos)
            while max_len < len(cur_seq):
                split_seqs.append(cur_seq[:max_len])
                seq_lens.append(max_len)
                cur_seq = cur_seq[max_len:]

            if len(cur_seq) > 0:
                split_seqs.append(cur_seq)
                seq_lens.append(len(cur_seq))
            for split_seq in split_seqs:
                en = encoder.encode_single(row[0],
                                           row[1],
                                           split_seq,
                                           vector_filename=None,
                                           matrix_filename=None,
                                           label=None
                                           )
                en_list.append(en)
        else:
            en = encoder.encode_single(row[0],
                                       row[1],
                                       row[2],
                                       vector_filename=None,
                                       matrix_filename=None,
                                       label=None
                                       )
            en_list = en
            seq_lens = len(row[2])
            if "matrix" in en and en["matrix"] is not None:
                seq_lens = min(seq_lens, en["matrix"].shape[0] - int(batch_convecter.prepend_bos) - int(batch_convecter.append_eos))
        batch_info.append([row[0], row[2]])
    batch = [en_list]
    if isinstance(batch[0], list):
        batch_features = []
        for cur_batch in batch[0]:
            cur_batch_features = batch_convecter([cur_batch])
            cur_batch_features, cur_sample_num = to_device(device, cur_batch_features)
            batch_features.append(cur_batch_features)

    else:
        batch_features = batch_convecter(batch)
        batch_features, cur_sample_num = to_device(device, batch_features)
    return batch_info, batch_features, [seq_lens]


def predict_probs(args, encoder, batch_convecter, model, row):
    batch_info, batch_features, seq_lens = transform_one_sample_2_feature(args.device, args.input_mode, encoder, batch_convecter, row)
    if isinstance(batch_features, list):
        probs = []
        for cur_batch_features in batch_features:
            cur_probs = model(**cur_batch_features)[1]
            if cur_probs.is_cuda:
                cur_probs = cur_probs.detach().cpu().numpy()
            else:
                cur_probs = cur_probs.detach().numpy()
            probs.append(cur_probs)
    else:
        probs = model(**batch_features)[1]
        if probs.is_cuda:
            probs = probs.detach().cpu().numpy()
        else:
            probs = probs.detach().numpy()
    return batch_info, probs, seq_lens


def predict_token_level_binary_class(args, encoder, batch_convecter, label_id_2_name, model, row):
    # to do
    pass


def predict_token_level_multi_class(args, encoder, batch_convecter, label_id_2_name, model, row):
    # to do
    pass


def predict_token_level_multi_label(args, encoder, batch_convecter, label_id_2_name, model, row):
    # to do
    pass


def predict_token_level_regression(args, encoder, batch_convecter, label_id_2_name, model, row):
    # to do
    pass


def predict_seq_level_binary_class(args, encoder, batch_convecter, label_id_2_name, model, row):
    batch_info, probs, seq_lens = predict_probs(args, encoder, batch_convecter, model, row)
    # print("probs dim: ", probs.ndim)
    preds = (probs >= args.threshold).astype(int).flatten()
    res = []
    for idx, info in enumerate(batch_info):
        if args.input_mode == "pair":
            cur_res = [info[0], info[1], info[2], info[3], float(probs[idx][0]), label_id_2_name[preds[idx]]]
            if len(info) > 4:
                cur_res += info[4:]
        else:
            cur_res = [info[0], info[1], float(probs[idx][0]), label_id_2_name[preds[idx]]]
            if len(info) > 2:
                cur_res += info[2:]
        res.append(cur_res)
    return res


def predict_seq_level_multi_class(args, encoder, batch_convecter, label_id_2_name, model, row, topk=5):
    batch_info, probs, seq_lens = predict_probs(args, encoder, batch_convecter, model, row)
    # print("probs dim: ", probs.ndim)

    if topk is not None and topk > 1:
        # print("topk: %d" % topk)
        preds = np.argmax(probs, axis=-1)
        preds_topk = np.argsort(probs, axis=-1)[:, ::-1][:, :topk]
        res = []
        for idx, info in enumerate(batch_info):
            cur_topk_probs = []
            cur_topk_labels = []
            for label_idx in preds_topk[idx]:
                cur_topk_probs.append(float(probs[idx][label_idx]))
                cur_topk_labels.append(label_id_2_name[label_idx])
            if args.input_mode == "pair":
                cur_res = [
                    info[0],
                    info[1],
                    info[2],
                    info[3],
                    float(probs[idx][preds[idx]]),
                    label_id_2_name[preds[idx]],
                    cur_topk_probs,
                    cur_topk_labels
                ]
                if len(info) > 4:
                    cur_res += info[4:]
            else:
                cur_res = [
                    info[0],
                    info[1],
                    float(probs[idx][preds[idx]]),
                    label_id_2_name[preds[idx]],
                    cur_topk_probs,
                    cur_topk_labels
                ]
                if len(info) > 2:
                    cur_res += info[2:]
            res.append(cur_res)
        return res
    else:
        preds = np.argmax(probs, axis=-1)
        res = []
        for idx, info in enumerate(batch_info):
            if args.input_mode == "pair":
                cur_res = [info[0], info[1], info[2], info[3],  float(probs[idx][preds[idx]]), label_id_2_name[preds[idx]]]
                if len(info) > 4:
                    cur_res += info[4:]
            else:
                cur_res = [info[0], info[1], float(probs[idx][preds[idx]]), label_id_2_name[preds[idx]]]
                if len(info) > 2:
                    cur_res += info[2:]
            res.append(cur_res)
        return res


def predict_seq_level_multi_label(args, encoder, batch_convecter, label_id_2_name, model, row):
    batch_info, probs, seq_lens = predict_probs(args, encoder, batch_convecter, model, row)
    # print("probs dim: ", probs.ndim)
    preds = relevant_indexes((probs >= args.threshold).astype(int))
    res = []
    for idx, info in enumerate(batch_info):
        if args.input_mode == "pair":
            cur_res = [info[0], info[1], info[2], info[3], [float(probs[idx][label_index]) for label_index in preds[idx]], [label_id_2_name[label_index] for label_index in preds[idx]]]
            if len(info) > 4:
                cur_res += info[4:]
        else:
            cur_res = [info[0], info[1], [float(probs[idx][label_index]) for label_index in preds[idx]], [label_id_2_name[label_index] for label_index in preds[idx]]]
            if len(info) > 2:
                cur_res += info[2:]
        res.append(cur_res)
    return res


def predict_seq_level_regression(args, encoder, batch_convecter, label_id_2_name, model, row):
    # to do
    pass


def load_tokenizer(args, model_dir, seq_tokenizer_class):
    seq_subword, seq_tokenizer = None, None
    if not hasattr("args", "has_seq_encoder") or args.has_seq_encoder:
        if args.seq_subword:
            if os.path.exists(os.path.join(model_dir, "sequence")):
                seq_tokenizer = seq_tokenizer_class.from_pretrained(os.path.join(model_dir, "sequence"), do_lower_case=args.do_lower_case)
            else:
                seq_tokenizer = seq_tokenizer_class.from_pretrained(os.path.join(model_dir, "tokenizer"), do_lower_case=args.do_lower_case)
            bpe_codes_prot = codecs.open(args.codes_file)
            seq_subword = BPE(bpe_codes_prot, merges=-1, separator='')
        else:
            seq_subword = None
            seq_tokenizer = seq_tokenizer_class.from_predefined(args.seq_vocab_path)
            if args.not_prepend_bos:
                seq_tokenizer.prepend_bos = False
            if args.not_append_eos:
                seq_tokenizer.append_eos = False
    return seq_subword, seq_tokenizer


def load_trained_model(model_config, args, model_class, model_dirpath):
    # load exists checkpoint
    print("load pretrained model: %s" % model_dirpath)
    try:
        model = model_class.from_pretrained(model_dirpath, args=args)
    except Exception as e:
        model = model_class(model_config, args=args)
        pretrained_net_dict = torch.load(os.path.join(args.model_dirpath, "pytorch.pth"),
                                         map_location=torch.device("cpu"))
        model_state_dict_keys = set()
        for key in model.state_dict():
            model_state_dict_keys.add(key)
        new_state_dict = OrderedDict()
        for k, v in pretrained_net_dict.items():
            if k.startswith("module."):
                # remove `module.`
                name = k[7:]
            else:
                name = k
            if name in model_state_dict_keys:
                new_state_dict[name] = v
        # print("diff:")
        # print(model_state_dict_keys.difference(new_state_dict.keys()))
        model.load_state_dict(new_state_dict)
    model.to(args.device)
    model.eval()
    return model


def load_model(args, model_name, model_dir):
    # load tokenizer and model
    begin_time = time.time()
    device = torch.device(args.device)
    print("load model on cuda:", device)
  
    if args.model_type in ["ppi", "lucappi"]:
        config_class, seq_tokenizer_class, model_class = BertConfig, Alphabet, LucaPPI
    elif args.model_type in ["lucappi2"]:
        config_class, seq_tokenizer_class, model_class = BertConfig, Alphabet, LucaPPI2
    elif args.model_type in ["luca_base"]:
        config_class, seq_tokenizer_class, model_class = BertConfig, Alphabet, LucaBase
    else:
        raise Exception("Not support the model_type=%s" % args.model_type)
    seq_subword, seq_tokenizer = load_tokenizer(args, model_dir, seq_tokenizer_class)

    # config = config_class(**json.load(open(os.path.join(model_dir, "config.json"), "r"), encoding="UTF-8"))
    model_config = config_class(**json.load(open(os.path.join(model_dir, "config.json"), "r")))

    model = load_trained_model(model_config, args, model_class, model_dir)
    print("the time for loading model:", time.time() - begin_time)

    return model_config, seq_subword, seq_tokenizer, model


def create_encoder_batch_convecter(model_args, seq_subword, seq_tokenizer):
    if hasattr(model_args, "input_mode") and model_args.input_mode in ["pair"]:
        assert model_args.seq_max_length is not None or (model_args.seq_max_length_a is not None and model_args.seq_max_length_b is not None)
        if model_args.seq_max_length is None:
            model_args.seq_max_length = max(model_args.seq_max_length_a, model_args.seq_max_length_b)
        # encoder_config
        encoder_config = {
            "llm_type": model_args.llm_type,
            "llm_version": model_args.llm_version,
            "llm_step": model_args.llm_step,
            "llm_dirpath": model_args.llm_dirpath,
            "input_type": model_args.input_type,
            "trunc_type": model_args.trunc_type,
            "seq_max_length": model_args.seq_max_length,
            "atom_seq_max_length": None,
            "vector_dirpath": model_args.vector_dirpath,
            "matrix_dirpath": model_args.matrix_dirpath,
            "local_rank": model_args.gpu_id,
            "max_sentence_length": model_args.max_sentence_length if hasattr(model_args, "max_sentence_length") else None,
            "max_sentences": model_args.max_sentences if hasattr(model_args, "max_sentences") else None,
            "embedding_complete": model_args.embedding_complete,
            "embedding_complete_seg_overlap": model_args.embedding_complete_seg_overlap
        }
    else:
        assert model_args.seq_max_length is not None
        encoder_config = {
            "llm_type": model_args.llm_type,
            "llm_version": model_args.llm_version,
            "llm_step": model_args.llm_step,
            "llm_dirpath": model_args.llm_dirpath,
            "input_type": model_args.input_type,
            "trunc_type": model_args.trunc_type,
            "seq_max_length": model_args.seq_max_length,
            "atom_seq_max_length": None,
            "vector_dirpath": model_args.vector_dirpath,
            "matrix_dirpath": model_args.matrix_dirpath,
            "matrix_add_special_token": model_args.matrix_add_special_token,
            "local_rank": model_args.gpu_id,
            "max_sentence_length": model_args.max_sentence_length if hasattr(model_args, "max_sentence_length") else None,
            "max_sentences": model_args.max_sentences if hasattr(model_args, "max_sentences") else None,
            "embedding_complete": model_args.embedding_complete,
            "embedding_complete_seg_overlap": model_args.embedding_complete_seg_overlap
        }
    encoder = Encoder(**encoder_config)

    batch_converter = BatchConverter(
        task_level_type=model_args.task_level_type,
        label_size=model_args.label_size,
        output_mode=model_args.output_mode,
        seq_subword=seq_subword,
        seq_tokenizer=seq_tokenizer,
        no_position_embeddings=model_args.no_position_embeddings,
        no_token_type_embeddings=model_args.no_token_type_embeddings,
        truncation_seq_length=model_args.truncation_seq_length if hasattr(model_args, "truncation_seq_length") else model_args.seq_max_length,
        truncation_matrix_length=model_args.truncation_matrix_length if hasattr(model_args, "truncation_matrix_length") else model_args.matrix_max_length,
        trunc_type=model_args.trunc_type if hasattr(model_args, "trunc_type") else "right",
        atom_tokenizer=None,
        atom_truncation_seq_length=None,
        atom_truncation_matrix_length=None,
        padding_idx=0,
        unk_idx=1,
        cls_idx=2,
        eos_idx=3,
        mask_idx=4,
        ignore_index=model_args.ignore_index,
        non_ignore=model_args.non_ignore,
        prepend_bos=not model_args.not_prepend_bos,
        append_eos=not model_args.not_append_eos,
        max_sentence_length=model_args.max_sentence_length if hasattr(model_args, "max_sentence_length") else None,
        max_sentences=model_args.max_sentences if hasattr(model_args, "max_sentences") else None,
        matrix_add_special_token=model_args.matrix_add_special_token if hasattr(model_args, "matrix_add_special_token") else False
    )
    return encoder, batch_converter


def run(sequences, llm_truncation_seq_length, model_path, dataset_name, dataset_type, task_type, task_level_type, model_type, input_type, time_str, step, gpu_id, threshold, topk):
    model_dir = "%s/models/%s/%s/%s/%s/%s/%s/%s" % (model_path, dataset_name, dataset_type, task_type, model_type, input_type,
                                                    time_str, step if step == "best" else "checkpoint-{}".format(step))
    config_dir = "%s/logs/%s/%s/%s/%s/%s/%s" % (model_path, dataset_name, dataset_type, task_type, model_type, input_type,
                                                time_str)

    model_args = torch.load(os.path.join(model_dir, "training_args.bin"))
    print("------model args------")
    print(model_args.__dict__)
    print("-" * 50)
    model_args.llm_truncation_seq_length = llm_truncation_seq_length
    model_args.seq_max_length = llm_truncation_seq_length
    model_args.atom_seq_max_length = None # to do
    model_args.truncation_seq_length = model_args.seq_max_length
    model_args.truncation_matrix_length = model_args.matrix_max_length
    model_args.emb_dir = None
    model_args.vector_dirpath = None
    model_args.matrix_dirpath = None
    model_args.dataset_name = dataset_name
    model_args.dataset_type = dataset_type
    model_args.task_type = task_type
    model_args.model_type = model_type
    model_args.time_str = time_str
    model_args.step = step
    model_args.task_level_type = task_level_type
    model_args.gpu_id = gpu_id
    if not hasattr(model_args, "embedding_complete"):
        model_args.embedding_complete = False
    if not hasattr(model_args, "embedding_complete_seg_overlap"):
        model_args.embedding_complete_seg_overlap = False

    if not hasattr(model_args, "non_ignore"):
        model_args.non_ignore = True
    model_args.threshold = threshold

    if model_args.label_filepath:
        model_args.label_filepath = model_args.label_filepath.replace("../", "%s/" % model_path)
    if not os.path.exists(model_args.label_filepath):
        model_args.label_filepath = os.path.join(config_dir, "label.txt")

    if gpu_id is None:
        gpu_id = available_gpu_id()
        model_args.gpu_id = gpu_id
    print("gpu_id: %d" % gpu_id)
    model_args.device = torch.device("cuda:%d" % gpu_id if gpu_id > -1 else "cpu")
    print("Before loading the model:")
    device_memory(None if gpu_id == -1 else gpu_id)

    # Step2: loading the tokenizer and model
    model_config, seq_subword, seq_tokenizer, trained_model = load_model(model_args, model_type, model_dir)
    print("After loading the model:")
    
    device_memory(None if gpu_id == -1 else gpu_id)
    encoder, batch_convecter = create_encoder_batch_convecter(model_args, seq_subword, seq_tokenizer)
    label_list = load_labels(model_args.label_filepath)
    label_id_2_name = {idx: name for idx, name in enumerate(label_list)}
    # llm_model, llm_alphabet, llm_args_info, llm_model_config, llm_version = None, None, None, None, None
    # Step 3: prediction
    if model_args.task_level_type in ["seq_level", "seq-level"] and task_type in ["binary_class", "binary-class"]:
        predict_func = predict_seq_level_binary_class
    elif model_args.task_level_type in ["seq_level", "seq-level"] and task_type in ["multi_class", "multi-class"]:
        predict_func = predict_seq_level_multi_class
    elif model_args.task_level_type in ["seq_level", "seq-level"] and task_type in ["multi_label", "multi-label"]:
        predict_func = predict_seq_level_multi_label
    elif model_args.task_level_type in ["seq_level", "seq-level"] and task_type in ["regression"]:
        predict_func = predict_seq_level_regression
    elif model_args.task_level_type not in ["seq_level", "seq-level"] and task_type in ["binary_class", "binary-class"]:
        predict_func = predict_token_level_binary_class
    elif model_args.task_level_type not in ["seq_level", "seq-level"] and task_type in ["multi_class", "multi-class"]:
        predict_func = predict_token_level_multi_class
    elif model_args.task_level_type not in ["seq_level", "seq-level"] and task_type in ["multi_label", "multi-label"]:
        predict_func = predict_token_level_multi_label
    elif model_args.task_level_type not in ["seq_level", "seq-level"] and task_type in ["regression"]:
        predict_func = predict_token_level_regression
    else:
        raise Exception("task type or task level type error")
    results = []
    print("device:", model_args.device)
    if hasattr(model_args, "input_mode") and model_args.input_mode in ["pair"]:
        for item in sequences:
            seq_id_a = item[0]
            seq_id_b = item[1]
            seq_type_a = item[2]
            seq_type_b = item[3]
            seq_a = item[4]
            seq_b = item[5]
            row = [seq_id_a, seq_id_b, seq_type_a, seq_type_b, seq_a, seq_b]
            if task_level_type in ["seq_level", "seq-level"] and task_type in ["multi_class", "multi-class"]:
                cur_res = predict_func(model_args,
                                       encoder,
                                       batch_convecter,
                                       label_id_2_name,
                                       trained_model,
                                       row,
                                       topk=topk)
                if topk is not None and topk > 1:
                    results.append([seq_id_a, seq_id_b, seq_a, seq_b, cur_res[0][4], cur_res[0][5], cur_res[0][6], cur_res[0][7]])
                else:
                    results.append([seq_id_a, seq_id_b, seq_a, seq_b, cur_res[0][4], cur_res[0][5]])
            else:
                cur_res = predict_func(model_args,
                                       encoder,
                                       batch_convecter,
                                       label_id_2_name,
                                       trained_model,
                                       row)
                results.append([seq_id_a, seq_id_b, seq_a, seq_b, cur_res[0][4], cur_res[0][5]])
    else:
        for item in sequences:
            seq_id = item[0]
            seq_type = item[1]
            seq = item[2]
            row = [seq_id, seq_type, seq]
            if task_level_type in ["seq_level", "seq-level"] and task_type in ["multi_class", "multi-class"]:
                # print("task_level_type: %s, task_type: %s" % (task_level_type, task_type))
                cur_res = predict_func(model_args,
                                       encoder,
                                       batch_convecter,
                                       label_id_2_name,
                                       trained_model,
                                       row,
                                       topk=topk)
                if topk is not None and topk > 1:
                    results.append([seq_id, seq, cur_res[0][2], cur_res[0][3], cur_res[0][4], cur_res[0][5]])
                else:
                    results.append([seq_id, seq, cur_res[0][2], cur_res[0][3]])
            else:
                cur_res = predict_func(model_args,
                                       encoder,
                                       batch_convecter,
                                       label_id_2_name,
                                       trained_model,
                                       row)
                results.append([seq_id, seq, cur_res[0][2], cur_res[0][3]])
    torch.cuda.empty_cache()
    return results


def run_args():
    parser = argparse.ArgumentParser(description="Prediction")
    parser.add_argument("--seq_id", default=None, type=str,  help="the seq id")
    parser.add_argument("--seq", default=None, type=str,  help="the sequence")
    parser.add_argument("--seq_type", default="prot", type=str, choices=["prot", "gene"], help="seq type.")
    parser.add_argument("--fasta", default=None, type=str, help="the sequences fasta file")
    parser.add_argument("--llm_truncation_seq_length", default=4096, type=int, required=True, help="truncation seq length for llm")
    parser.add_argument("--topk", default=None, type=int, help="topk for multi-class")
    parser.add_argument("--model_path", default=None, type=str, help="the model dir. default: None")
    parser.add_argument("--save_path", default=None, type=str, help="the result save path")
    parser.add_argument("--emb_dir", default=None, type=str, help="the structural embedding save dir. default: None")
    parser.add_argument("--dataset_name", default=None, type=str, required=True, help="the dataset name for model buliding.")
    parser.add_argument("--dataset_type", default=None, type=str, required=True, help="the dataset type for model buliding.")
    parser.add_argument("--task_type", default=None, type=str, required=True, choices=["multi_label", "multi_class", "binary_class"], help="the task type for model buliding.")
    parser.add_argument("--task_level_type", default=None, type=str, required=True, choices=["seq_level", "token_level"], help="the task level type for model buliding.")
    parser.add_argument("--model_type", default=None, type=str, required=True, help="model type.")
    parser.add_argument("--input_type", default=None, type=str, required=True, help="input type.")
    parser.add_argument("--input_mode", default=None, type=str, required=True, help="input mode.")
    parser.add_argument("--ground_truth_idx", default=None, type=int, help="ground truth idx")
    parser.add_argument("--per_num", default=10000, type=int, help="per num to print")
    parser.add_argument("--time_str", default=None, type=str, required=True, help="the running time string(yyyymmddHimiss) of model building.")
    parser.add_argument("--step", default=None, type=str, required=True, help="the training global step of model finalization.")
    parser.add_argument("--gpu_id", default=None, type=int, help="gpu_id.")
    parser.add_argument("--threshold",  default=0.5, type=float, help="sigmoid threshold for binary-class or multi-label classification, None for multi-class classification, defualt: 0.5.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = run_args()
    print("------run args------")
    print(args.__dict__)
    print("-" * 50)
    if args.fasta is not None and os.path.exists(args.fasta):
        exists_ids = set()
        exists_res = []
        if os.path.exists(args.save_path):
            print("save_path=%s exists." % args.save_path)
            if args.input_mode == "pair":
                for row in csv_reader(args.save_path, header=True, header_filter=True):
                    if len(row) < 4:
                        continue
                    exists_ids.add(row[0] + "_" + row[1])
                    exists_res.append(row)
                print("exists records: %d" % len(exists_res))
            else:
                for row in csv_reader(args.save_path, header=True, header_filter=True):
                    if len(row) < 4:
                        continue
                    exists_ids.add(row[0])
                    exists_res.append(row)
                print("exists records: %d" % len(exists_res))
        elif not os.path.exists(os.path.dirname(args.save_path)):
            os.makedirs(os.path.dirname(args.save_path))
        with open(args.save_path, "w") as wfp:
            writer = csv.writer(wfp)
            if args.input_mode == "pair":
                if args.task_type == "multi_class" and args.topk is not None and args.topk > 1 and args.task_level_type == "seq_level":
                    header = ["seq_id_a", "seq_id_b", "seq_a", "seq_b", "top1_prob", "top1_label", "top%d_probs" % args.topk, "top%d_labels" % args.topk]
                else:
                    header = ["seq_id_a", "seq_id_b", "seq_a", "seq_b", "prob", "label"]
            else:
                if args.task_type == "multi_class" and args.topk is not None and args.topk > 1 and args.task_level_type == "seq_level":
                    header = ["seq_id", "seq", "top1_prob", "top1_label", "top%d_probs" % args.topk, "top%d_labels" % args.topk]
                else:
                    header = ["seq_id", "seq", "prob", "label"]
            if args.ground_truth_idx is not None and args.ground_truth_idx >= 0:
                header.append("ground_truth")
            writer.writerow(header)
            for item in exists_res:
                writer.writerow(item)
            exists_res = []
            batch_data = []
            batch_ground_truth = []
            had_done = 0
            reader = file_reader(args.fasta) if args.fasta.endswith(".csv") or args.fasta.endswith(".tsv") else fasta_reader(args.fasta)
            for row in reader:
                if args.input_mode == "pair":
                    if row[0] + "_" + row[1] in exists_ids:
                        continue
                    batch_data.append([row[0], row[1], row[2], row[3], row[4], row[5]])
                    if args.ground_truth_idx is not None and args.ground_truth_idx >= 0:
                        batch_ground_truth.append(row[args.ground_truth_idx])
                else:
                    if row[0] in exists_ids:
                        continue
                    if len(row) == 2:
                        batch_data.append([row[0], "prot" if args.seq_type is None else args.seq_type, args.row[1]])
                    elif len(row) > 2:
                        if args.ground_truth_idx is not None and args.ground_truth_idx >= 0:
                            batch_ground_truth.append(row[args.ground_truth_idx])
                        batch_data.append([row[0], row[1], row[2]])
                    else:
                        continue

                if len(batch_data) % args.per_num == 0:
                    batch_results = run(batch_data,
                                        args.llm_truncation_seq_length,
                                        args.model_path,
                                        args.dataset_name,
                                        args.dataset_type,
                                        args.task_type,
                                        args.task_level_type,
                                        args.model_type,
                                        args.input_type,
                                        args.time_str,
                                        args.step,
                                        args.gpu_id,
                                        args.threshold,
                                        topk=args.topk)
                    for item_idx, item in enumerate(batch_results):
                        if args.ground_truth_idx is not None and args.ground_truth_idx >= 0:
                            item.append(batch_ground_truth[item_idx])
                        writer.writerow(item)
                    wfp.flush()
                    had_done += len(batch_data)
                    print("done %d, had_done: %d" % (len(batch_data), had_done))
                    batch_data = []
                    batch_ground_truth = []
            if len(batch_data) > 0:
                batch_results = run(batch_data,
                                    args.llm_truncation_seq_length,
                                    args.model_path,
                                    args.dataset_name,
                                    args.dataset_type,
                                    args.task_type,
                                    args.task_level_type,
                                    args.model_type,
                                    args.input_type,
                                    args.time_str,
                                    args.step,
                                    args.gpu_id,
                                    args.threshold,
                                    topk=args.topk)
                for item_idx, item in enumerate(batch_results):
                    if args.ground_truth_idx is not None and args.ground_truth_idx >= 0:
                        item.append(batch_ground_truth[item_idx])
                    writer.writerow(item)
                wfp.flush()
                batch_data = []
                batch_ground_truth = []
    elif args.seq_id is not None and args.seq is not None:
        data = [[args.seq_id, "prot" if args.seq_type is None else args.seq_type, args.seq]]
        results = run(data,
                      args.llm_truncation_seq_length,
                      args.model_path,
                      args.dataset_name,
                      args.dataset_type,
                      args.task_type,
                      args.task_level_type,
                      args.model_type,
                      args.input_type,
                      args.time_str,
                      args.step,
                      args.gpu_id,
                      args.threshold,
                      topk=args.topk)
        print("results:")
        print("seq_id=%s" % args.seq_id)
        print("seq=%s" % args.seq)
        print("prob=%f" % results[0][2])
        print("label=%s" % results[0][3])
    else:
        raise Exception("input error")
