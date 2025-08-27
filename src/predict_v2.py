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
import uuid
import os, sys
import torch
import codecs
import time, shutil
import numpy as np
import argparse
from datetime import datetime
from collections import OrderedDict
from subword_nmt.apply_bpe import BPE
from transformers import BertConfig
sys.path.append(".")
sys.path.append("..")
sys.path.append("../src")
try:
    from utils import to_device, device_memory, available_gpu_id, load_labels, seq_type_is_match_seq, \
        download_trained_checkpoint_lucaone, download_trained_checkpoint_downstream_tasks
    from common.multi_label_metrics import relevant_indexes
    from common.model_config import LucaConfig
    from encoder import Encoder
    from batch_converter import BatchConverter
    from common.alphabet import Alphabet
    from file_operator import csv_reader, fasta_reader, csv_writer, file_reader
    from common.luca_base import LucaBase
    from ppi.models.LucaPPI import LucaPPI
    from ppi.models.LucaPPI2 import LucaPPI2
    from ppi.models.LucaPairHomo import LucaPairHomo
    from ppi.models.LucaPairHeter import LucaPairHeter
    from ppi.models.LucaPairIntraInter import LucaPairIntraInter
except ImportError:
    from src.utils import to_device, device_memory, available_gpu_id, load_labels, seq_type_is_match_seq, \
        download_trained_checkpoint_lucaone, download_trained_checkpoint_downstream_tasks
    from src.common.multi_label_metrics import relevant_indexes
    from src.common.model_config import LucaConfig
    from src.encoder import Encoder
    from src.batch_converter import BatchConverter
    from src.common.alphabet import Alphabet
    from src.file_operator import csv_reader, fasta_reader, csv_writer, file_reader
    from src.common.luca_base import LucaBase
    from src.ppi.models.LucaPPI import LucaPPI
    from src.ppi.models.LucaPPI2 import LucaPPI2
    from src.ppi.models.LucaPairHomo import LucaPairHomo
    from src.ppi.models.LucaPairHeter import LucaPairHeter
    from src.ppi.models.LucaPairIntraInter import LucaPairIntraInter


def transform_one_sample_2_feature(
        device,
        input_mode,
        input_type,
        encoder,
        batch_convecter,
        row
):
    batch_info = []
    sample_ids = []
    if input_mode == "pair":
        sample_ids.append(row[0] + "#" + row[1])
        if input_type in ["seq_vs_seq", "seq_vs_vector", "seq_vs_matrix", "vector_vs_vector",
                          "vector_vs_matrix", "matrix_vs_matrix", "matrix_express_vs_matrix", "matrix_express_vs_matrix_express"]:
            if input_type == "matrix_express_vs_matrix":
                en = encoder.encode_pair(
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                    vector_filename_a=row[6],
                    vector_filename_b=row[7],
                    matrix_filename_a=row[8],
                    matrix_filename_b=row[9],
                    express_list_a=row[10],
                    label=None
                )
            elif input_type == "matrix_express_vs_matrix_express":
                en = encoder.encode_pair(
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                    vector_filename_a=row[6],
                    vector_filename_b=row[7],
                    matrix_filename_a=row[8],
                    matrix_filename_b=row[9],
                    express_list_a=row[10],
                    express_list_b=row[11],
                    label=None
                )
            else:
                en = encoder.encode_pair(
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                    vector_filename_a=row[6],
                    vector_filename_b=row[7],
                    matrix_filename_a=row[8],
                    matrix_filename_b=row[9],
                    label=None
                )
            en_list = en
            if input_type == "seq_vs_seq":
                batch_info.append([row[0], row[1], row[4], row[5]])
                seq_lens = [len(row[4]), len(row[5])]
            elif input_type == "seq_vs_vector":
                batch_info.append([row[0], row[1], row[4], row[7]])
                seq_lens = [len(row[4]), 1]
            elif input_type == "seq_vs_matrix":
                batch_info.append([row[0], row[1], row[4], row[9]])
                seq_lens = [len(row[4]), en["matrix_b"].shape[0]]
            elif input_type == "vector_vs_vector":
                batch_info.append([row[0], row[1], row[6], row[7]])
                seq_lens = [1, 1]
            elif input_type == "vector_vs_matrix":
                batch_info.append([row[0], row[1], row[6], row[9]])
                seq_lens = [1, en["matrix_b"].shape[0]]
            elif input_type == "matrix_vs_matrix":
                batch_info.append([row[0], row[1], row[8], row[9]])
                seq_lens = [en["matrix_a"].shape[0], en["matrix_b"].shape[0]]
            elif input_type == "matrix_express_vs_matrix":
                batch_info.append([row[0], row[1], row[8], row[9]])
                seq_lens = [en["matrix_a"].shape[0], en["matrix_b"].shape[0]]
            elif input_type == "matrix_express_vs_matrix_express":
                batch_info.append([row[0], row[1], row[8], row[9]])
                seq_lens = [en["matrix_a"].shape[0], en["matrix_b"].shape[0]]
            else:
                raise Exception("Not support input_mode=%s, input_type=%s" % (input_mode, input_type))
        else:
            en = encoder.encode_pair(
                row[0],
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
        sample_ids.append(row[0])
        seq_lens = []
        en_list = []
        cur_seq = row[2]
        if batch_convecter.task_level_type not in ["seq_level", "seq-level"]:
            split_seqs = []
            # segment for long sequence in token-level task
            max_len = 10240 - int(batch_convecter.prepend_bos) - int(batch_convecter.append_eos)
            while max_len < len(cur_seq):
                split_seqs.append(cur_seq[:max_len])
                seq_lens.append(max_len)
                cur_seq = cur_seq[max_len:]
            if len(cur_seq) > 0:
                split_seqs.append(cur_seq)
                seq_lens.append(len(cur_seq))
            for split_seq in split_seqs:
                en = encoder.encode_single(
                    row[0],
                    row[1],
                    split_seq,
                    vector_filename=None,
                    matrix_filename=None,
                    label=None
                )
                en_list.append(en)
        else:
            en = encoder.encode_single(
                row[0],
                row[1],
                row[2],
                vector_filename=row[3] if len(row) > 3 else None,
                matrix_filename=row[4] if len(row) > 4 else None,
                express_list=row[5] if len(row) > 5 else None,
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
    return batch_info, batch_features, [seq_lens], sample_ids


def predict_probs(args, encoder, batch_convecter, model, row):
    model.to(torch.device("cpu"))
    batch_info, batch_features, seq_lens, sample_ids = transform_one_sample_2_feature(
        args.device,
        args.input_mode,
        args.input_type,
        encoder,
        batch_convecter,
        row
    )
    model.to(args.device)
    if isinstance(batch_features, list):
        probs = []
        for cur_batch_features in batch_features:
            cur_probs = model(
                **cur_batch_features,
                sample_ids=sample_ids,
                attention_scores_savepath=args.output_attention_scores_dirpath,
                attention_pooling_scores_savepath=args.output_attention_pooling_scores_dirpath,
                output_classification_vector_dirpath=args.output_classification_vector_dirpath
            )[1]
            if cur_probs.is_cuda:
                cur_probs = cur_probs.detach().cpu().numpy()
            else:
                cur_probs = cur_probs.detach().numpy()
            probs.append(cur_probs)
    else:
        probs = model(
            **batch_features,
            sample_ids=sample_ids,
            attention_scores_savepath=args.output_attention_scores_dirpath,
            attention_pooling_scores_savepath=args.output_attention_pooling_scores_dirpath,
            output_classification_vector_dirpath=args.output_classification_vector_dirpath
        )[1]
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
    batch_info, probs, seq_lens = predict_probs(args, encoder, batch_convecter, model, row)
    res = []
    for idx, info in enumerate(batch_info):
        if args.input_mode == "pair":
            cur_res = [info[0], info[1], info[2], info[3], float(probs[idx][0]), str(probs[idx][0])]
            if len(info) > 4:
                cur_res += info[4:]
        else:
            cur_res = [info[0], info[1], float(probs[idx][0]), str(probs[idx][0])]
            if len(info) > 2:
                cur_res += info[2:]
        res.append(cur_res)
    return res


def load_tokenizer(args, model_dir, seq_tokenizer_class):
    seq_subword, seq_tokenizer = None, None
    if not hasattr(args, "has_seq_encoder") or args.has_seq_encoder:
        if args.seq_subword:
            if os.path.exists(os.path.join(model_dir, "sequence")):
                seq_tokenizer = seq_tokenizer_class.from_pretrained(os.path.join(model_dir, "sequence"), do_lower_case=args.do_lower_case)
            else:
                seq_tokenizer = seq_tokenizer_class.from_pretrained(os.path.join(model_dir, "tokenizer"), do_lower_case=args.do_lower_case)
            bpe_codes = codecs.open(args.codes_file)
            seq_subword = BPE(bpe_codes, merges=-1, separator='')
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
        pretrained_net_dict = torch.load(os.path.join(model_dirpath, "pytorch.pth"),
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
    elif args.model_type in ["lucapair_homo"]:
        config_class, seq_tokenizer_class, model_class = LucaConfig, Alphabet, LucaPairHomo
    elif args.model_type in ["lucapair_heter"]:
        config_class, seq_tokenizer_class, model_class = LucaConfig, Alphabet, LucaPairHeter
    elif args.model_type in ["lucapair_intrainter"]:
        config_class, seq_tokenizer_class, model_class = LucaConfig, Alphabet, LucaPairIntraInter
    else:
        raise Exception("Not support the model_type=%s" % args.model_type)
    seq_subword, seq_tokenizer = load_tokenizer(args, model_dir, seq_tokenizer_class)

    # config = config_class(**json.load(open(os.path.join(model_dir, "config.json"), "r"), encoding="UTF-8"))
    model_config = config_class(**json.load(open(os.path.join(model_dir, "config.json"), "r")))

    model = load_trained_model(model_config, args, model_class, model_dir)
    print("the time for loading model:", time.time() - begin_time)

    return model_config, seq_subword, seq_tokenizer, model


def create_encoder_batch_convecter(
        model_args,
        seq_subword,
        seq_tokenizer
):
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
            "prepend_bos": True,
            "append_eos": True,
            "vector_dirpath": model_args.vector_dirpath,
            "matrix_dirpath": model_args.matrix_dirpath,
            "local_rank": model_args.gpu_id,
            "max_sentence_length": model_args.max_sentence_length if hasattr(model_args, "max_sentence_length") else None,
            "max_sentences": model_args.max_sentences if hasattr(model_args, "max_sentences") else None,
            "embedding_complete": model_args.embedding_complete,
            "embedding_complete_seg_overlap": model_args.embedding_complete_seg_overlap,
            "matrix_add_special_token": model_args.matrix_add_special_token,
            "embedding_fixed_len_a_time": model_args.embedding_fixed_len_a_time,
            "matrix_embedding_exists": model_args.matrix_embedding_exists,
            "use_cpu": True if model_args.gpu_id < 0 else False
        }
    else:
        assert model_args.seq_max_length is not None
        encoder_config = {
            "llm_dirpath": model_args.llm_dirpath,
            "llm_type": model_args.llm_type,
            "llm_version": model_args.llm_version,
            "llm_step": model_args.llm_step,
            "input_type": model_args.input_type,
            "trunc_type": model_args.trunc_type,
            "seq_max_length": model_args.seq_max_length,
            "atom_seq_max_length": None,
            "prepend_bos": True,
            "append_eos": True,
            "vector_dirpath": model_args.vector_dirpath,
            "matrix_dirpath": model_args.matrix_dirpath,
            "local_rank": model_args.gpu_id,
            "max_sentence_length": model_args.max_sentence_length if hasattr(model_args, "max_sentence_length") else None,
            "max_sentences": model_args.max_sentences if hasattr(model_args, "max_sentences") else None,
            "embedding_complete": model_args.embedding_complete,
            "embedding_complete_seg_overlap": model_args.embedding_complete_seg_overlap,
            "matrix_add_special_token": model_args.matrix_add_special_token,
            "embedding_fixed_len_a_time": model_args.embedding_fixed_len_a_time,
            "matrix_embedding_exists": model_args.matrix_embedding_exists,
            "use_cpu": True if model_args.gpu_id < 0 else False
        }
    encoder = Encoder(**encoder_config)

    batch_converter = BatchConverter(
        input_type=model_args.input_type if hasattr(model_args, "input_type") else False,
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


# global
global_model_config, global_seq_subword, global_seq_tokenizer, global_trained_model = None, None, None, None


def run(
        sequences,
        llm_truncation_seq_length,
        model_path,
        dataset_name,
        dataset_type,
        task_type,
        task_level_type,
        model_type,
        input_type,
        input_mode,
        time_str,
        step,
        gpu_id,
        threshold,
        topk,
        emb_dir,
        matrix_embedding_exists,
        output_attention_scores_dirpath,
        output_attention_pooling_scores_dirpath,
        output_classification_vector_dirpath
):
    global global_model_config, global_seq_subword, global_seq_tokenizer, global_trained_model
    model_dir = "%s/models/%s/%s/%s/%s/%s/%s/%s" % (
        model_path, dataset_name, dataset_type, task_type, model_type, input_type,
        time_str, step if step == "best" else "checkpoint-{}".format(step)
    )
    config_dir = "%s/logs/%s/%s/%s/%s/%s/%s" % (
        model_path, dataset_name, dataset_type, task_type, model_type, input_type, time_str
    )

    model_args = torch.load(os.path.join(model_dir, "training_args.bin"))
    print("-" * 25 + "Trained Model Args" + "-" * 25)
    print(model_args.__dict__)
    print("-" * 50)

    # download LLM(LucaOne)
    if not hasattr(model_args, "llm_type"):
        model_args.llm_type = "lucaone"
    if not hasattr(model_args, "llm_version"):
        model_args.llm_version = "lucaone"
    if not hasattr(model_args, "llm_step"):
        model_args.llm_step = "5600000"
    download_trained_checkpoint_lucaone(
        llm_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "llm/"),
        llm_type=model_args.llm_type,
        llm_version=model_args.llm_version,
        llm_step=model_args.llm_step
    )
    # download trained downstream task models
    '''
    download_trained_checkpoint_downstream_tasks(
        save_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    '''

    '''
    model_args.llm_truncation_seq_length = llm_truncation_seq_length
    model_args.seq_max_length = llm_truncation_seq_length
    model_args.atom_seq_max_length = None # to do
    model_args.truncation_seq_length = model_args.seq_max_length
    model_args.truncation_matrix_length = model_args.matrix_max_length
    '''
    model_args.output_attention_scores_dirpath = output_attention_scores_dirpath
    model_args.output_attention_pooling_scores_dirpath = output_attention_pooling_scores_dirpath
    model_args.output_classification_vector_dirpath = output_classification_vector_dirpath
    model_args.llm_truncation_seq_length = llm_truncation_seq_length
    model_args.seq_max_length = max(model_args.seq_max_length, llm_truncation_seq_length)
    model_args.atom_seq_max_length = None # to do
    model_args.truncation_seq_length = model_args.seq_max_length
    model_args.truncation_matrix_length = max(model_args.matrix_max_length, llm_truncation_seq_length)

    model_args.matrix_embedding_exists = matrix_embedding_exists
    # embedding saved dir during prediction
    if emb_dir and not matrix_embedding_exists:
        # now = datetime.now()
        # formatted_time = now.strftime("%Y%m%d%H%M%S")
        # emb_dir = os.path.join(emb_dir, "%s-%d" % (formatted_time, gpu_id))
        unique_string = str(uuid.uuid4())
        emb_dir = os.path.join(emb_dir, "%s-%d" % (unique_string, gpu_id))
    model_args.emb_dir = emb_dir
    model_args.vector_dirpath = model_args.emb_dir if model_args.emb_dir else None
    model_args.matrix_dirpath = model_args.emb_dir if model_args.emb_dir else None

    model_args.dataset_name = dataset_name
    model_args.dataset_type = dataset_type
    model_args.task_type = task_type
    model_args.model_type = model_type
    model_args.input_type = input_type
    model_args.time_str = time_str
    model_args.step = step
    model_args.task_level_type = task_level_type
    model_args.gpu_id = gpu_id

    if not hasattr(model_args, "embedding_complete"):
        model_args.embedding_complete = False
    if not hasattr(model_args, "embedding_complete_seg_overlap"):
        model_args.embedding_complete_seg_overlap = False
    if not hasattr(model_args, "embedding_fixed_len_a_time"):
        model_args.embedding_fixed_len_a_time = None
    if not hasattr(model_args, "matrix_add_special_token"):
        model_args.matrix_add_special_token = False

    if not hasattr(model_args, "non_ignore"):
        model_args.non_ignore = True
    model_args.threshold = threshold

    if model_args.label_filepath:
        model_args.label_filepath = model_args.label_filepath.replace("../", "%s/" % model_path)
    if not os.path.exists(model_args.label_filepath):
        model_args.label_filepath = os.path.join(config_dir, "label.txt")

    if gpu_id is None or gpu_id < 0:
        # gpu_id = available_gpu_id()
        gpu_id = -1
        model_args.gpu_id = gpu_id
    print("------Before loading the model:------")
    print("GPU ID: %d" % gpu_id)
    model_args.device = torch.device("cuda:%d" % gpu_id if gpu_id > -1 else "cpu")
    device_memory(None if gpu_id == -1 else gpu_id)

    # Step2: loading the tokenizer and model
    if global_trained_model is None or next(global_trained_model.parameters()).device != model_args.device:
        global_trained_model = None
        model_config, seq_subword, seq_tokenizer, trained_model = load_model(model_args, model_type, model_dir)
        global_model_config = model_config
        global_seq_subword = seq_subword
        global_seq_tokenizer = seq_tokenizer
        global_trained_model = trained_model
    else:
        model_config = global_model_config
        seq_subword = global_seq_subword
        seq_tokenizer = global_seq_tokenizer
        trained_model = global_trained_model

    print("------After loaded the model:------")
    
    device_memory(None if gpu_id == -1 else gpu_id)
    encoder, batch_convecter = create_encoder_batch_convecter(model_args, seq_subword, seq_tokenizer)

    # embedding in advance
    print("matrix_embedding_exists: %r, gpu_id: %d, input_type: %s" % (matrix_embedding_exists, gpu_id, input_type))
    if not matrix_embedding_exists and gpu_id > -1 and input_type != "seq":
        # 先to cpu
        trained_model.to(torch.device("cpu"))
        assert model_args.emb_dir is not None
        if not os.path.exists(model_args.emb_dir):
            os.makedirs(model_args.emb_dir)
        for item in sequences:
            if input_mode == "pair":
                seq_id_a = item[0]
                seq_id_b = item[1]
                seq_type_a = item[2]
                seq_type_b = item[3]
                seq_a = item[4]
                seq_b = item[5]
                encoder.__get_embedding__(
                    seq_id=seq_id_a,
                    seq_type=seq_type_a,
                    seq=seq_a,
                    embedding_type="matrix" if "matrix" in input_type else "vector"
                )
                encoder.__get_embedding__(
                    seq_id=seq_id_b,
                    seq_type=seq_type_b,
                    seq=seq_b,
                    embedding_type="matrix" if "matrix" in input_type else "vector"
                )
            else:
                seq_id = item[0]
                seq_type = item[1]
                seq = item[2]
                encoder.__get_embedding__(
                    seq_id=seq_id,
                    seq_type=seq_type,
                    seq=seq,
                    embedding_type="matrix" if "matrix" in input_type else "vector"
                )
            torch.cuda.empty_cache()
        encoder.matrix_embedding_exists = True
        # embedding 完之后to device
        trained_model.to(model_args.device)

    label_list = load_labels(model_args.label_filepath)
    label_id_2_name = {idx: name for idx, name in enumerate(label_list)}

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
        raise Exception("the task_type=%s or task_level_type=%s error" % (task_type, model_args.task_level_type))

    predicted_results = []
    print()
    print("Device:", model_args.device)
    if hasattr(model_args, "input_mode") and model_args.input_mode in ["pair"]:
        for item in sequences:
            seq_id_a = item[0]
            seq_id_b = item[1]
            seq_type_a = item[2]
            seq_type_b = item[3]
            seq_a = item[4]
            seq_b = item[5]
            row = [seq_id_a, seq_id_b, seq_type_a, seq_type_b, seq_a, seq_b]
            if "_vs_" in model_args.input_type:
                vector_filename_a = item[6]
                vector_filename_b = item[7]
                matrix_filename_a = item[8]
                matrix_filename_b = item[9]
                row = row + [vector_filename_a, vector_filename_b, matrix_filename_a, matrix_filename_b]
                if model_args.input_type == "matrix_express_vs_matrix":
                    express_list_a = item[10]
                    row = row + [express_list_a]
                elif model_args.input_type == "matrix_express_vs_matrix_express":
                    express_list_a = item[10]
                    express_list_b = item[11]
                    row = row + [express_list_a, express_list_b]
            if task_level_type in ["seq_level", "seq-level"] and task_type in ["multi_class", "multi-class"]:
                cur_res = predict_func(
                    model_args,
                    encoder,
                    batch_convecter,
                    label_id_2_name,
                    trained_model,
                    row,
                    topk=topk
                )
                if topk is not None and topk > 1:
                    predicted_results.append([
                        cur_res[0][0],
                        cur_res[0][1],
                        cur_res[0][2],
                        cur_res[0][3],
                        cur_res[0][4],
                        cur_res[0][5],
                        cur_res[0][6],
                        cur_res[0][7]
                    ])
                else:
                    predicted_results.append([
                        cur_res[0][0],
                        cur_res[0][1],
                        cur_res[0][2],
                        cur_res[0][3],
                        cur_res[0][4],
                        cur_res[0][5]
                    ])
            else:
                cur_res = predict_func(
                    model_args,
                    encoder,
                    batch_convecter,
                    label_id_2_name,
                    trained_model,
                    row
                )
                predicted_results.append([
                    cur_res[0][0],
                    cur_res[0][1],
                    cur_res[0][2],
                    cur_res[0][3],
                    cur_res[0][4],
                    cur_res[0][5]
                ])
    else:
        for item in sequences:
            seq_id = item[0]
            seq_type = item[1]
            seq = item[2]
            if len(item) > 3:
                row = [seq_id, seq_type, seq, *item[3:]]
            else:
                row = [seq_id, seq_type, seq]
            if task_level_type in ["seq_level", "seq-level"] and task_type in ["multi_class", "multi-class"]:
                # print("task_level_type: %s, task_type: %s" % (task_level_type, task_type))
                cur_res = predict_func(
                    model_args,
                    encoder,
                    batch_convecter,
                    label_id_2_name,
                    trained_model,
                    row,
                    topk=topk
                )
                if topk is not None and topk > 1:
                    predicted_results.append([
                        seq_id, seq, cur_res[0][2], cur_res[0][3], cur_res[0][4], cur_res[0][5]
                    ])
                else:
                    predicted_results.append([
                        seq_id, seq, cur_res[0][2], cur_res[0][3]
                    ])
            else:
                cur_res = predict_func(
                    model_args,
                    encoder,
                    batch_convecter,
                    label_id_2_name,
                    trained_model,
                    row
                )
                predicted_results.append([
                    seq_id, seq, cur_res[0][2], cur_res[0][3]
                ])
    # torch.cuda.empty_cache()
    # 删除embedding
    if not matrix_embedding_exists and os.path.exists(model_args.emb_dir) and input_type != "seq":
        shutil.rmtree(model_args.emb_dir)
    return predicted_results


def run_args():
    parser = argparse.ArgumentParser(description="Prediction")
    # for one seq sample of the input
    parser.add_argument("--seq_id", default=None, type=str,  help="the seq id")
    parser.add_argument("--seq_type", default=None, type=str, choices=["prot", "gene"], help="seq type.")
    parser.add_argument("--seq", default=None, type=str,  help="the sequence")

    # for one seq-seq sample of the input
    parser.add_argument("--seq_id_a", default=None, type=str,  help="the seq id a")
    parser.add_argument("--seq_type_a", default=None, type=str, choices=["prot", "gene"], help="seq type a.")
    parser.add_argument("--seq_a", default=None, type=str,  help="the sequence a")
    parser.add_argument("--seq_id_b", default=None, type=str,  help="the seq id b")
    parser.add_argument("--seq_type_b", default=None, type=str, choices=["prot", "gene"], help="seq type b.")
    parser.add_argument("--seq_b", default=None, type=str,  help="the sequence b")

    # for many samples
    parser.add_argument(
        "--input_file",
        default=None,
        type=str,
        help="the fasta or csv format file for single-seq model,or the csv format file for pair-seq model"
    )
    # for csv
    parser.add_argument(
        "--seq_id_col_idx",
        default=None,
        type=int,
        help="the seq_id column index for csv/tsv file"
    )
    parser.add_argument(
        "--seq_type_col_idx",
        default=None,
        type=int,
        help="the seq_type idx column index for csv/tsv file"
    )
    parser.add_argument(
        "--seq_col_idx",
        default=None,
        type=int,
        help="the seq idx column index for csv/tsv file"
    )
    parser.add_argument(
        "--vector_col_idx",
        default=None,
        type=int,
        help="the vector_filename column index for csv/tsv file"
    )
    parser.add_argument(
        "--matrix_col_idx",
        default=None,
        type=int,
        help="the matrix_filename column index for csv/tsv file"
    )
    parser.add_argument(
        "--express_col_idx",
        default=None,
        type=int,
        help="the matrix_filename column index for csv/tsv file"
    )

    parser.add_argument(
        "--seq_id_col_idx_a",
        default=None,
        type=int,
        help="the seq_id_a column index for csv/tsv file"
    )
    parser.add_argument(
        "--seq_type_col_idx_a",
        default=None,
        type=int,
        help="the seq_type_a idx column index for csv/tsv file"
    )
    parser.add_argument(
        "--seq_col_idx_a",
        default=None,
        type=int,
        help="the seq_a column index for csv/tsv file"
    )
    parser.add_argument(
        "--vector_col_idx_a",
        default=None,
        type=int,
        help="the vector_filename_a column index for csv/tsv file"
    )
    parser.add_argument(
        "--matrix_col_idx_a",
        default=None,
        type=int,
        help="the matrix_filename_a column index for csv/tsv file"
    )
    parser.add_argument(
        "--express_col_idx_a",
        default=None,
        type=int,
        help="the express_col_idx_a column index for csv/tsv file"
    )
    parser.add_argument(
        "--seq_id_col_idx_b",
        default=None,
        type=int,
        help="the seq_id_b column index for csv/tsv file"
    )
    parser.add_argument(
        "--seq_type_col_idx_b",
        default=None,
        type=int,
        help="the seq_type_b idx column index for csv/tsv file"
    )
    parser.add_argument(
        "--seq_col_idx_b",
        default=None,
        type=int,
        help="the seq_b column index for csv/tsv file"
    )
    parser.add_argument(
        "--vector_col_idx_b",
        default=None,
        type=int,
        help="the vector_filename_b column index for csv/tsv file"
    )
    parser.add_argument(
        "--matrix_col_idx_b",
        default=None,
        type=int,
        help="the matrix_filename_b column index for csv/tsv file"
    )
    parser.add_argument(
        "--express_col_idx_b",
        default=None,
        type=int,
        help="the express_col_idx_b column index for csv/tsv file"
    )

    # for embedding
    parser.add_argument(
        "--llm_truncation_seq_length",
        default=4096,
        type=int,
        required=True,
        help="the max seq-length for llm embedding"
    )
    parser.add_argument(
        "--matrix_embedding_exists",
        action="store_true",
        help="the structural embedding is or not in advance. default: False"
    )
    parser.add_argument(
        "--emb_dir",
        default=None,
        type=str,
        help="the structural embedding save dir. default: None"
    )

    # for trained model
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        help="the model dir. default: None"
    )
    parser.add_argument(
        "--dataset_name",
        default=None,
        type=str,
        required=True,
        help="the dataset name for model building."
    )
    parser.add_argument(
        "--dataset_type",
        default=None,
        type=str,
        required=True,
        help="the dataset type for model building."
    )
    parser.add_argument(
        "--task_type",
        default=None,
        type=str,
        required=True,
        choices=["multi_label", "multi_class", "binary_class", "regression"],
        help="the task type for model building."
    )
    parser.add_argument(
        "--task_level_type",
        default=None,
        type=str,
        required=True,
        choices=["seq_level", "token_level"],
        help="the task level type for model building."
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        choices=["luca_base", "lucappi", "lucappi2", "lucapair_homo", "lucapair_heter", "lucapair_intrainter"],
        help="the model type."
    )
    parser.add_argument(
        "--input_type", default=None, type=str, required=True,
        choices=[
            "seq",
            "matrix",
            "vector",
            "seq_matrix",
            "seq_vector",
            "matrix_express",
            "seq_vs_seq",
            "seq_vs_vector",
            "seq_vs_matrix",
            "vector_vs_vector",
            "vector_vs_matrix",
            "matrix_vs_matrix",
            "matrix_express_vs_matrix",
            "matrix_express_vs_matrix_express"
        ],
        help="the input type."
    )
    parser.add_argument(
        "--input_mode",
        default=None,
        type=str,
        required=True,
        choices=["single", "pair"],
        help="the input mode."
    )
    parser.add_argument(
        "--time_str",
        default=None,
        type=str,
        required=True,
        help="the running time string(yyyymmddHimiss) of model building."
    )
    parser.add_argument(
        "--step",
        default=None,
        type=str,
        required=True,
        help="the training global checkpoint step of model finalization."
    )

    parser.add_argument(
        "--topk",
        default=None,
        type=int, help="the topk labels for multi-class"
    )
    parser.add_argument(
        "--threshold",
        default=0.5,
        type=float,
        help="sigmoid threshold for binary-class or multi-label classification, None for multi-class classification or regression, default: 0.5."
    )
    parser.add_argument(
        "--ground_truth_idx",
        default=None,
        type=int,
        help="the ground truth idx, when the input file contains"
    )

    # for results(csv format, contain header)
    parser.add_argument(
        "--save_path",
        default=None,
        type=str,
        help="the result save path"
    )
    # for print info
    parser.add_argument(
        "--print_per_num",
        default=10000,
        type=int,
        help="per num to print"
    )
    parser.add_argument(
        "--output_attention_scores_dirpath",
        default=None,
        type=str,
        help="the save path output the attention scores(one file for each one sample)"
    )
    parser.add_argument(
        "--output_attention_pooling_scores_dirpath",
        default=None,
        type=str,
        help="the save path output the attention pooling scores(one file for each one sample)"
    )
    parser.add_argument(
        "--output_classification_vector_dirpath",
        default=None,
        type=str,
        help="the save path output the attention pooling scores(one file for each one sample)"
    )
    parser.add_argument(
        "--gpu_id",
        default=None,
        type=int,
        help="the used gpu index, -1 for cpu"
    )
    input_args = parser.parse_args()
    return input_args


if __name__ == "__main__":
    args = run_args()
    print("-" * 25 + "Run Args" + "-" * 25)
    print(args.__dict__)
    print("-" * 50)
    if args.output_attention_scores_dirpath:
        args.output_attention_scores = True
        if not os.path.exists(args.output_attention_scores_dirpath):
            os.makedirs(args.output_attention_scores_dirpath)
    if args.output_attention_pooling_scores_dirpath:
        args.output_attention_pooling_scores = True
        if not os.path.exists(args.output_attention_pooling_scores_dirpath):
            os.makedirs(args.output_attention_pooling_scores_dirpath)
    if args.output_classification_vector_dirpath:
        args.output_classification_vectors = True
        if not os.path.exists(args.output_classification_vector_dirpath):
            os.makedirs(args.output_classification_vector_dirpath)
    if args.input_file is not None:
        input_file_suffix = os.path.basename(args.input_file).split(".")[-1]
        if args.input_mode == "pair":
            if input_file_suffix not in ["csv", "tsv"]:
                print("Error! the input file is not in .csv or .tsv format for the pair seqs task.")
                sys.exit(-1)
        else:
            if input_file_suffix in ["fasta", "faa", "fas", "fa"] and args.seq_type is None:
                print("Error! input a fasta file, please set arg: --seq_type, value: gene or prot")
                sys.exit(-1)

    if args.input_file is not None and os.path.exists(args.input_file):
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
                if "_vs_" in args.input_type:
                    if args.input_type == "seq_vs_seq":
                        if args.task_type == "multi_class" and args.topk is not None and args.topk > 1 and args.task_level_type == "seq_level":
                            header = ["seq_id_a", "seq_id_b", "seq_a", "seq_b", "top1_prob", "top1_label", "top%d_probs" % args.topk, "top%d_labels" % args.topk]
                        else:
                            header = ["seq_id_a", "seq_id_b", "seq_a", "seq_b", "prob", "label"]
                    elif args.input_type == "seq_vs_vector":
                        if args.task_type == "multi_class" and args.topk is not None and args.topk > 1 and args.task_level_type == "seq_level":
                            header = ["seq_id_a", "seq_id_b", "seq_a", "vector_filename_b", "top1_prob", "top1_label", "top%d_probs" % args.topk, "top%d_labels" % args.topk]
                        else:
                            header = ["seq_id_a", "seq_id_b", "seq_a", "vector_filename_b", "prob", "label"]
                    elif args.input_type == "seq_vs_matrix":
                        if args.task_type == "multi_class" and args.topk is not None and args.topk > 1 and args.task_level_type == "seq_level":
                            header = ["seq_id_a", "seq_id_b", "seq_a", "matrix_filename_b", "top1_prob", "top1_label", "top%d_probs" % args.topk, "top%d_labels" % args.topk]
                        else:
                            header = ["seq_id_a", "seq_id_b", "seq_a", "matrix_filename_b", "prob", "label"]
                    elif args.input_type == "vector_vs_vector":
                        if args.task_type == "multi_class" and args.topk is not None and args.topk > 1 and args.task_level_type == "seq_level":
                            header = ["seq_id_a", "seq_id_b", "vector_filename_a", "vector_filename_b", "top1_prob", "top1_label", "top%d_probs" % args.topk, "top%d_labels" % args.topk]
                        else:
                            header = ["seq_id_a", "seq_id_b", "vector_filename_a", "vector_filename_b", "prob", "label"]
                    elif args.input_type == "vector_vs_matrix":
                        if args.task_type == "multi_class" and args.topk is not None and args.topk > 1 and args.task_level_type == "seq_level":
                            header = ["seq_id_a", "seq_id_b", "vector_filename_a", "matrix_filename_b", "top1_prob", "top1_label", "top%d_probs" % args.topk, "top%d_labels" % args.topk]
                        else:
                            header = ["seq_id_a", "seq_id_b", "vector_filename_a", "matrix_filename_b", "prob", "label"]
                    elif args.input_type == "matrix_vs_matrix":
                        if args.task_type == "multi_class" and args.topk is not None and args.topk > 1 and args.task_level_type == "seq_level":
                            header = ["seq_id_a", "seq_id_b", "matrix_filename_a", "matrix_filename_b", "top1_prob", "top1_label", "top%d_probs" % args.topk, "top%d_labels" % args.topk]
                        else:
                            header = ["seq_id_a", "seq_id_b", "matrix_filename_a", "matrix_filename_b", "prob", "label"]
                    elif args.input_type == "matrix_express_vs_matrix":
                        if args.task_type == "multi_class" and args.topk is not None and args.topk > 1 and args.task_level_type == "seq_level":
                            header = ["seq_id_a", "seq_id_b", "matrix_filename_a", "matrix_filename_b", "top1_prob", "top1_label", "top%d_probs" % args.topk, "top%d_labels" % args.topk]
                        else:
                            header = ["seq_id_a", "seq_id_b", "matrix_filename_a", "matrix_filename_b", "prob", "label"]
                    elif args.input_type == "matrix_express_vs_matrix_express":
                        if args.task_type == "multi_class" and args.topk is not None and args.topk > 1 and args.task_level_type == "seq_level":
                            header = ["seq_id_a", "seq_id_b", "matrix_filename_a", "matrix_filename_b", "top1_prob", "top1_label", "top%d_probs" % args.topk, "top%d_labels" % args.topk]
                        else:
                            header = ["seq_id_a", "seq_id_b", "matrix_filename_a", "matrix_filename_b", "prob", "label"]
                    else:
                        raise Exception("Not support input_mode=%s, input_type=%s" % (args.input_mode, args.input_type))
                else:
                    if args.task_type == "multi_class" and args.topk is not None and args.topk > 1 and args.task_level_type == "seq_level":
                        header = [
                            "seq_id_a",
                            "seq_id_b",
                            "seq_a",
                            "seq_b",
                            "top1_prob",
                            "top1_label",
                            "top%d_probs" % args.topk,
                            "top%d_labels" % args.topk
                        ]
                    else:
                        header = [
                            "seq_id_a",
                            "seq_id_b",
                            "seq_a",
                            "seq_b",
                            "prob",
                            "label"
                        ]
            else:
                if args.task_type == "multi_class" and args.topk is not None and args.topk > 1 and args.task_level_type == "seq_level":
                    header = [
                        "seq_id",
                        "seq",
                        "top1_prob",
                        "top1_label",
                        "top%d_probs" % args.topk,
                        "top%d_labels" % args.topk
                    ]
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

            reader = file_reader(args.input_file) if args.input_file.endswith(".csv") or args.input_file.endswith(".tsv") else fasta_reader(args.input_file)
            for row in reader:
                if args.input_mode == "pair":
                    if args.seq_id_col_idx_a is None:
                        args.seq_id_col_idx_a = 0
                    if args.seq_id_col_idx_b is None:
                        args.seq_id_col_idx_b = 1
                    if args.seq_type_col_idx_a is None:
                        args.seq_type_col_idx_a = 2
                    if args.seq_type_col_idx_b is None:
                        args.seq_type_col_idx_b = 3
                    if args.seq_col_idx_a is None:
                        args.seq_col_idx_a = 4
                    if args.seq_col_idx_b is None:
                        args.seq_col_idx_b = 5
                    if args.vector_col_idx_a is None:
                        args.vector_col_idx_a = 6
                    if args.vector_col_idx_b is None:
                        args.vector_col_idx_b = 7
                    if args.matrix_col_idx_a is None:
                        args.matrix_col_idx_a = 8
                    if args.matrix_col_idx_b is None:
                        args.matrix_col_idx_b = 9
                    if args.express_col_idx_a is None:
                        args.express_col_idx_a = 10
                    if args.express_col_idx_b is None:
                        args.express_col_idx_b = 11
                    if row[args.seq_id_col_idx_a] + "_" + row[args.seq_id_col_idx_b] in exists_ids:
                        continue
                    # seq_id_a, seq_id_b, seq_type_a, seq_type_b, seq_a, seq_b
                    if args.input_type in ["seq", "seq_vector", "seq_matrix", "seq_vs_seq", "seq_vs_vector", "seq_vs_matrix"] \
                            and not seq_type_is_match_seq(
                            row[args.seq_type_col_idx_a], row[args.seq_col_idx_a]
                    ):
                        print("Error! the input seq_a(seq_id_a=%s) not match the seq_type_a=%s: %s" % (
                            row[args.seq_id_col_idx_a],
                            row[args.seq_type_col_idx_a],
                            row[args.seq_col_idx_a]
                        ))
                        sys.exit(-1)
                    if args.input_type in ["seq", "seq_vector", "seq_matrix", "seq_vs_seq", "seq_vs_vector", "seq_vs_matrix"] \
                            and not seq_type_is_match_seq(
                            row[args.seq_type_col_idx_b], row[args.seq_col_idx_b]
                    ):
                        print("Error! the input seq_b(seq_id_b=%s) not match the seq_type_b=%s: %s" % (
                            row[args.seq_id_col_idx_b],
                            row[args.seq_type_col_idx_b],
                            row[args.seq_col_idx_b]
                        ))
                        sys.exit(-1)
                    if "_vs_" in args.input_type:
                        if args.input_type in ["matrix_express_vs_matrix", "matrix_express_vs_matrix_express"]:
                            batch_data.append([
                                row[args.seq_id_col_idx_a],
                                row[args.seq_id_col_idx_b],
                                row[args.seq_type_col_idx_a],
                                row[args.seq_type_col_idx_b],
                                row[args.seq_col_idx_a],
                                row[args.seq_col_idx_b],
                                row[args.vector_col_idx_a],
                                row[args.vector_col_idx_b],
                                row[args.matrix_col_idx_a],
                                row[args.matrix_col_idx_b],
                                row[args.express_col_idx_a],
                                row[args.express_col_idx_b],
                            ])
                        else:
                            batch_data.append([
                                row[args.seq_id_col_idx_a],
                                row[args.seq_id_col_idx_b],
                                row[args.seq_type_col_idx_a],
                                row[args.seq_type_col_idx_b],
                                row[args.seq_col_idx_a],
                                row[args.seq_col_idx_b],
                                row[args.vector_col_idx_a],
                                row[args.vector_col_idx_b],
                                row[args.matrix_col_idx_a],
                                row[args.matrix_col_idx_b]
                            ])
                    else:
                        batch_data.append([
                            row[args.seq_id_col_idx_a],
                            row[args.seq_id_col_idx_b],
                            row[args.seq_type_col_idx_a],
                            row[args.seq_type_col_idx_b],
                            row[args.seq_col_idx_a],
                            row[args.seq_col_idx_b]
                        ])
                    if args.ground_truth_idx is not None and args.ground_truth_idx >= 0:
                        batch_ground_truth.append(row[args.ground_truth_idx])
                else:
                    if args.seq_id_col_idx is None:
                        args.seq_id_col_idx = 0
                    if args.seq_type_col_idx is None:
                        args.seq_type_col_idx = 1
                    if args.seq_col_idx is None:
                        args.seq_col_idx = 2
                    if args.vector_col_idx is None:
                        args.vector_col_idx = 3
                    if args.matrix_col_idx is None:
                        args.matrix_col_idx = 4
                    if args.express_col_idx is None:
                        args.express_col_idx = 5
                    if row[args.seq_id_col_idx] in exists_ids:
                        continue
                    if len(row) == 2:
                        args.seq_type_col_idx = None
                        args.seq_col_idx = 1
                        if not seq_type_is_match_seq(args.seq_type, row[args.seq_col_idx]):
                            print("Error! the input seq(seq_id=%s) not match the arg: --seq_type=%s: %s" % (
                                row[args.seq_id_col_idx],
                                args.seq_type,
                                row[args.seq_col_idx]
                            ))
                            sys.exit(-1)
                        batch_data.append([row[args.seq_id_col_idx], args.seq_type, row[args.seq_col_idx ]])
                    elif len(row) > 2:
                        if not seq_type_is_match_seq(row[args.seq_type_col_idx], row[args.seq_col_idx]):
                            print("Error! the input seq(seq_id=%s) not match the seq_type=%s: %s" % (
                                row[args.seq_id_col_idx],
                                row[args.seq_type_col_idx],
                                row[args.seq_col_idx]
                            ))
                            sys.exit(-1)
                        if args.ground_truth_idx is not None and args.ground_truth_idx >= 0:
                            batch_ground_truth.append(row[args.ground_truth_idx])
                        # seq_id, seq_type, seq, vector, matrix, express
                        if len(row) == 3:
                            batch_data.append([
                                row[args.seq_id_col_idx],
                                row[args.seq_type_col_idx],
                                row[args.seq_col_idx]
                            ])
                        else:
                            batch_data.append([
                                row[args.seq_id_col_idx],
                                row[args.seq_type_col_idx],
                                row[args.seq_col_idx],
                                row[args.vector_col_idx],
                                row[args.matrix_col_idx],
                                row[args.express_col_idx],
                            ])
                    else:
                        continue
                if len(batch_data) % args.print_per_num == 0:
                    batch_results = run(
                        batch_data,
                        args.llm_truncation_seq_length,
                        args.model_path,
                        args.dataset_name,
                        args.dataset_type,
                        args.task_type,
                        args.task_level_type,
                        args.model_type,
                        args.input_type,
                        args.input_mode,
                        args.time_str,
                        args.step,
                        args.gpu_id,
                        args.threshold,
                        topk=args.topk,
                        emb_dir=args.emb_dir,
                        matrix_embedding_exists=args.matrix_embedding_exists,
                        output_attention_scores_dirpath=args.output_attention_scores_dirpath,
                        output_attention_pooling_scores_dirpath=args.output_attention_pooling_scores_dirpath,
                        output_classification_vector_dirpath=args.output_classification_vector_dirpath
                    )
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
                batch_results = run(
                    batch_data,
                    args.llm_truncation_seq_length,
                    args.model_path,
                    args.dataset_name,
                    args.dataset_type,
                    args.task_type,
                    args.task_level_type,
                    args.model_type,
                    args.input_type,
                    args.input_mode,
                    args.time_str,
                    args.step,
                    args.gpu_id,
                    args.threshold,
                    topk=args.topk,
                    emb_dir=args.emb_dir,
                    matrix_embedding_exists=args.matrix_embedding_exists,
                    output_attention_scores_dirpath=args.output_attention_scores_dirpath,
                    output_attention_pooling_scores_dirpath=args.output_attention_pooling_scores_dirpath,
                    output_classification_vector_dirpath=args.output_classification_vector_dirpath
                )
                for item_idx, item in enumerate(batch_results):
                    if args.ground_truth_idx is not None and args.ground_truth_idx >= 0:
                        item.append(batch_ground_truth[item_idx])
                    writer.writerow(item)
                wfp.flush()
                had_done += len(batch_data)
                batch_data = []
                batch_ground_truth = []
            print("over, had_done: %d" % had_done)
    elif args.seq_id is not None and args.seq is not None:
        if args.seq_type is None:
            print("Please set arg: --seq_type, value: gene or prot")
            sys.exit(-1)
        if not seq_type_is_match_seq(args.seq_type, args.seq):
            print("Error! the input seq(seq_id=%s) not match its seq_type=%s: %s" % (args.seq_id, args.seq_type, args.seq))
            sys.exit(-1)
        data = [[args.seq_id, args.seq_type, args.seq]]
        results = run(
            data,
            args.llm_truncation_seq_length,
            args.model_path,
            args.dataset_name,
            args.dataset_type,
            args.task_type,
            args.task_level_type,
            args.model_type,
            args.input_type,
            args.input_mode,
            args.time_str,
            args.step,
            args.gpu_id,
            args.threshold,
            topk=args.topk,
            emb_dir=args.emb_dir,
            matrix_embedding_exists=args.matrix_embedding_exists,
            output_attention_scores_dirpath=args.output_attention_scores_dirpath,
            output_attention_pooling_scores_dirpath=args.output_attention_pooling_scores_dirpath,
            output_classification_vector_dirpath=args.output_classification_vector_dirpath
        )
        print("Predicted Result:")
        print("seq_id=%s" % args.seq_id)
        print("seq=%s" % args.seq)
        print("prob=%f" % results[0][2])
        print("label=%s" % results[0][3])
    elif args.seq_id_a is not None and args.seq_a is not None and args.seq_id_b is not None and args.seq_b is not None:
        if args.seq_type_a is None:
            print("Please set arg: --seq_type_a, value: gene or prot")
            sys.exit(-1)
        if args.seq_type_b is None:
            print("Please set arg: --seq_type_b, value: gene or prot")
            sys.exit(-1)
        if not seq_type_is_match_seq(args.seq_type_a, args.seq_a):
            print("Error! the input seq_a(seq_id_a=%s) not match its seq_type_a=%s: %s" % (args.seq_id_a, args.seq_type_a, args.seq_a))
            sys.exit(-1)
        if not seq_type_is_match_seq(args.seq_type_b, args.seq_b):
            print("Error! the input seq_b(seq_id_b=%s) not match its seq_type_b=%s: %s" % (args.seq_id_b, args.seq_type_b, args.seq_b))
            sys.exit(-1)
        data = [[
            args.seq_id_a,
            args.seq_id_b,
            args.seq_type_a,
            args.seq_type_b,
            args.seq_a,
            args.seq_b
        ]]
        results = run(
            data,
            args.llm_truncation_seq_length,
            args.model_path,
            args.dataset_name,
            args.dataset_type,
            args.task_type,
            args.task_level_type,
            args.model_type,
            args.input_type,
            args.input_mode,
            args.time_str,
            args.step,
            args.gpu_id,
            args.threshold,
            topk=args.topk,
            emb_dir=args.emb_dir,
            matrix_embedding_exists=args.matrix_embedding_exists,
            output_attention_scores_dirpath=args.output_attention_scores_dirpath,
            output_attention_pooling_scores_dirpath=args.output_attention_pooling_scores_dirpath,
            output_classification_vector_dirpath=args.output_classification_vector_dirpath
        )
        print("Predicted Result:")
        print("seq_id_a=%s, seq_id_b=%s" % (args.seq_id_a, args.seq_id_b))
        print("seq_a=%s" % args.seq_a)
        print("seq_b=%s" % args.seq_b)
        print("prob=%f" % results[0][4])
        print("label=%s" % results[0][5])
    else:
        raise Exception("input error, usage: --hep")
