#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/4/21 16:10
@project: LucaOneTasks
@file: encoder
@desc: encoder for LucaOneTasks
'''
import os
import torch
import sys
import numpy as np
sys.path.append(".")
sys.path.append("../")
sys.path.append("../src")
try:
    from utils import clean_seq_esm, calc_emb_filename_by_seq_id
    from common.alphabet import Alphabet
    from llm.lucagplm.get_embedding import predict_embedding as predict_embedding_luca
    from llm.esm.predict_embedding import predict_embedding as predict_embedding_esm
    from llm.dnabert2.inference_embedding import predict_embedding as predict_embedding_dnabert2
    from llm.dnaberts.inference_embedding import predict_embedding as predict_embedding_dnaberts
except ImportError as e:
    from src.utils import clean_seq_esm, calc_emb_filename_by_seq_id
    from src.common.alphabet import Alphabet
    from src.llm.lucagplm.get_embedding import predict_embedding as predict_embedding_luca
    from src.llm.esm.predict_embedding import predict_embedding as predict_embedding_esm
    from src.llm.dnabert2.inference_embedding import predict_embedding as predict_embedding_dnabert2
    from src.llm.dnaberts.inference_embedding import predict_embedding as predict_embedding_dnaberts

MAX_SEQ_LEN = 10240


def complete_embedding_matrix(
        seq_id,
        seq_type,
        seq,
        truncation_seq_length,
        init_emb,
        llm_dirpath,
        trunc_type,
        embedding_type,
        matrix_add_special_token,
        embedding_complete,
        embedding_complete_seg_overlap,
        device,
        use_cpu=False
):
    if init_emb is not None and embedding_complete and ("representations" in embedding_type or "matrix" in embedding_type):
        torch.cuda.empty_cache()
        ori_seq_len = min(len(seq), MAX_SEQ_LEN)
        # 每次能处理这么长度
        # print("init_emb:", init_emb.shape)
        cur_segment_len = init_emb.shape[0]
        if matrix_add_special_token:
            first_emb = init_emb[1:cur_segment_len - 1]
        else:
            first_emb = init_emb
        if matrix_add_special_token:
            cur_segment_len = cur_segment_len - 2
        # print("cur_segment_len: %d" % cur_segment_len)
        init_cur_segment_len = cur_segment_len
        segment_num = int((ori_seq_len + cur_segment_len - 1) / cur_segment_len)
        if segment_num <= 1:
            return init_emb
        append_emb = None
        if embedding_complete_seg_overlap:
            sliding_window = init_cur_segment_len // 2
            print("Embedding Complete Seg Overlap: %r, ori seq len: %d, segment len: %d, init sliding windown: %d" % (
                embedding_complete_seg_overlap, ori_seq_len, init_cur_segment_len, sliding_window))
            while True:
                print("updated window: %d" % sliding_window)
                try:
                    # 第一个已经处理，滑动窗口
                    if trunc_type == "right":
                        last_end = init_cur_segment_len
                        seg_idx = 0
                        for pos_idx in range(init_cur_segment_len, ori_seq_len - sliding_window, sliding_window):
                            seg_idx += 1
                            last_end = min(pos_idx + sliding_window, ori_seq_len)
                            seg_seq = seq[pos_idx - sliding_window:last_end]
                            print("segment idx: %d, seg seq len: %d" % (seg_idx, len(seg_seq)))
                            seg_emb, seg_processed_seq_len = predict_embedding_luca(
                                llm_dirpath,
                                [seq_id + "_seg_%d" % seg_idx, seq_type, seg_seq],
                                trunc_type,
                                embedding_type,
                                repr_layers=[-1],
                                truncation_seq_length=truncation_seq_length,
                                matrix_add_special_token=False,
                                device=device if not use_cpu else torch.device("cpu")
                            )
                            # 有seq overlap 所以要截取
                            if append_emb is None:
                                append_emb = seg_emb[sliding_window:]
                            else:
                                append_emb = np.concatenate((append_emb, seg_emb[sliding_window:]), axis=0)
                        if last_end < ori_seq_len:
                            seg_idx += 1
                            remain = ori_seq_len - last_end
                            seg_seq = seq[ori_seq_len - 2 * sliding_window:ori_seq_len]
                            seg_emb, seg_processed_seq_len = predict_embedding_luca(
                                llm_dirpath,
                                [seq_id + "_seg_%d" % seg_idx, seq_type, seg_seq],
                                trunc_type,
                                embedding_type,
                                repr_layers=[-1],
                                truncation_seq_length=truncation_seq_length,
                                matrix_add_special_token=False,
                                device=device if not use_cpu else torch.device("cpu")
                            )
                            # 有seq overlap 所以要截取
                            if append_emb is None:
                                append_emb = seg_emb[-remain:]
                            else:
                                append_emb = np.concatenate((append_emb, seg_emb[-remain:]), axis=0)
                    else:
                        last_start = -init_cur_segment_len
                        seg_idx = 0
                        for pos_idx in range(-init_cur_segment_len, -ori_seq_len + sliding_window, -sliding_window):
                            seg_idx += 1
                            last_start = max(pos_idx - sliding_window, -ori_seq_len)
                            seg_seq = seq[last_start: pos_idx + sliding_window]
                            seg_emb, seg_processed_seq_len = predict_embedding_luca(
                                llm_dirpath,
                                [seq_id + "_seg_%d" % seg_idx, seq_type, seg_seq],
                                trunc_type,
                                embedding_type,
                                repr_layers=[-1],
                                truncation_seq_length=truncation_seq_length,
                                matrix_add_special_token=False,
                                device=device if not use_cpu else torch.device("cpu")
                            )
                            # 有seq overlap 所以要截取
                            if append_emb is None:
                                append_emb = seg_emb[:sliding_window]
                            else:
                                append_emb = np.concatenate((seg_emb[:sliding_window], append_emb), axis=0)
                        if last_start > -ori_seq_len:
                            seg_idx += 1
                            remain = last_start + ori_seq_len
                            seg_seq = seq[-ori_seq_len:-ori_seq_len + 2 * sliding_window]
                            seg_emb, seg_processed_seq_len = predict_embedding_luca(
                                llm_dirpath,
                                [seq_id + "_seg_%d" % seg_idx, seq_type, seg_seq],
                                trunc_type,
                                embedding_type,
                                repr_layers=[-1],
                                truncation_seq_length=truncation_seq_length,
                                matrix_add_special_token=False,
                                device=device if not use_cpu else torch.device("cpu")
                            )
                            # 有seq overlap 所以要截取
                            if append_emb is None:
                                append_emb = seg_emb[:remain]
                            else:
                                append_emb = np.concatenate((seg_emb[:remain], append_emb), axis=0)
                except Exception as e:
                    append_emb = None
                if append_emb is not None:
                    break
                print("fail, change sliding window: %d -> %d" % (sliding_window, int(sliding_window * 0.95)))
                sliding_window = int(sliding_window * 0.95)
        else:
            while True:
                print("ori seq len: %d, segment len: %d" % (ori_seq_len, cur_segment_len))
                try:
                    # 第一个已经处理，最后一个单独处理（需要向左/向右扩充至cur_segment_len长度）
                    if trunc_type == "right":
                        begin_seq_idx = 0
                    else:
                        begin_seq_idx = ori_seq_len - (segment_num - 1) * cur_segment_len
                    for seg_idx in range(1, segment_num - 1):
                        seg_seq = seq[begin_seq_idx + seg_idx * cur_segment_len: begin_seq_idx + (seg_idx + 1) * cur_segment_len]
                        # print("segment idx: %d, seg_seq(%d): %s" % (seg_idx, len(seg_seq), seg_seq))
                        print("segment idx: %d, seg seq len: %d" % (seg_idx, len(seg_seq)))
                        seg_emb, seg_processed_seq_len = predict_embedding_luca(
                            llm_dirpath,
                            [seq_id + "_seg_%d" % seg_idx, seq_type, seg_seq],
                            trunc_type,
                            embedding_type,
                            repr_layers=[-1],
                            truncation_seq_length=truncation_seq_length,
                            matrix_add_special_token=False,
                            device=device if not use_cpu else torch.device("cpu")
                        )

                        if append_emb is None:
                            append_emb = seg_emb
                        else:
                            '''
                            if trunc_type == "right":
                                append_emb = np.concatenate((append_emb, seg_emb), axis=0)
                            else:
                                append_emb = np.concatenate((seg_emb, append_emb), axis=0)
                            '''
                            append_emb = np.concatenate((append_emb, seg_emb), axis=0)
                    if trunc_type == "right":
                        # 处理最后一个
                        last_seg_seq = seq[-cur_segment_len:]
                        really_len = (ori_seq_len - (segment_num - 1) * cur_segment_len)
                        # print("last seg seq: %s" % last_seg_seq)
                        print("last seg seq len: %d, really len: %d" % (len(last_seg_seq), really_len))
                        last_seg_emb, last_seg_processed_seq_len = predict_embedding_luca(
                            llm_dirpath,
                            [seq_id + "_seg_%d" % (segment_num - 1), seq_type, last_seg_seq],
                            trunc_type,
                            embedding_type,
                            repr_layers=[-1],
                            truncation_seq_length=truncation_seq_length,
                            matrix_add_special_token=False,
                            device=device if not use_cpu else torch.device("cpu")
                        )
                        last_seg_emb = last_seg_emb[-really_len:, :]
                        append_emb = np.concatenate((append_emb, last_seg_emb), axis=0)
                    else:
                        # 处理第一个
                        first_seg_seq = seq[:cur_segment_len]
                        really_len = (ori_seq_len - (segment_num - 1) * cur_segment_len)
                        # print("first seg seq: %s" % first_seg_seq)
                        print("first seg seq len: %d, really len: %d" % (len(first_seg_seq), really_len))
                        first_seg_emb, first_seg_processed_seq_len = predict_embedding_luca(
                            llm_dirpath,
                            [seq_id + "_seg_0", seq_type, first_seg_seq],
                            trunc_type,
                            embedding_type,
                            repr_layers=[-1],
                            truncation_seq_length=truncation_seq_length,
                            matrix_add_special_token=False,
                            device=device if not use_cpu else torch.device("cpu")
                        )
                        first_seg_emb = first_seg_emb[:really_len, :]
                        append_emb = np.concatenate((first_seg_emb, append_emb), axis=0)
                except Exception as e:
                    append_emb = None
                if append_emb is not None:
                    break
                print("fail, change segment len: %d -> %d, change seg num: %d -> %d" % (
                    cur_segment_len,
                    int(cur_segment_len * 0.95),
                    segment_num,
                    int((ori_seq_len + cur_segment_len - 1) / cur_segment_len)
                ))
                cur_segment_len = int(cur_segment_len * 0.95)
                segment_num = int((ori_seq_len + cur_segment_len - 1) / cur_segment_len)

            append_emb = append_emb[init_cur_segment_len - cur_segment_len:]
        if trunc_type == "right":
            complete_emb = np.concatenate((first_emb, append_emb), axis=0)
        else:
            complete_emb = np.concatenate((append_emb, first_emb), axis=0)
        print("seq len: %d, seq embedding matrix len: %d" % (ori_seq_len, complete_emb.shape[0] + (2 if matrix_add_special_token else 0)))
        print("-" * 50)
        assert complete_emb.shape[0] == ori_seq_len
        if matrix_add_special_token:
            complete_emb = np.concatenate((init_emb[0:1, :], complete_emb, init_emb[-1:, :]), axis=0)
        init_emb = complete_emb
    return init_emb


class Encoder(object):
    def __init__(
            self,
            llm_dirpath,
            llm_type,
            llm_version,
            input_type,
            trunc_type,
            seq_max_length,
            atom_seq_max_length=None,
            prepend_bos=True,
            append_eos=True,
            vector_dirpath=None,
            matrix_dirpath=None,
            local_rank=-1,
            use_cpu=False,
            embedding_fixed_len_a_time=None,
            **kwargs
    ):
        print("-" * 25 + "Encoder" + "-" * 25)
        self.llm_dirpath = llm_dirpath
        self.llm_type = llm_type
        self.llm_version = llm_version
        self.input_type = input_type
        self.trunc_type = trunc_type
        self.seq_max_length = seq_max_length
        self.atom_seq_max_length = atom_seq_max_length
        self.embedding_fixed_len_a_time = embedding_fixed_len_a_time

        # vector
        if vector_dirpath and "#" in vector_dirpath:
            self.vector_dirpath = list(vector_dirpath.split("#"))
        elif vector_dirpath:
            self.vector_dirpath = [vector_dirpath]
        else:
            self.vector_dirpath = None
        # matrix
        if matrix_dirpath and "#" in matrix_dirpath:
            self.matrix_dirpath = list(matrix_dirpath.split("#"))
        elif matrix_dirpath:
            self.matrix_dirpath = [matrix_dirpath]
        else:
            self.matrix_dirpath = None
        # special tokens
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        self.alphabet = Alphabet.from_predefined("gene_prot")

        self.matrix_add_special_token = False
        if "matrix_add_special_token" in kwargs and kwargs["matrix_add_special_token"]:
            self.matrix_add_special_token = kwargs["matrix_add_special_token"]

        print("Encoder: prepend_bos=%r, append_eos=%r" % (self.prepend_bos, self.append_eos))
        if self.matrix_add_special_token:
            self.prepend_bos = True
            self.append_eos = True
        if "max_sentence_length" in kwargs and kwargs["max_sentence_length"]:
            self.max_sentence_length = kwargs["max_sentence_length"] - int(self.prepend_bos) - int(self.append_eos)
            print("Encoder: max_sentence_length=%d" % self.max_sentence_length)
        if "max_sentences" in kwargs and kwargs["max_sentences"]:
            self.max_sentences = kwargs["max_sentences"]
            print("Encoder: max_sentences=%d" % self.max_sentences)
        if "embedding_complete" in kwargs and kwargs["embedding_complete"]:
            self.embedding_complete = kwargs["embedding_complete"]
            print("Encoder: embedding_complete=%r" % self.embedding_complete)
        else:
            self.embedding_complete = False
        if "embedding_complete_seg_overlap" in kwargs and kwargs["embedding_complete_seg_overlap"]:
            self.embedding_complete_seg_overlap = kwargs["embedding_complete_seg_overlap"]
            print("Encoder: embedding_complete_seg_overlap=%r" % self.embedding_complete_seg_overlap)
        else:
            self.embedding_complete_seg_overlap = False

        if "matrix_embedding_exists" in kwargs and kwargs["matrix_embedding_exists"]:
            self.matrix_embedding_exists = kwargs["matrix_embedding_exists"]
        else:
            self.matrix_embedding_exists = False

        if local_rank == -1 and not use_cpu and torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.cuda.is_available() and local_rank > -1:
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        print("Encoder device: ", device)
        self.device = device
        self.seq_id_2_emb_filename = {}
        # embedding buffer
        self.embedding_buffer = {}
        if "buffer_size" in kwargs:
            self.embedding_buffer_size = kwargs["buffer_size"]
        else:
            self.embedding_buffer_size = 0
        print("Encoder: prepend_bos=%r, append_eos=%r" % (self.prepend_bos, self.append_eos))
        print("Encoder: matrix_add_special_token=%r, "
              "embedding_complete=%r, "
              "embedding_complete_seg_overlap=%r, "
              "embedding_fixed_len_a_time=%d, "
              "matrix_embedding_exists=%r" %
              (self.matrix_add_special_token,
               self.embedding_complete,
               self.embedding_complete_seg_overlap,
               self.embedding_fixed_len_a_time if self.embedding_fixed_len_a_time else -1,
               self.matrix_embedding_exists)
              )
        print("-" * 50)

    def put_into_buffer(self, seq_id, embedding_info):
        if self.embedding_buffer_size > 0:
            if len(self.embedding_buffer) >= self.embedding_buffer_size:
                self.embedding_buffer = {}
            self.embedding_buffer[seq_id] = embedding_info

    def __get_embedding__(self, seq_id, seq_type, seq, embedding_type):
        embedding_info = None
        if seq_id in self.embedding_buffer:
            return self.embedding_buffer[seq_id]
        elif seq_id in self.seq_id_2_emb_filename:
            emb_filename = self.seq_id_2_emb_filename[seq_id]
            try:
                dirpath_list = self.vector_dirpath if embedding_type in ["bos", "vector"] else self.matrix_dirpath
                for dirpath in dirpath_list:
                    emb_filepath = os.path.join(dirpath, emb_filename)
                    if os.path.exists(emb_filepath):
                        embedding_info = torch.load(emb_filepath)
                        self.put_into_buffer(seq_id, embedding_info)
                        return embedding_info
            except Exception as e:
                print(e)
                embedding_info = None
        elif embedding_type in ["bos", "vector"] and self.vector_dirpath is not None or embedding_type not in ["bos", "vector"] and self.matrix_dirpath is not None:
            emb_filename = calc_emb_filename_by_seq_id(seq_id=seq_id, embedding_type=embedding_type)
            try:
                dirpath_list = self.vector_dirpath if embedding_type in ["bos", "vector"] else self.matrix_dirpath
                for dirpath in dirpath_list:
                    emb_filepath = os.path.join(dirpath, emb_filename)
                    if os.path.exists(emb_filepath):
                        embedding_info = torch.load(emb_filepath)
                        self.seq_id_2_emb_filename[seq_id] = emb_filename
                        self.put_into_buffer(seq_id, embedding_info)
                        return embedding_info
            except Exception as e:
                print(e)
                embedding_info = None

        if embedding_info is None:
            if self.matrix_embedding_exists:
                with open("matrix_embedding_not_exists.txt", "a+") as wfp:
                    print("seq_id: %s" % seq_id)
                    wfp.write("seq_id: %s\n" % seq_id)
                    wfp.flush()

        if embedding_info is None:
            if self.matrix_embedding_exists:
                print("seq_id: %s 's embedding file not exists in advance" % seq_id)
                print(1/0)

            if "onehot" in self.llm_type:
                if "multi_" in seq_type:
                    embedding_info = []
                    cur_seqs = seq.split(",")
                    if hasattr(self, "max_sentences"):
                        if self.trunc_type == "right":
                            cur_seqs = cur_seqs[:self.max_sentences]
                        else:
                            cur_seqs = cur_seqs[-self.max_sentences:]
                    for cur_seq in cur_seqs:
                        truncation_seq_length = self.seq_max_length - int(self.prepend_bos) - int(self.append_eos)
                        if len(cur_seq) > truncation_seq_length:
                            if self.trunc_type == "right":
                                cur_seq = cur_seq[:truncation_seq_length]
                            else:
                                cur_seq = cur_seq[-truncation_seq_length:]

                        if self.matrix_add_special_token:
                            cur_embedding_info = np.zeros((truncation_seq_length + 2, self.alphabet.vocab_size))
                            cur_embedding_info[0][self.alphabet.get_idx("[CLS]")] = 1.0
                            cur_embedding_info[-1][self.alphabet.get_idx("[SEP]")] = 1.0
                            for ch_idx, ch in enumerate(cur_seq):
                                cur_embedding_info[ch_idx + 1][self.alphabet.get_idx(ch)] = 1.0
                        else:
                            cur_embedding_info = np.zeros((truncation_seq_length, self.alphabet.vocab_size))
                            for ch_idx, ch in enumerate(cur_seq):
                                cur_embedding_info[ch_idx][self.alphabet.get_idx(ch)] = 1.0

                        embedding_info.append(cur_embedding_info)
                else:
                    truncation_seq_length = self.seq_max_length - int(self.prepend_bos) - int(self.append_eos)
                    cur_seq = seq
                    if len(cur_seq) > truncation_seq_length:
                        if self.trunc_type == "right":
                            cur_seq = cur_seq[:truncation_seq_length]
                        else:
                            cur_seq = cur_seq[-truncation_seq_length:]

                    if self.matrix_add_special_token:
                        cur_embedding_info = np.zeros((truncation_seq_length + 2, self.alphabet.vocab_size))
                        cur_embedding_info[0][self.alphabet.get_idx("[CLS]")] = 1.0
                        cur_embedding_info[-1][self.alphabet.get_idx("[SEP]")] = 1.0
                        for ch_idx, ch in enumerate(cur_seq):
                            cur_embedding_info[ch_idx + 1][self.alphabet.get_idx(ch)] = 1.0
                    else:
                        cur_embedding_info = np.zeros((truncation_seq_length, self.alphabet.vocab_size))
                        for ch_idx, ch in enumerate(cur_seq):
                            cur_embedding_info[ch_idx][self.alphabet.get_idx(ch)] = 1.0
                    embedding_info = cur_embedding_info
            elif "esm" in self.llm_type and "prot" in seq_type:
                if seq_type == "multi_prot":
                    embedding_info = []
                    cur_seqs = seq.split(",")
                    if hasattr(self, "max_sentences"):
                        if self.trunc_type == "right":
                            cur_seqs = cur_seqs[:self.max_sentences]
                        else:
                            cur_seqs = cur_seqs[-self.max_sentences:]
                    for cur_seq in cur_seqs:
                        cur_seq_len = len(cur_seq)
                        truncation_seq_length = self.seq_max_length - int(self.prepend_bos) - int(self.append_eos)
                        while True:
                            cur_embedding_info, cur_processed_seq_len = predict_embedding_esm(
                                [seq_id, cur_seq],
                                self.trunc_type,
                                embedding_type,
                                repr_layers=[-1],
                                truncation_seq_length=truncation_seq_length,
                                matrix_add_special_token=self.matrix_add_special_token,
                                device=self.device
                            )
                            if cur_embedding_info is not None:
                                break
                            truncation_seq_length = (truncation_seq_length + int(self.prepend_bos) + int(self.append_eos)) * 0.95 \
                                                    - int(self.prepend_bos) - int(self.append_eos)
                            truncation_seq_length = int(truncation_seq_length)
                            print("%s embedding error, truncation_seq_length: %d->%d" % (seq_id, cur_seq_len, truncation_seq_length))
                        embedding_info.append(cur_embedding_info)
                else:
                    truncation_seq_length = self.seq_max_length - int(self.prepend_bos) - int(self.append_eos)
                    cur_seq_len = len(seq)
                    while True:
                        embedding_info, processed_seq_len = predict_embedding_esm(
                            [seq_id, seq],
                            self.trunc_type,
                            embedding_type,
                            repr_layers=[-1],
                            truncation_seq_length=truncation_seq_length,
                            matrix_add_special_token=self.matrix_add_special_token,
                            device=self.device
                        )
                        if embedding_info is not None:
                            break
                        truncation_seq_length = (truncation_seq_length + int(self.prepend_bos) + int(self.append_eos)) * 0.95 \
                                                - int(self.prepend_bos) - int(self.append_eos)
                        truncation_seq_length = int(truncation_seq_length)
                        print("%s embedding error, truncation_seq_length: %d->%d" % (seq_id, cur_seq_len, truncation_seq_length))
            elif "dnaberts" in self.llm_type and "gene" in seq_type:
                if seq_type == "multi_gene":
                    embedding_info = []
                    cur_seqs = seq.split(",")
                    if hasattr(self, "max_sentences"):
                        if self.trunc_type == "right":
                            cur_seqs = cur_seqs[:self.max_sentences]
                        else:
                            cur_seqs = cur_seqs[-self.max_sentences:]
                    for cur_seq in cur_seqs:
                        cur_seq_len = len(cur_seq)
                        truncation_seq_length = self.seq_max_length - int(self.prepend_bos) - int(self.append_eos)
                        while True:
                            cur_embedding_info, cur_processed_seq_len = predict_embedding_dnaberts(
                                [seq_id, cur_seq],
                                self.trunc_type,
                                embedding_type,
                                repr_layers=[-1],
                                truncation_seq_length=truncation_seq_length,
                                matrix_add_special_token=self.matrix_add_special_token,
                                device=self.device
                            )
                            if cur_embedding_info is not None:
                                break
                            truncation_seq_length = (truncation_seq_length + int(self.prepend_bos) + int(self.append_eos)) * 0.95 \
                                                    - int(self.prepend_bos) - int(self.append_eos)
                            truncation_seq_length = int(truncation_seq_length)
                            print("%s embedding error, truncation_seq_length: %d->%d" % (seq_id, cur_seq_len, truncation_seq_length))
                        embedding_info.append(cur_embedding_info)
                else:
                    cur_seq_len = len(seq)
                    truncation_seq_length = self.seq_max_length - int(self.prepend_bos) - int(self.append_eos)
                    while True:
                        embedding_info, processed_seq_len = predict_embedding_dnaberts(
                            [seq_id, seq],
                            self.trunc_type,
                            embedding_type,
                            repr_layers=[-1],
                            truncation_seq_length=truncation_seq_length,
                            matrix_add_special_token=self.matrix_add_special_token,
                            device=self.device
                        )
                        if embedding_info is not None:
                            break
                        truncation_seq_length = (truncation_seq_length + int(self.prepend_bos) + int(self.append_eos)) * 0.95 \
                                                - int(self.prepend_bos) - int(self.append_eos)
                        truncation_seq_length = int(truncation_seq_length)
                        print("%s embedding error, truncation_seq_length: %d->%d" % (seq_id, cur_seq_len, truncation_seq_length))
            elif ("dnabert2" in self.llm_type or "dnabert" in self.llm_type) and "gene" in seq_type:
                if seq_type == "multi_gene":
                    embedding_info = []
                    cur_seqs = seq.split(",")
                    if hasattr(self, "max_sentences"):
                        if self.trunc_type == "right":
                            cur_seqs = cur_seqs[:self.max_sentences]
                        else:
                            cur_seqs = cur_seqs[-self.max_sentences:]
                    for cur_seq in cur_seqs:
                        cur_seq_len = len(cur_seq)
                        truncation_seq_length = self.seq_max_length - int(self.prepend_bos) - int(self.append_eos)
                        while True:
                            cur_embedding_info, cur_processed_seq_len = predict_embedding_dnabert2(
                                [seq_id, cur_seq],
                                self.trunc_type,
                                embedding_type,
                                repr_layers=[-1],
                                truncation_seq_length=truncation_seq_length,
                                matrix_add_special_token=self.matrix_add_special_token,
                                device=self.device
                            )

                            if cur_embedding_info is not None:
                                break
                            truncation_seq_length = (truncation_seq_length + int(self.prepend_bos) + int(self.append_eos)) * 0.95 \
                                                    - int(self.prepend_bos) - int(self.append_eos)
                            truncation_seq_length = int(truncation_seq_length)
                            print("%s embedding error, truncation_seq_length: %d->%d" % (seq_id, cur_seq_len, truncation_seq_length))
                        embedding_info.append(cur_embedding_info)
                else:
                    cur_seq_len = len(seq)
                    truncation_seq_length = self.seq_max_length - int(self.prepend_bos) - int(self.append_eos)
                    while True:
                        embedding_info, processed_seq = predict_embedding_dnabert2(
                            [seq_id, seq],
                            self.trunc_type,
                            embedding_type,
                            repr_layers=[-1],
                            truncation_seq_length=truncation_seq_length,
                            matrix_add_special_token=self.matrix_add_special_token,
                            device=self.device
                        )

                        if embedding_info is not None:
                            break
                        truncation_seq_length = (truncation_seq_length + int(self.prepend_bos) + int(self.append_eos)) * 0.95 \
                                                - int(self.prepend_bos) - int(self.append_eos)
                        truncation_seq_length = int(truncation_seq_length)
                        print("%s embedding error, truncation_seq_length: %d->%d" % (seq_id, cur_seq_len, truncation_seq_length))
            elif "molecule" in seq_type:
                # to do
                pass
            elif "lucaone" in self.llm_type and "lucaone-separated" in self.llm_version:
                if seq_type in ["multi_gene", "multi_prot"]:
                    embedding_info = []
                    cur_seqs = seq.split(",")
                    if hasattr(self, "max_sentences"):
                        # print("self.trunc_type: %s" % self.trunc_type)
                        if self.trunc_type == "right":
                            cur_seqs = cur_seqs[:self.max_sentences]
                        else:
                            cur_seqs = cur_seqs[-self.max_sentences:]
                    for cur_seq in cur_seqs:
                        cur_seq_len = len(cur_seq)
                        if hasattr(self, "embedding_complete") and self.embedding_complete:
                            truncation_seq_length = min(cur_seq_len, MAX_SEQ_LEN)
                        else:
                            truncation_seq_length = self.seq_max_length - int(self.prepend_bos) - int(self.append_eos)
                            truncation_seq_length = min(cur_seq_len, truncation_seq_length)
                        cur_seq_type = "gene" if seq_type == "multi_gene" else "prot"
                        while True:
                            # 设置了一次性推理长度
                            if self.embedding_fixed_len_a_time and self.embedding_fixed_len_a_time > 0:
                                cur_embedding_info, cur_processed_seq = predict_embedding_luca(
                                    self.llm_dirpath,
                                    [seq_id, cur_seq_type, cur_seq],
                                    self.trunc_type,
                                    embedding_type,
                                    repr_layers=[-1],
                                    truncation_seq_length=self.embedding_fixed_len_a_time,
                                    matrix_add_special_token=self.matrix_add_special_token,
                                    device=self.device
                                )
                                use_cpu = False
                                if cur_embedding_info is None:
                                    cur_embedding_info, cur_processed_seq = predict_embedding_luca(
                                        self.llm_dirpath,
                                        [seq_id, cur_seq_type, cur_seq],
                                        self.trunc_type,
                                        embedding_type,
                                        repr_layers=[-1],
                                        truncation_seq_length=self.embedding_fixed_len_a_time,
                                        matrix_add_special_token=self.matrix_add_special_token,
                                        device=torch.device("cpu")
                                    )
                                    use_cpu = True

                                if cur_embedding_info is not None and hasattr(self, "embedding_complete") \
                                        and self.embedding_complete and cur_seq_len > self.embedding_fixed_len_a_time:
                                    cur_embedding_info = complete_embedding_matrix(
                                        seq_id,
                                        cur_seq_type,
                                        cur_seq,
                                        self.embedding_fixed_len_a_time,
                                        cur_embedding_info,
                                        self.llm_dirpath,
                                        self.trunc_type,
                                        embedding_type,
                                        self.matrix_add_special_token,
                                        self.embedding_complete,
                                        self.embedding_complete_seg_overlap,
                                        self.device,
                                        use_cpu=use_cpu
                                    )
                            else:
                                cur_embedding_info, cur_processed_seq = predict_embedding_luca(
                                    self.llm_dirpath,
                                    [seq_id, cur_seq_type, cur_seq],
                                    self.trunc_type,
                                    embedding_type,
                                    repr_layers=[-1],
                                    truncation_seq_length=truncation_seq_length,
                                    matrix_add_special_token=self.matrix_add_special_token,
                                    device=self.device
                                )
                                use_cpu = False
                                if cur_embedding_info is None:
                                    cur_embedding_info, cur_processed_seq = predict_embedding_luca(
                                        self.llm_dirpath,
                                        [seq_id, cur_seq_type, cur_seq],
                                        self.trunc_type,
                                        embedding_type,
                                        repr_layers=[-1],
                                        truncation_seq_length=truncation_seq_length,
                                        matrix_add_special_token=self.matrix_add_special_token,
                                        device=torch.device("cpu")
                                    )
                                    use_cpu = True

                                if cur_embedding_info is not None and hasattr(self, "embedding_complete") \
                                        and self.embedding_complete and cur_seq_len > truncation_seq_length:
                                    cur_embedding_info = complete_embedding_matrix(
                                        seq_id,
                                        cur_seq_type,
                                        cur_seq,
                                        truncation_seq_length,
                                        cur_embedding_info,
                                        self.llm_dirpath,
                                        self.trunc_type,
                                        embedding_type,
                                        self.matrix_add_special_token,
                                        self.embedding_complete,
                                        self.embedding_complete_seg_overlap,
                                        self.device,
                                        use_cpu=use_cpu
                                    )
                            if use_cpu:
                                print("use_cpu: %r" % use_cpu)
                            if cur_embedding_info is not None:
                                break
                            truncation_seq_length = (truncation_seq_length + int(self.prepend_bos) + int(self.append_eos)) * 0.95 \
                                                    - int(self.prepend_bos) - int(self.append_eos)
                            truncation_seq_length = int(truncation_seq_length)
                            print("truncation_seq_length: %d->%s" % (cur_seq_len, truncation_seq_length))
                        embedding_info.append(cur_embedding_info)
                else:
                    cur_seq_len = len(seq)
                    if hasattr(self, "embedding_complete") and self.embedding_complete:
                        truncation_seq_length = min(cur_seq_len, MAX_SEQ_LEN)
                    else:
                        # to do
                        truncation_seq_length = self.seq_max_length - int(self.prepend_bos) - int(self.append_eos)
                        truncation_seq_length = min(cur_seq_len, truncation_seq_length)
                    while True:
                        # 设置了一次性推理长度
                        if self.embedding_fixed_len_a_time and self.embedding_fixed_len_a_time > 0:
                            embedding_info, processed_seq = predict_embedding_luca(
                                self.llm_dirpath,
                                [seq_id, seq_type, seq],
                                self.trunc_type,
                                embedding_type,
                                repr_layers=[-1],
                                truncation_seq_length=self.embedding_fixed_len_a_time,
                                matrix_add_special_token=self.matrix_add_special_token,
                                device=self.device
                            )
                            use_cpu = False
                            if embedding_info is None:
                                embedding_info, processed_seq = predict_embedding_luca(
                                    self.llm_dirpath,
                                    [seq_id, seq_type, seq],
                                    self.trunc_type,
                                    embedding_type,
                                    repr_layers=[-1],
                                    truncation_seq_length=self.embedding_fixed_len_a_time,
                                    matrix_add_special_token=self.matrix_add_special_token,
                                    device=torch.device("cpu")
                                )
                                use_cpu = True
                            if embedding_info is not None and \
                                    hasattr(self, "embedding_complete") and self.embedding_complete and cur_seq_len > self.embedding_fixed_len_a_time:
                                embedding_info = complete_embedding_matrix(
                                    seq_id,
                                    seq_type,
                                    seq,
                                    self.embedding_fixed_len_a_time,
                                    embedding_info,
                                    self.llm_dirpath,
                                    self.trunc_type,
                                    embedding_type,
                                    self.matrix_add_special_token,
                                    self.embedding_complete,
                                    self.embedding_complete_seg_overlap,
                                    self.device,
                                    use_cpu=use_cpu
                                )
                        else:
                            embedding_info, processed_seq = predict_embedding_luca(
                                self.llm_dirpath,
                                [seq_id, seq_type, seq],
                                self.trunc_type,
                                embedding_type,
                                repr_layers=[-1],
                                truncation_seq_length=truncation_seq_length,
                                matrix_add_special_token=self.matrix_add_special_token,
                                device=self.device
                            )
                            use_cpu = False
                            if embedding_info is None:
                                embedding_info, processed_seq = predict_embedding_luca(
                                    self.llm_dirpath,
                                    [seq_id, seq_type, seq],
                                    self.trunc_type,
                                    embedding_type,
                                    repr_layers=[-1],
                                    truncation_seq_length=truncation_seq_length,
                                    matrix_add_special_token=self.matrix_add_special_token,
                                    device=torch.device("cpu")
                                )
                                use_cpu = True
                            if embedding_info is not None and \
                                    hasattr(self, "embedding_complete") and self.embedding_complete and cur_seq_len > truncation_seq_length:
                                embedding_info = complete_embedding_matrix(
                                    seq_id,
                                    seq_type,
                                    seq,
                                    truncation_seq_length,
                                    embedding_info,
                                    self.llm_dirpath,
                                    self.trunc_type,
                                    embedding_type,
                                    self.matrix_add_special_token,
                                    self.embedding_complete,
                                    self.embedding_complete_seg_overlap,
                                    self.device,
                                    use_cpu=use_cpu
                                )
                        if use_cpu:
                            print("use_cpu: %r" % use_cpu)
                        if embedding_info is not None:
                            break
                        truncation_seq_length = (truncation_seq_length + int(self.prepend_bos) + int(self.append_eos)) * 0.95 \
                                                - int(self.prepend_bos) - int(self.append_eos)
                        truncation_seq_length = int(truncation_seq_length)
                        print("%s embedding error, truncation_seq_length: %d->%d" % (seq_id, cur_seq_len, truncation_seq_length))
            else:
                if seq_type in ["multi_gene", "multi_prot"]:
                    embedding_info = []
                    cur_seqs = seq.split(",")
                    if hasattr(self, "max_sentences"):
                        # print("self.trunc_type: %s" % self.trunc_type)
                        if self.trunc_type == "right":
                            cur_seqs = cur_seqs[:self.max_sentences]
                        else:
                            cur_seqs = cur_seqs[-self.max_sentences:]
                    for cur_seq in cur_seqs:
                        cur_seq_len = len(cur_seq)
                        if hasattr(self, "embedding_complete") and self.embedding_complete:
                            truncation_seq_length = min(cur_seq_len, MAX_SEQ_LEN)
                        else:
                            truncation_seq_length = self.seq_max_length - int(self.prepend_bos) - int(self.append_eos)
                            truncation_seq_length = min(cur_seq_len, truncation_seq_length)
                        cur_seq_type = "gene" if seq_type == "multi_gene" else "prot"
                        while True:
                            # 设置了一次性推理长度
                            if self.embedding_fixed_len_a_time and self.embedding_fixed_len_a_time > 0:
                                cur_embedding_info, cur_processed_seq = predict_embedding_luca(
                                    self.llm_dirpath,
                                    [seq_id, cur_seq_type, cur_seq],
                                    self.trunc_type,
                                    embedding_type,
                                    repr_layers=[-1],
                                    truncation_seq_length=self.embedding_fixed_len_a_time,
                                    matrix_add_special_token=self.matrix_add_special_token,
                                    device=self.device
                                )
                                use_cpu = False
                                if cur_embedding_info is None:
                                    cur_embedding_info, cur_processed_seq = predict_embedding_luca(
                                        self.llm_dirpath,
                                        [seq_id, cur_seq_type, cur_seq],
                                        self.trunc_type,
                                        embedding_type,
                                        repr_layers=[-1],
                                        truncation_seq_length=self.embedding_fixed_len_a_time,
                                        matrix_add_special_token=self.matrix_add_special_token,
                                        device=torch.device("cpu")
                                    )
                                    use_cpu = True
                                if cur_embedding_info is not None and hasattr(self, "embedding_complete") \
                                        and self.embedding_complete and cur_seq_len > self.embedding_fixed_len_a_time:
                                    cur_embedding_info = complete_embedding_matrix(
                                        seq_id,
                                        cur_seq_type,
                                        cur_seq,
                                        self.embedding_fixed_len_a_time,
                                        cur_embedding_info,
                                        self.llm_dirpath,
                                        self.trunc_type,
                                        embedding_type,
                                        self.matrix_add_special_token,
                                        self.embedding_complete,
                                        self.embedding_complete_seg_overlap,
                                        self.device,
                                        use_cpu=use_cpu
                                    )
                            else:
                                cur_embedding_info, cur_processed_seq = predict_embedding_luca(
                                    self.llm_dirpath,
                                    [seq_id, cur_seq_type, cur_seq],
                                    self.trunc_type,
                                    embedding_type,
                                    repr_layers=[-1],
                                    truncation_seq_length=truncation_seq_length,
                                    matrix_add_special_token=self.matrix_add_special_token,
                                    device=self.device
                                )
                                use_cpu = False
                                if cur_embedding_info is None:
                                    cur_embedding_info, cur_processed_seq = predict_embedding_luca(
                                        self.llm_dirpath,
                                        [seq_id, cur_seq_type, cur_seq],
                                        self.trunc_type,
                                        embedding_type,
                                        repr_layers=[-1],
                                        truncation_seq_length=truncation_seq_length,
                                        matrix_add_special_token=self.matrix_add_special_token,
                                        device=torch.device("cpu")
                                    )
                                    use_cpu = True
                                if cur_embedding_info is not None and hasattr(self, "embedding_complete") \
                                        and self.embedding_complete and cur_seq_len > truncation_seq_length:
                                    cur_embedding_info = complete_embedding_matrix(
                                        seq_id,
                                        cur_seq_type,
                                        cur_seq,
                                        truncation_seq_length,
                                        cur_embedding_info,
                                        self.llm_dirpath,
                                        self.trunc_type,
                                        embedding_type,
                                        self.matrix_add_special_token,
                                        self.embedding_complete,
                                        self.embedding_complete_seg_overlap,
                                        self.device,
                                        use_cpu=use_cpu
                                    )
                            if use_cpu:
                                print("use_cpu: %r" % use_cpu)
                            if cur_embedding_info is not None:
                                break
                            truncation_seq_length = (truncation_seq_length + int(self.prepend_bos) + int(self.append_eos)) * 0.95 \
                                                    - int(self.prepend_bos) - int(self.append_eos)
                            truncation_seq_length = int(truncation_seq_length)
                            print("%s embedding error, truncation_seq_length: %d->%d" % (seq_id, cur_seq_len, truncation_seq_length))
                        embedding_info.append(cur_embedding_info)
                else:
                    cur_seq_len = len(seq)
                    if hasattr(self, "embedding_complete") and self.embedding_complete:
                        truncation_seq_length = min(cur_seq_len, MAX_SEQ_LEN)
                    else:
                        truncation_seq_length = self.seq_max_length - int(self.prepend_bos) - int(self.append_eos)
                        truncation_seq_length = min(cur_seq_len, truncation_seq_length)

                    while True:
                        # 设置了一次性推理长度
                        if self.embedding_fixed_len_a_time and self.embedding_fixed_len_a_time > 0:
                            embedding_info, processed_seq = predict_embedding_luca(
                                self.llm_dirpath,
                                [seq_id, seq_type, seq],
                                self.trunc_type,
                                embedding_type,
                                repr_layers=[-1],
                                truncation_seq_length=self.embedding_fixed_len_a_time,
                                matrix_add_special_token=self.matrix_add_special_token,
                                device=self.device
                            )
                            use_cpu = False
                            if embedding_info is None:
                                embedding_info, processed_seq = predict_embedding_luca(
                                    self.llm_dirpath,
                                    [seq_id, seq_type, seq],
                                    self.trunc_type,
                                    embedding_type,
                                    repr_layers=[-1],
                                    truncation_seq_length=self.embedding_fixed_len_a_time,
                                    matrix_add_special_token=self.matrix_add_special_token,
                                    device=torch.device("cpu")
                                )
                                use_cpu = True
                            if embedding_info is not None and hasattr(self, "embedding_complete") and self.embedding_complete \
                                    and cur_seq_len > self.embedding_fixed_len_a_time:
                                embedding_info = complete_embedding_matrix(
                                    seq_id,
                                    seq_type,
                                    seq,
                                    self.embedding_fixed_len_a_time,
                                    embedding_info,
                                    self.llm_dirpath,
                                    self.trunc_type,
                                    embedding_type,
                                    self.matrix_add_special_token,
                                    self.embedding_complete,
                                    self.embedding_complete_seg_overlap,
                                    self.device,
                                    use_cpu=use_cpu
                                )
                        else:
                            embedding_info, processed_seq = predict_embedding_luca(
                                self.llm_dirpath,
                                [seq_id, seq_type, seq],
                                self.trunc_type,
                                embedding_type,
                                repr_layers=[-1],
                                truncation_seq_length=truncation_seq_length,
                                matrix_add_special_token=self.matrix_add_special_token,
                                device=self.device
                            )
                            use_cpu = False
                            if embedding_info is None:
                                embedding_info, processed_seq = predict_embedding_luca(
                                    self.llm_dirpath,
                                    [seq_id, seq_type, seq],
                                    self.trunc_type,
                                    embedding_type,
                                    repr_layers=[-1],
                                    truncation_seq_length=truncation_seq_length,
                                    matrix_add_special_token=self.matrix_add_special_token,
                                    device=torch.device("cpu")
                                )
                                use_cpu = True
                            if embedding_info is not None and hasattr(self, "embedding_complete") and self.embedding_complete \
                                    and cur_seq_len > truncation_seq_length:
                                embedding_info = complete_embedding_matrix(
                                    seq_id,
                                    seq_type,
                                    seq,
                                    truncation_seq_length,
                                    embedding_info,
                                    self.llm_dirpath,
                                    self.trunc_type,
                                    embedding_type,
                                    self.matrix_add_special_token,
                                    self.embedding_complete,
                                    self.embedding_complete_seg_overlap,
                                    self.device,
                                    use_cpu=use_cpu
                                )
                        if use_cpu:
                            print("use_cpu: %r" % use_cpu)
                        if embedding_info is not None:
                            break
                        truncation_seq_length = (truncation_seq_length + int(self.prepend_bos) + int(self.append_eos)) * 0.95 \
                                                - int(self.prepend_bos) - int(self.append_eos)
                        truncation_seq_length = int(truncation_seq_length)
                        print("%s embedding error, truncation_seq_length: %d->%d" % (seq_id, cur_seq_len, truncation_seq_length))
            if embedding_type in ["bos", "vector"] and self.vector_dirpath is not None \
                    or embedding_type not in ["bos", "vector"] and self.matrix_dirpath is not None:
                emb_filename = calc_emb_filename_by_seq_id(seq_id=seq_id, embedding_type=embedding_type)
                dirpath_list = self.vector_dirpath if embedding_type in ["bos", "vector"] else self.matrix_dirpath
                dirpath = dirpath_list[0]
                emb_filepath = os.path.join(dirpath, emb_filename)
                # print("seq_len: %d" % len(seq))
                # print("emb shape:", embedding_info.shape)
                torch.save(embedding_info, emb_filepath)
                self.seq_id_2_emb_filename[seq_id] = emb_filename
                self.put_into_buffer(seq_id, embedding_info)
        return embedding_info

    def encode_single(
            self,
            seq_id,
            seq_type,
            seq,
            vector_filename=None,
            matrix_filename=None,
            express_list=None,
            label=None
    ):
        seq_type = seq_type.strip().lower()
        # for embedding vector
        vector = None
        if self.input_type in ["vector", "seq_vector"]:
            if vector_filename is None:
                if seq is None:
                    raise Exception("seq is none and vector_filename is none")
                elif seq_type == "molecule":
                    raise Exception("now not support embedding of the seq_type=%s" % seq_type)
                else:
                    vector = self.__get_embedding__(seq_id, seq_type, seq, "vector")
            elif isinstance(vector_filename, str):
                for vector_dir in self.vector_dirpath:
                    vector_filepath = os.path.join(vector_dir, vector_filename)
                    if os.path.exists(vector_filepath):
                        vector = torch.load(vector_filepath)
                        break
            elif isinstance(vector_filename, np.ndarray):
                vector = vector_filename
            else:
                raise Exception("vector is not filepath-str and np.ndarray")

        # for embedding matrix
        matrix = None
        if self.input_type in ["matrix", "seq_matrix", "matrix_express"]:
            if matrix_filename is None:
                if seq is None:
                    raise Exception("seq is none and matrix_filename is none")
                elif seq_type == "molecule":
                    raise Exception("now not support embedding of the seq_type=%s" % seq_type)
                else:
                    matrix = self.__get_embedding__(seq_id, seq_type, seq, "matrix")
            elif isinstance(matrix_filename, str):
                for matrix_dir in self.matrix_dirpath:
                    matrix_filepath = os.path.join(matrix_dir, matrix_filename)
                    if os.path.exists(matrix_filepath):
                        matrix = torch.load(matrix_filepath)
                        break
            elif isinstance(matrix_filename, np.ndarray):
                matrix = matrix_filename
            else:
                raise Exception("matrix is not filepath-str and np.ndarray")

        # for seq
        if seq_type == "molecule":
            # to do
            pass
        else:
            seq = seq.upper()

        '''
        Asx、B可代表天冬氨酸（Asp、D）或天冬酰胺（Asn、N）。
        Glx、Z可代表谷氨酸（Glu、E）或谷氨酰胺（Gln、Q）。
        Xle、J可代表亮氨酸（Leu、L）或异亮氨酸（Ile、I）。
        Xaa（亦用Unk）、X可代表任意氨基酸或未知氨基酸。
        '''
        # 蛋白质且使用esm进行embedding，则需要去掉蛋白质J
        if self.input_type in ["matrix", "seq_matrix"] and "esm" in self.llm_type:
            if seq_type == "prot":
                seq = clean_seq_esm(seq_id, seq)
            elif seq_type == "multi_prot":
                seq = ",".join([clean_seq_esm(seq_id, v) for v in seq.split(",")])

        if express_list is not None:
            if not isinstance(express_list, list):
                express_list = eval(express_list)
        return {
            "seq_id": seq_id,
            "seq": seq,
            "seq_type": seq_type,
            "vector": vector,
            "matrix": matrix,
            "express_list": express_list,
            "label": label
        }

    def encode_pair(
            self,
            seq_id_a,
            seq_id_b,
            seq_type_a,
            seq_type_b,
            seq_a,
            seq_b,
            vector_filename_a=None,
            vector_filename_b=None,
            matrix_filename_a=None,
            matrix_filename_b=None,
            express_list_a=None,
            express_list_b=None,
            label=None

    ):
        seq_type_a = seq_type_a.strip().lower()
        seq_type_b = seq_type_b.strip().lower()
        # for embedding vector
        vector_a, vector_b = None, None
        if self.input_type in [
            "vector",
            "seq_vector",
            "seq_vs_vector",
            "vector_vs_seq",
            "vector_vs_vector",
            "vector_vs_matrix",
            "matrix_vs_vector"
        ]:
            if self.input_type not in ["seq_vs_vector", "matrix_vs_vector"]:
                if vector_filename_a is None:
                    if seq_a is None:
                        raise Exception("seq_a is none and vector_filename_a is none")
                    elif seq_type_a == "molecule":
                        raise Exception("now not support embedding of the seq_type_a=%s" % seq_type_a)
                    else:
                        vector_a = self.__get_embedding__(seq_id_a, seq_type_a, seq_a, "vector")
                elif isinstance(vector_filename_a, str):
                    for vector_dir in self.vector_dirpath:
                        vector_filepath_a = os.path.join(vector_dir, vector_filename_a)
                        if os.path.exists(vector_filepath_a):
                            vector_a = torch.load(vector_filepath_a)
                            break
                elif isinstance(vector_filename_a, np.ndarray):
                    vector_a = vector_filename_a
                else:
                    raise Exception("vector_a is not filepath-str and np.ndarray")
            if self.input_type not in ["vector_vs_seq", "vector_vs_matrix"]:
                if vector_filename_b is None:
                    if seq_b is None:
                        raise Exception("seq_b is none and vector_filename_b is none")
                    elif seq_type_b == "molecule":
                        raise Exception("now not support embedding of the seq_type_b=%s" % seq_type_b)
                    else:
                        vector_b = self.__get_embedding__(seq_id_b, seq_type_b, seq_b, "vector")
                elif isinstance(vector_filename_b, str):
                    for vector_dir in self.vector_dirpath:
                        vector_filepath_b = os.path.join(vector_dir, vector_filename_b)
                        if os.path.exists(vector_filepath_b):
                            vector_b = torch.load(vector_filepath_b)
                            break
                elif isinstance(vector_filename_b, np.ndarray):
                    vector_b = vector_filename_b
                else:
                    raise Exception("vector_b is not filepath-str and np.ndarray")

        # for embedding matrix
        matrix_a, matrix_b = None, None
        if self.input_type in [
            "matrix",
            "seq_matrix",
            "seq_vs_matrix",
            "vector_vs_matrix",
            "matrix_vs_seq",
            "matrix_vs_vector",
            "matrix_vs_matrix",
            "matrix_express_vs_matrix",
            "matrix_express_vs_matrix_express"
        ]:
            if self.input_type not in ["seq_vs_matrix", "vector_vs_matrix"]:
                if matrix_filename_a is None:
                    if seq_a is None:
                        raise Exception("seq_a is none and matrix_filename_a is none")
                    else:
                        matrix_a = self.__get_embedding__(seq_id_a, seq_type_a, seq_a, "matrix")
                elif isinstance(matrix_filename_a, str):
                    for matrix_dir in self.matrix_dirpath:
                        matrix_filepath_a = os.path.join(matrix_dir, matrix_filename_a)
                        if os.path.exists(matrix_filepath_a):
                            matrix_a = torch.load(matrix_filepath_a)
                            break
                elif isinstance(matrix_filename_a, np.ndarray):
                    matrix_a = matrix_filename_a
                else:
                    raise Exception("matrix_a is not filepath-str and np.ndarray")
            if self.input_type not in ["matrix_vs_seq", "matrix_vs_vector"]:
                if matrix_filename_b is None:
                    if seq_b is None:
                        raise Exception("seq_b is none and matrix_filename_b is none")
                    else:
                        matrix_b = self.__get_embedding__(seq_id_b, seq_type_b, seq_b, "matrix")
                elif isinstance(matrix_filename_b, str):
                    for matrix_dir in self.matrix_dirpath:
                        matrix_filepath_b = os.path.join(matrix_dir, matrix_filename_b)
                        if os.path.exists(matrix_filepath_b):
                            matrix_b = torch.load(matrix_filepath_b)
                            break
                elif isinstance(matrix_filename_b, np.ndarray):
                    matrix_b = matrix_filename_b
                else:
                    raise Exception("matrix_b is not filepath-str and np.ndarray")

        # for seq
        if seq_type_a == "molecule":
            # to do
            pass
        elif "seq" in self.input_type:
            seq_a = seq_a.upper()
        if seq_type_b == "molecule":
            # to do
            pass
        elif "seq" in self.input_type:
            seq_b = seq_b.upper()
        # 蛋白质且使用esm进行embedding，则需要去掉蛋白质J
        if "matrix" in self.input_type and "esm" in self.llm_type:
            if self.input_type not in ["seq_vs_matrix", "vector_vs_matrix"]:
                if seq_type_a == "prot":
                    seq_a = clean_seq_esm(seq_id_a, seq_a)
                elif seq_type_a == "multi_prot":
                    seq_a = ",".join([clean_seq_esm(seq_id_a, v) for v in seq_a.split(",")])
            if self.input_type not in ["matrix_vs_seq", "matrix_vs_vector"]:
                if seq_type_b == "prot":
                    seq_b = clean_seq_esm(seq_id_b, seq_b)
                elif seq_type_b == "multi_prot":
                    seq_b = ",".join([clean_seq_esm(seq_id_b, v) for v in seq_b.split(",")])
        if express_list_a is not None:
            if not isinstance(express_list_a, list):
                express_list_a = eval(express_list_a)
        if express_list_b is not None:
            if not isinstance(express_list_b, list):
                express_list_b = eval(express_list_b)
        return {
            "seq_id_a": seq_id_a,
            "seq_a": seq_a,
            "seq_type_a": seq_type_a,
            "vector_a": vector_a,
            "matrix_a": matrix_a,
            "express_list_a": express_list_a,
            "seq_id_b": seq_id_b,
            "seq_b": seq_b,
            "seq_type_b": seq_type_b,
            "vector_b": vector_b,
            "matrix_b": matrix_b,
            "express_list_b": express_list_b,
            "label": label
        }



