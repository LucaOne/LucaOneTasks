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
    from .utils import clean_seq, device_memory
    from .common.alphabet import Alphabet
    from .llm.lucagplm.get_embedding import predict_embedding as predict_embedding_luca
    from .llm.esm.predict_embedding import predict_embedding as predict_embedding_esm
    from .llm.dnabert2.inference_embedding import predict_embedding as predict_embedding_dnabert2
    from .llm.dnaberts.inference_embedding import predict_embedding as predict_embedding_dnaberts
except ImportError as e:
    from src.utils import clean_seq, device_memory
    from src.common.alphabet import Alphabet
    from src.llm.lucagplm.get_embedding import predict_embedding as predict_embedding_luca
    from src.llm.esm.predict_embedding import predict_embedding as predict_embedding_esm
    from src.llm.dnabert2.inference_embedding import predict_embedding as predict_embedding_dnabert2
    from src.llm.dnaberts.inference_embedding import predict_embedding as predict_embedding_dnaberts


def complete_embedding_matrix(seq_id,
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
                              device):
    if init_emb is not None and embedding_complete and ("representations" in embedding_type or "matrix" in embedding_type):
        ori_seq_len = min(len(seq), 10000)
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
            print("Embedding Complete Seg Overlap: %r, ori seq len: %d, segment len: %d, init sliding windown: %d" % (embedding_complete_seg_overlap, ori_seq_len, init_cur_segment_len, sliding_window))
            while True:
                print("updated window: %d" % sliding_window)
                try:
                    # 第一个已经处理，滑动窗口
                    if trunc_type == "right":
                        last_end = 0
                        seg_idx = 0
                        for pos_idx in range(init_cur_segment_len, ori_seq_len - sliding_window, sliding_window):
                            seg_idx += 1
                            last_end = min(pos_idx + sliding_window, ori_seq_len)
                            seg_seq = seq[pos_idx - sliding_window:last_end]
                            print("segment idx: %d, seg seq len: %d" % (seg_idx, len(seg_seq)))
                            seg_emb, seg_processed_seq_len = predict_embedding_luca(llm_dirpath,
                                                                                    [seq_id + "_seg_%d" % seg_idx, seq_type, seg_seq],
                                                                                    trunc_type,
                                                                                    embedding_type,
                                                                                    repr_layers=[-1],
                                                                                    truncation_seq_length=truncation_seq_length,
                                                                                    matrix_add_special_token=False,
                                                                                    device=device)
                            # 有seq overlap 所以要截取
                            if append_emb is None:
                                append_emb = seg_emb[sliding_window:]
                            else:
                                append_emb = np.concatenate((append_emb, seg_emb[sliding_window:]), axis=0)
                        if last_end < ori_seq_len:
                            seg_idx += 1
                            remain = ori_seq_len - last_end
                            seg_seq = seq[ori_seq_len - 2 * sliding_window:ori_seq_len]
                            seg_emb, seg_processed_seq_len = predict_embedding_luca(llm_dirpath,
                                                                                    [seq_id + "_seg_%d" % seg_idx, seq_type, seg_seq],
                                                                                    trunc_type,
                                                                                    embedding_type,
                                                                                    repr_layers=[-1],
                                                                                    truncation_seq_length=truncation_seq_length,
                                                                                    matrix_add_special_token=False,
                                                                                    device=device)
                            # 有seq overlap 所以要截取
                            if append_emb is None:
                                append_emb = seg_emb[-remain:]
                            else:
                                append_emb = np.concatenate((append_emb, seg_emb[-remain:]), axis=0)
                    else:
                        last_start = -init_cur_segment_len - sliding_window
                        seg_idx = 0
                        for pos_idx in range(-init_cur_segment_len, -ori_seq_len + sliding_window, -sliding_window):
                            seg_idx += 1
                            last_start = min(pos_idx - sliding_window, -ori_seq_len)
                            seg_seq = seq[last_start: pos_idx + sliding_window]
                            seg_emb, seg_processed_seq_len = predict_embedding_luca(llm_dirpath,
                                                                                    [seq_id + "_seg_%d" % seg_idx, seq_type, seg_seq],
                                                                                    trunc_type,
                                                                                    embedding_type,
                                                                                    repr_layers=[-1],
                                                                                    truncation_seq_length=truncation_seq_length,
                                                                                    matrix_add_special_token=False,
                                                                                    device=device)
                            # 有seq overlap 所以要截取
                            if append_emb is None:
                                append_emb = seg_emb[:sliding_window]
                            else:
                                append_emb = np.concatenate((seg_emb[:sliding_window], append_emb), axis=0)
                        if last_start > -ori_seq_len:
                            seg_idx += 1
                            remain = last_start - ori_seq_len
                            seg_seq = seq[-ori_seq_len:-ori_seq_len + 2 * sliding_window]
                            seg_emb, seg_processed_seq_len = predict_embedding_luca(llm_dirpath,
                                                                                    [seq_id + "_seg_%d" % seg_idx, seq_type, seg_seq],
                                                                                    trunc_type,
                                                                                    embedding_type,
                                                                                    repr_layers=[-1],
                                                                                    truncation_seq_length=truncation_seq_length,
                                                                                    matrix_add_special_token=False,
                                                                                    device=device)
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
                        seg_emb, seg_processed_seq_len = predict_embedding_luca(llm_dirpath,
                                                                                [seq_id + "_seg_%d" % seg_idx, seq_type, seg_seq],
                                                                                trunc_type,
                                                                                embedding_type,
                                                                                repr_layers=[-1],
                                                                                truncation_seq_length=truncation_seq_length,
                                                                                matrix_add_special_token=False,
                                                                                device=device)

                        if append_emb is None:
                            append_emb = seg_emb
                        else:
                            if trunc_type == "right":
                                append_emb = np.concatenate((append_emb, seg_emb), axis=0)
                            else:
                                append_emb = np.concatenate((seg_emb, append_emb), axis=0)

                    if trunc_type == "right":
                        # 处理最后一个
                        last_seg_seq = seq[-cur_segment_len:]
                        really_len = (ori_seq_len - (segment_num - 1) * cur_segment_len)
                        # print("last seg seq: %s" % last_seg_seq)
                        print("last seg seq len: %d, really len: %d" % (len(last_seg_seq), really_len))
                        last_seg_emb, last_seg_processed_seq_len = predict_embedding_luca(llm_dirpath,
                                                                                          [seq_id + "_seg_%d" % (segment_num - 1), seq_type, last_seg_seq],
                                                                                          trunc_type,
                                                                                          embedding_type,
                                                                                          repr_layers=[-1],
                                                                                          truncation_seq_length=truncation_seq_length,
                                                                                          matrix_add_special_token=False,
                                                                                          device=device)
                        last_seg_emb = last_seg_emb[-really_len:, :]
                        append_emb = np.concatenate((append_emb, last_seg_emb), axis=0)
                    else:
                        # 处理第一个
                        first_seg_seq = seq[:cur_segment_len]
                        really_len = (ori_seq_len - (segment_num - 1) * cur_segment_len)
                        # print("first seg seq: %s" % first_seg_seq)
                        print("first seg seq len: %d, really len: %d" % (len(first_seg_seq), really_len))
                        first_seg_emb, first_seg_processed_seq_len = predict_embedding_luca(llm_dirpath,
                                                                                            [seq_id + "_seg_0", seq_type, first_seg_seq],
                                                                                            trunc_type,
                                                                                            embedding_type,
                                                                                            repr_layers=[-1],
                                                                                            truncation_seq_length=truncation_seq_length,
                                                                                            matrix_add_special_token=False,
                                                                                            device=device)
                        first_seg_emb = first_seg_emb[:really_len, :]
                        append_emb = np.concatenate((first_seg_emb, append_emb), axis=0)
                except Exception as e:
                    append_emb = None
                if append_emb is not None:
                    break
                print("fail, change segment len: %d -> %d, change seg num: %d -> %d" % (cur_segment_len, int(cur_segment_len * 0.95), segment_num, int((ori_seq_len + cur_segment_len - 1) / cur_segment_len)))
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
    def __init__(self,
                 llm_type,
                 llm_dirpath,
                 input_type,
                 trunc_type,
                 seq_max_length,
                 atom_seq_max_length=None,
                 prepend_bos=True,
                 append_eos=True,
                 vector_dirpath=None,
                 matrix_dirpath=None,
                 local_rank=-1,
                 **kwargs):
        self.llm_type = llm_type
        self.llm_dirpath = llm_dirpath
        self.input_type = input_type
        self.trunc_type = trunc_type
        self.seq_max_length = seq_max_length
        self.atom_seq_max_length = atom_seq_max_length
        self.vector_dirpath = vector_dirpath
        self.matrix_dirpath = matrix_dirpath
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        self.alphabet = Alphabet.from_predefined("gene_prot")

        self.matrix_add_special_token = False
        if "matrix_add_special_token" in kwargs and kwargs["matrix_add_special_token"]:
            self.matrix_add_special_token = kwargs["matrix_add_special_token"]

        print("Encoder: prepend_bos=", self.prepend_bos, ",append_eos=", self.append_eos)
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

        if local_rank == -1 and torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.cuda.is_available() and local_rank > -1:
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        self.device = device
        self.seq_id_2_emb_filename = {}
        print("Encoder: prepend_bos=", self.prepend_bos, ",append_eos=", self.append_eos)

    def encode_single(self,
                      seq_id,
                      seq_type,
                      seq,
                      vector_filename=None,
                      matrix_filename=None,
                      label=None):
        vector = None
        if self.input_type in ["vector", "seq_vector"]:
            if isinstance(vector_filename, str):
                vector = torch.load(os.path.join(self.vector_dirpath, vector_filename))
            elif vector_filename is None:
                if seq is None:
                    raise Exception("seq is none and vector_filename is none")
                elif seq_type == "molecule":
                    raise Exception("now not support embedding of the seq_type=%s" % seq_type)
                else:
                    vector = self.__get_embedding__(seq_id, seq_type, seq, "vector")
            elif isinstance(vector_filename, np.ndarray):
                vector = vector_filename
            else:
                raise Exception("vector is not filepath-str and np.ndarray")
        matrix = None
        if self.input_type in ["matrix", "seq_matrix"]:
            if isinstance(matrix_filename, str):
                matrix = torch.load(os.path.join(self.matrix_dirpath, matrix_filename))
            elif matrix_filename is None:
                if seq is None:
                    raise Exception("seq is none and matrix_filename is none")
                elif seq_type == "molecule":
                    raise Exception("now not support embedding of the seq_type=%s" % seq_type)
                else:
                    matrix = self.__get_embedding__(seq_id, seq_type, seq, "matrix")
            elif isinstance(matrix_filename, np.ndarray):
                matrix = matrix_filename
            else:
                raise Exception("matrix is not filepath-str and np.ndarray")
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
                seq = clean_seq(seq_id, seq)
            elif seq_type == "multi_prot":
                seq = ",".join([clean_seq(seq_id, v) for v in seq.split(",")])
        return {
            "seq_id": seq_id,
            "seq": seq,
            "seq_type": seq_type,
            "vector": vector,
            "matrix": matrix,
            "label": label
        }

    def __get_embedding__(self, seq_id, seq_type, seq, embedding_type):
        embedding_info = None
        if seq_id in self.seq_id_2_emb_filename:
            emb_filename = self.seq_id_2_emb_filename[seq_id]
            try:
                embedding_info = torch.load(os.path.join(self.vector_dirpath if embedding_type in ["bos", "vector"] else self.matrix_dirpath, emb_filename))
                return embedding_info
            except Exception as e:
                print(e)
                embedding_info = None
        elif embedding_type in ["bos", "vector"] and self.vector_dirpath is not None or embedding_type not in ["bos", "vector"] and self.matrix_dirpath is not None:
            if "|" in seq_id:
                strs = seq_id.split("|")
                if len(strs) > 3:
                    emb_filename = embedding_type + "_" + strs[1].strip() + ".pt"
                else:
                    emb_filename = embedding_type + "_" + seq_id.replace(" ", "").replace("/", "_") + ".pt"
            else:
                emb_filename = embedding_type + "_" + seq_id.replace(" ", "").replace("/", "_") + ".pt"
            embedding_path = os.path.join(self.vector_dirpath if embedding_type in ["bos", "vector"] else self.matrix_dirpath, emb_filename)
            if os.path.exists(embedding_path):
                try:
                    embedding_info = torch.load(embedding_path)
                except Exception as e:
                    print(e)
                    embedding_info = None
        if embedding_info is None:
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
                            cur_embedding_info, cur_processed_seq = predict_embedding_esm([seq_id, cur_seq],
                                                                                          self.trunc_type,
                                                                                          embedding_type,
                                                                                          repr_layers=[-1],
                                                                                          truncation_seq_length=truncation_seq_length,
                                                                                          matrix_add_special_token=self.matrix_add_special_token,
                                                                                          device=self.device)
                            if cur_embedding_info is not None:
                                break
                            truncation_seq_length = (truncation_seq_length + int(self.prepend_bos) + int(self.append_eos)) * 0.9 - int(self.prepend_bos) - int(self.append_eos)
                            truncation_seq_length = int(truncation_seq_length)
                            print("truncation_seq_length: %d->%d" % (cur_seq_len, truncation_seq_length))
                        embedding_info.append(cur_embedding_info)
                else:
                    truncation_seq_length = self.seq_max_length - int(self.prepend_bos) - int(self.append_eos)
                    cur_seq_len = len(seq)
                    while True:
                        embedding_info, processed_seq = predict_embedding_esm([seq_id, seq],
                                                                              self.trunc_type,
                                                                              embedding_type,
                                                                              repr_layers=[-1],
                                                                              truncation_seq_length=truncation_seq_length,
                                                                              matrix_add_special_token=self.matrix_add_special_token,
                                                                              device=self.device)
                        if embedding_info is not None:
                            break
                        truncation_seq_length = (truncation_seq_length + int(self.prepend_bos) + int(self.append_eos)) * 0.9 - int(self.prepend_bos) - int(self.append_eos)
                        truncation_seq_length = int(truncation_seq_length)
                        print("truncation_seq_length: %d->%d" % (cur_seq_len, truncation_seq_length))
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
                            cur_embedding_info, cur_processed_seq = predict_embedding_dnaberts([seq_id, cur_seq],
                                                                                               self.trunc_type,
                                                                                               embedding_type,
                                                                                               repr_layers=[-1],
                                                                                               truncation_seq_length=truncation_seq_length,
                                                                                               matrix_add_special_token=self.matrix_add_special_token,
                                                                                               device=self.device)


                            if cur_embedding_info is not None:
                                break
                            truncation_seq_length = (truncation_seq_length + int(self.prepend_bos) + int(self.append_eos)) * 0.9 - int(self.prepend_bos) - int(self.append_eos)
                            truncation_seq_length = int(truncation_seq_length)
                            print("truncation_seq_length: %d->%d" % (cur_seq_len, truncation_seq_length))
                        embedding_info.append(cur_embedding_info)
                else:
                    cur_seq_len = len(seq)
                    truncation_seq_length = self.seq_max_length - int(self.prepend_bos) - int(self.append_eos)
                    while True:
                        embedding_info, processed_seq = predict_embedding_dnaberts([seq_id, seq],
                                                                                   self.trunc_type,
                                                                                   embedding_type,
                                                                                   repr_layers=[-1],
                                                                                   truncation_seq_length=truncation_seq_length,
                                                                                   matrix_add_special_token=self.matrix_add_special_token,
                                                                                   device=self.device)

                        if embedding_info is not None:
                            break
                        truncation_seq_length = (truncation_seq_length + int(self.prepend_bos) + int(self.append_eos)) * 0.9 - int(self.prepend_bos) - int(self.append_eos)
                        truncation_seq_length = int(truncation_seq_length)
                        print("truncation_seq_length: %d->%d" % (cur_seq_len, truncation_seq_length))
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
                            cur_embedding_info, cur_processed_seq = predict_embedding_dnabert2([seq_id, cur_seq],
                                                                                               self.trunc_type,
                                                                                               embedding_type,
                                                                                               repr_layers=[-1],
                                                                                               truncation_seq_length=truncation_seq_length,
                                                                                               matrix_add_special_token=self.matrix_add_special_token,
                                                                                               device=self.device)


                            if cur_embedding_info is not None:
                                break
                            truncation_seq_length = (truncation_seq_length + int(self.prepend_bos) + int(self.append_eos)) * 0.9 - int(self.prepend_bos) - int(self.append_eos)
                            truncation_seq_length = int(truncation_seq_length)
                            print("truncation_seq_length: %d->%d" % (cur_seq_len, truncation_seq_length))
                        embedding_info.append(cur_embedding_info)
                else:
                    cur_seq_len = len(seq)
                    truncation_seq_length = self.seq_max_length - int(self.prepend_bos) - int(self.append_eos)
                    while True:
                        embedding_info, processed_seq = predict_embedding_dnabert2([seq_id, seq],
                                                                                   self.trunc_type,
                                                                                   embedding_type,
                                                                                   repr_layers=[-1],
                                                                                   truncation_seq_length=truncation_seq_length,
                                                                                   matrix_add_special_token=self.matrix_add_special_token,
                                                                                   device=self.device)

                        if embedding_info is not None:
                            break
                        truncation_seq_length = (truncation_seq_length + int(self.prepend_bos) + int(self.append_eos)) * 0.9 - int(self.prepend_bos) - int(self.append_eos)
                        truncation_seq_length = int(truncation_seq_length)
                        print("truncation_seq_length: %d->%d" % (cur_seq_len, truncation_seq_length))
            elif "molecule" in seq_type:
                # to do
                pass
            elif "luca_separated" in self.llm_type:
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
                            truncation_seq_length = min(cur_seq_len, 10000)
                        else:
                            truncation_seq_length = self.seq_max_length - int(self.prepend_bos) - int(self.append_eos)
                            truncation_seq_length = min(cur_seq_len, truncation_seq_length)
                        cur_seq_type = "gene" if seq_type == "multi_gene" else "prot"
                        while True:
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
                            if cur_embedding_info is not None:
                                cur_embedding_info = complete_embedding_matrix(seq_id,
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
                                                                               self.device)
                                break

                            truncation_seq_length = (truncation_seq_length + int(self.prepend_bos) + int(self.append_eos)) * 0.9 - int(self.prepend_bos) - int(self.append_eos)
                            truncation_seq_length = int(truncation_seq_length)
                            print("truncation_seq_length: %d->%s" % (cur_seq_len, truncation_seq_length))
                        embedding_info.append(cur_embedding_info)
                else:
                    cur_seq_len = len(seq)
                    if hasattr(self, "embedding_complete") and self.embedding_complete:
                        truncation_seq_length = min(cur_seq_len, 10000)
                    else:
                        # to do
                        truncation_seq_length = self.seq_max_length - int(self.prepend_bos) - int(self.append_eos)
                        truncation_seq_length = min(cur_seq_len, truncation_seq_length)
                    while True:
                        embedding_info, processed_seq = predict_embedding_luca(self.llm_dirpath,
                                                                               [seq_id, seq_type, seq],
                                                                               self.trunc_type,
                                                                               embedding_type,
                                                                               repr_layers=[-1],
                                                                               truncation_seq_length=truncation_seq_length,
                                                                               matrix_add_special_token=self.matrix_add_special_token,
                                                                               device=self.device)
                        if embedding_info is not None:
                            if hasattr(self, "embedding_complete") and self.embedding_complete:
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
                                    self.device
                                )

                            break
                        truncation_seq_length = (truncation_seq_length + int(self.prepend_bos) + int(self.append_eos)) * 0.9 - int(self.prepend_bos) - int(self.append_eos)
                        truncation_seq_length = int(truncation_seq_length)
                        print("truncation_seq_length: %d->%d" % (cur_seq_len, truncation_seq_length))
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
                            truncation_seq_length = min(cur_seq_len, 10000)
                        else:
                            truncation_seq_length = self.seq_max_length - int(self.prepend_bos) - int(self.append_eos)
                            truncation_seq_length = min(cur_seq_len, truncation_seq_length)
                        cur_seq_type = "gene" if seq_type == "multi_gene" else "prot"
                        while True:
                            cur_embedding_info, cur_processed_seq = predict_embedding_luca(self.llm_dirpath,
                                                                                           [seq_id, cur_seq_type, cur_seq],
                                                                                           self.trunc_type,
                                                                                           embedding_type,
                                                                                           repr_layers=[-1],
                                                                                           truncation_seq_length=truncation_seq_length,
                                                                                           matrix_add_special_token=self.matrix_add_special_token,
                                                                                           device=self.device)
                            if cur_embedding_info is not None:
                                cur_embedding_info = complete_embedding_matrix(seq_id, cur_seq_type, cur_seq, truncation_seq_length, cur_embedding_info,
                                                                               self.llm_dirpath, self.trunc_type, embedding_type, self.matrix_add_special_token, self.embedding_complete, self.embedding_complete_seg_overlap, self.device)
                                break

                            truncation_seq_length = (truncation_seq_length + int(self.prepend_bos) + int(self.append_eos)) * 0.9 - int(self.prepend_bos) - int(self.append_eos)
                            truncation_seq_length = int(truncation_seq_length)
                            print("truncation_seq_length: %d->%d" % (cur_seq_len, truncation_seq_length))
                        embedding_info.append(cur_embedding_info)
                else:
                    cur_seq_len = len(seq)
                    if hasattr(self, "embedding_complete") and self.embedding_complete:
                        truncation_seq_length = min(cur_seq_len, 10000)
                    else:
                        truncation_seq_length = self.seq_max_length - int(self.prepend_bos) - int(self.append_eos)
                        truncation_seq_length = min(cur_seq_len, truncation_seq_length)

                    while True:
                        embedding_info, processed_seq = predict_embedding_luca(self.llm_dirpath,
                                                                               [seq_id, seq_type, seq],
                                                                               self.trunc_type,
                                                                               embedding_type,
                                                                               repr_layers=[-1],
                                                                               truncation_seq_length=truncation_seq_length,
                                                                               matrix_add_special_token=self.matrix_add_special_token,
                                                                               device=self.device)
                        if embedding_info is not None:
                            if hasattr(self, "embedding_complete") and self.embedding_complete:
                                embedding_info = complete_embedding_matrix(seq_id, seq_type, seq, truncation_seq_length, embedding_info,
                                                                           self.llm_dirpath, self.trunc_type, embedding_type, self.matrix_add_special_token, self.embedding_complete, self.embedding_complete_seg_overlap, self.device)

                            break
                        truncation_seq_length = (truncation_seq_length + int(self.prepend_bos) + int(self.append_eos)) * 0.9 - int(self.prepend_bos) - int(self.append_eos)
                        truncation_seq_length = int(truncation_seq_length)
                        print("truncation_seq_length: %d->%d" % (cur_seq_len, truncation_seq_length))
            if embedding_type in ["bos", "vector"] and self.vector_dirpath is not None or embedding_type not in ["bos", "vector"] and self.matrix_dirpath is not None:
                if "|" in seq_id:
                    strs = seq_id.split("|")
                    if len(strs) > 3:
                        emb_filename = embedding_type + "_" + strs[1].strip() + ".pt"
                    else:
                        emb_filename = embedding_type + "_" + seq_id.replace(" ", "").replace("/", "_") + ".pt"
                else:
                    emb_filename = embedding_type + "_" + seq_id.replace(" ", "").replace("/", "_") + ".pt"
                # emb_filename = embedding_type + "_" + seq_id.replace(" ", "").replace("/", "_") + ".pt"
                embedding_filepath = os.path.join(self.vector_dirpath if embedding_type in ["bos", "vector"] else self.matrix_dirpath, emb_filename)
                # print("seq_len: %d" % len(seq))
                # print("emb shape:", embedding_info.shape)
                torch.save(embedding_info, embedding_filepath)
                self.seq_id_2_emb_filename[seq_id] = emb_filename
        return embedding_info

    def encode_pair(self,
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
                    label=None
                    ):

        vector_a, vector_b = None, None
        if self.input_type in ["vector", "seq_vector"]:
            if vector_filename_a is None:
                if seq_a is None:
                    raise Exception("seq_a is none and vector_filename_a is none")
                elif seq_type_a == "molecule":
                    raise Exception("now not support embedding of the seq_type_a=%s" % seq_type_a)
                else:
                    vector_a = self.__get_embedding__(seq_id_a, seq_type_a, seq_a, "vector")
            elif isinstance(vector_filename_a, str):
                vector_a = torch.load(os.path.join(self.vector_dirpath, vector_filename_a))
            else:
                vector_a = vector_filename_a
            if vector_filename_b is None:
                if seq_b is None:
                    raise Exception("seq_b is none and vector_filename_b is none")
                elif seq_type_b == "molecule":
                    raise Exception("now not support embedding of the seq_type_b=%s" % seq_type_b)
                else:
                    vector_b = self.__get_embedding__(seq_id_b, seq_type_b, seq_b, "vector")
            elif isinstance(vector_filename_b, str):
                vector_b = torch.load(os.path.join(self.vector_dirpath, vector_filename_b))
            else:
                vector_b = vector_filename_b
        matrix_a, matrix_b = None, None
        if self.input_type in ["matrix", "seq_matrix"]:
            if matrix_filename_a is None:
                if seq_a is None:
                    raise Exception("seq_a is none and matrix_filename_a is none")
                else:
                    matrix_a = self.__get_embedding__(seq_id_a, seq_type_a, seq_a, "matrix")
            elif isinstance(matrix_filename_a, str):
                matrix_a = torch.load(os.path.join(self.matrix_dirpath, matrix_filename_a))
            else:
                matrix_a = matrix_filename_a
            if matrix_filename_b is None:
                if seq_b is None:
                    raise Exception("seq_b is none and matrix_filename_b is none")
                else:
                    matrix_b = self.__get_embedding__(seq_id_b, seq_type_b, seq_b, "matrix")
            elif isinstance(matrix_filename_b, str):
                matrix_b = torch.load(os.path.join(self.matrix_dirpath, matrix_filename_b))
            else:
                matrix_b = matrix_filename_b
        if seq_type_a == "molecule":
            # to do
            pass
        else:
            seq_a = seq_a.upper()
        if seq_type_b == "molecule":
            # to do
            pass
        else:
            seq_b = seq_b.upper()
        # 蛋白质且使用esm进行embedding，则需要去掉蛋白质J
        if self.input_type in ["matrix", "seq_matrix"] and "esm" in self.llm_type:
            if seq_type_a == "prot":
                seq_a = clean_seq(seq_id_a, seq_a)
            elif seq_type_a == "multi_prot":
                seq_a = ",".join([clean_seq(seq_id_a, v) for v in seq_a.split(",")])

            if seq_type_b == "prot":
                seq_b = clean_seq(seq_id_b, seq_b)
            elif seq_type_a == "multi_prot":
                seq_b = ",".join([clean_seq(seq_id_b, v) for v in seq_b.split(",")])

        return {
            "seq_id_a": seq_id_a,
            "seq_a": seq_a,
            "seq_type_a": seq_type_a,
            "vector_a": vector_a,
            "matrix_a": matrix_a,
            "seq_id_b": seq_id_b,
            "seq_b": seq_b,
            "seq_type_b": seq_type_b,
            "vector_b": vector_b,
            "matrix_b": matrix_b,
            "label": label
        }



