#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/7/24 15:14
@project: LucaOneTasks
@file: batch_converter
@desc: batch converter for LucaOneTasks
'''
import sys
import torch
from typing import Sequence
import random
sys.path.append(".")
sys.path.append("..")
sys.path.append("../src")
try:
    from .common.alphabet_atom import AlphabetAtom
    from .utils import gene_seq_replace
except ImportError:
    from src.common.alphabet_atom import AlphabetAtom
    from src.utils import gene_seq_replace


class BatchConverter(object):

    def __init__(
            self,
            task_level_type,
            label_size,
            output_mode,
            seq_subword,
            seq_tokenizer,
            no_position_embeddings,
            no_token_type_embeddings,
            truncation_seq_length: int = None,
            truncation_matrix_length: int = None,
            atom_tokenizer: AlphabetAtom = None,
            atom_truncation_seq_length: int = None,
            atom_truncation_matrix_length: int = None,
            ignore_index: int = -100,
            padding_idx: int = 0,
            unk_idx: int = 1,
            cls_idx: int = 2,
            eos_idx: int = 3,
            mask_idx: int = 4,
            non_ignore: bool = False,
            mlm_probability=0.15,
            prepend_bos=None,
            append_eos=None,
            **kwargs
    ):
        print("------BatchConverter------")
        print("BatchConverter, kwargs:")
        print(kwargs)
        self.task_level_type = task_level_type
        self.label_size = label_size
        self.output_mode = output_mode
        self.seq_tokenizer = seq_tokenizer
        self.seq_subword = seq_subword
        self.ignore_index = ignore_index
        self.non_ignore = non_ignore
        self.mlm_probability = mlm_probability
        self.truncation_seq_length = truncation_seq_length
        self.truncation_matrix_length = truncation_matrix_length

        # subword 则必包含两个特殊字符
        if prepend_bos is None:
            if seq_subword is not None:
                self.prepend_bos = True
            else:
                self.prepend_bos = False
        else:
            self.prepend_bos = prepend_bos
        if append_eos is None:
            if seq_subword is not None:
                self.append_eos = True
            else:
                self.append_eos = False
        else:
            self.append_eos = append_eos

        self.padding_idx = padding_idx
        self.unk_idx = unk_idx
        self.cls_idx = cls_idx
        self.eos_idx = eos_idx
        self.mask_idx = mask_idx
        if self.seq_tokenizer is None:
            self.append_len = 0
        else:
            if hasattr(seq_tokenizer, "prepend_bos"):
                self.prepend_bos = self.seq_tokenizer.prepend_bos
            if hasattr(seq_tokenizer, "append_eos"):
                self.append_eos = self.seq_tokenizer.append_eos
            if hasattr(seq_tokenizer, "padding_idx"):
                self.padding_idx = self.seq_tokenizer.padding_idx
            if hasattr(seq_tokenizer, "unk_idx"):
                self.unk_idx = self.seq_tokenizer.unk_idx
            if hasattr(seq_tokenizer, "cls_idx"):
                self.cls_idx = self.seq_tokenizer.cls_idx
            if hasattr(seq_tokenizer, "eos_idx"):
                self.eos_idx = self.seq_tokenizer.eos_idx
            if hasattr(seq_tokenizer, "mask_idx"):
                self.mask_idx = self.seq_tokenizer.mask_idx
            if hasattr(seq_tokenizer, "all_special_token_idx_list"):
                self.all_special_token_idx_list = self.seq_tokenizer.all_special_token_idx_list
            else:
                self.all_special_token_idx_list = [self.padding_idx, self.unk_idx, self.cls_idx, self.eos_idx, self.mask_idx]
            self.append_len = int(self.prepend_bos) + int(self.append_eos)

        # for atom
        self.atom_tokenizer = atom_tokenizer
        self.atom_truncation_seq_length = atom_truncation_seq_length
        self.atom_truncation_matrix_length = atom_truncation_matrix_length
        self.atom_prepend_bos = False
        self.atom_append_eos = False
        self.atom_padding_idx = padding_idx
        self.atom_unk_idx = unk_idx
        self.atom_cls_idx = cls_idx
        self.atom_eos_idx = eos_idx
        self.atom_mask_idx = mask_idx
        if self.atom_tokenizer is None:
            self.atom_append_len = 0
        else:
            if hasattr(atom_tokenizer, "padding_idx"):
                self.atom_padding_idx = self.atom_tokenizer.padding_idx
            elif hasattr(atom_tokenizer, "pad_idx"):
                self.atom_padding_idx = self.atom_tokenizer.pad_idx
            elif hasattr(atom_tokenizer, "pad_token_id"):
                self.atom_padding_idx = self.atom_tokenizer.pad_token_id

            if hasattr(atom_tokenizer, "unk_idx"):
                self.atom_unk_idx = self.atom_tokenizer.unk_idx
            elif hasattr(atom_tokenizer, "unk_token_id"):
                self.atom_unk_idx = self.atom_tokenizer.unk_token_id

            if hasattr(atom_tokenizer, "cls_idx"):
                self.atom_cls_idx = self.atom_tokenizer.cls_idx
            elif hasattr(atom_tokenizer, "cls_token_id"):
                self.atom_cls_idx = self.atom_tokenizer.cls_token_id
            elif hasattr(atom_tokenizer, "bos_idx"):
                self.atom_cls_idx = self.atom_tokenizer.bos_idx
            elif hasattr(atom_tokenizer, "bos_token_id"):
                self.atom_cls_idx = self.atom_tokenizer.bos_token_id

            if hasattr(atom_tokenizer, "eos_idx"):
                self.atom_eos_idx = self.atom_tokenizer.eos_idx
            elif hasattr(atom_tokenizer, "eos_token_id"):
                self.atom_eos_idx = self.atom_tokenizer.eos_token_id
            elif hasattr(atom_tokenizer, "sep_token_id"):
                self.atom_eos_idx = self.atom_tokenizer.sep_token_id

            if hasattr(atom_tokenizer, "mask_idx"):
                self.atom_mask_idx = self.atom_tokenizer.mask_idx
            elif hasattr(atom_tokenizer, "mask_token_id"):
                self.atom_mask_idx = self.atom_tokenizer.mask_token_id

            if hasattr(atom_tokenizer, "all_special_token_idx_list"):
                self.atom_all_special_token_idx_list = self.atom_tokenizer.all_special_token_idx_list
            else:
                self.atom_all_special_token_idx_list = [self.atom_padding_idx, self.atom_unk_idx, self.atom_cls_idx, self.atom_eos_idx, self.atom_mask_idx]
            self.atom_append_len = int(self.atom_prepend_bos) + int(self.atom_append_eos)

        print("BatchConverter: prepend_bos=%r, append_eos=%r" % (self.prepend_bos, self.append_eos))
        print("BatchConverter: atom_prepend_bos=%r, atom_append_eos=%r" % (self.atom_prepend_bos, self.atom_append_eos))
        self.matrix_add_special_token = False
        if "matrix_add_special_token" in kwargs and kwargs["matrix_add_special_token"]:
            self.matrix_add_special_token = kwargs["matrix_add_special_token"]
        if self.matrix_add_special_token:
            self.prepend_bos = True
            self.append_eos = True
            self.atom_prepend_bos = True
            self.atom_append_eos = True
            self.append_len = int(self.prepend_bos) + int(self.append_eos)
            self.atom_append_len = int(self.atom_prepend_bos) + int(self.atom_append_eos)

        # 减去特殊字符之后的长度
        self.truncation_seq_length -= self.append_len
        self.truncation_matrix_length -= self.append_len
        # 减去特殊字符之后的长度
        if self.atom_truncation_seq_length:
            self.atom_truncation_seq_length -= self.atom_append_len
        if self.atom_truncation_matrix_length:
            self.atom_truncation_matrix_length -= self.atom_append_len

        self.input_type = None
        if "input_type" in kwargs and kwargs["input_type"]:
            self.input_type = kwargs["input_type"]

        if "max_sentence_length" in kwargs and kwargs["max_sentence_length"]:
            self.max_sentence_length = kwargs["max_sentence_length"] - self.append_len
            print("BatchConverter: self.max_sentence_length=%d" % self.max_sentence_length)
            if atom_tokenizer is not None:
                self.atom_max_sentence_length = kwargs["max_sentence_length"] - self.atom_append_len
                print("BatchConverter: self.atom_max_sentence_length=%d" % self.atom_max_sentence_length)
        if "max_sentences" in kwargs and kwargs["max_sentences"]:
            self.max_sentences = kwargs["max_sentences"]
            print("BatchConverter: self.max_sentences=%d" % self.max_sentences)
            if atom_tokenizer is not None:
                self.atom_max_sentences = kwargs["max_sentences"]
                print("BatchConverter: self.atom_max_sentences=%d" % self.atom_max_sentences)
        self.trunc_type = "right"
        if "trunc_type" in kwargs and kwargs["trunc_type"]:
            self.trunc_type = kwargs["trunc_type"]
            print("BatchConverter: self.trunc_type=%s" % self.trunc_type)

        self.no_position_embeddings = no_position_embeddings
        self.no_token_type_embeddings = no_token_type_embeddings
        print("BatchConverter: prepend_bos=%r, append_eos=%r" % (self.prepend_bos, self.append_eos))
        print("BatchConverter: atom_prepend_bos=%r, atom_append_eos=%r" % (self.atom_prepend_bos, self.atom_append_eos))
        print("-" * 50)

    def __parse_label__(self, max_length, task_level_type, label_size, output_mode, label):
        if isinstance(label, str):
            label = eval(label)
        '''
        print("label:")
        print(label)
        '''
        # 需要是padding长度
        cur_len = max_length
        if task_level_type in ["token_level", "structure_level"]:
            if output_mode in ["multi_label", "multi-label"]:
                # N * seq_len * label_size
                new_label = []
                for _ in range(cur_len):
                    tmp = []
                    for _ in range(label_size):
                        tmp.append(0 if self.non_ignore else self.ignore_index)
                    new_label.append(tmp)
            else:
                # N * seq_len
                new_label = []
                for _ in range(cur_len):
                    new_label.append(0 if self.non_ignore else self.ignore_index)
            if label is not None and len(label) > 0:
                begin_idx = 0
                end_idx = cur_len
                if self.prepend_bos:
                    begin_idx = 1
                if self.append_eos:
                    end_idx = cur_len - 1
                for idx, item in enumerate(label):
                    idx += begin_idx
                    if idx >= end_idx:
                        break
                    if output_mode in ["multi_label", "multi-label"]:
                        for v in item:
                            new_label[idx][v] = 1
                    else:
                        new_label[idx] = item
        elif task_level_type == "span_level":
            if output_mode in ["multi_label", "multi-label"]:
                # N * seq_len * label_size
                new_label = []
                for _ in range(cur_len):
                    tmp = []
                    for _ in range(label_size):
                        tmp.append(0 if self.non_ignore else self.ignore_index)
                    new_label.append(tmp)
            else:
                # N * seq_len
                new_label = []
                for _ in range(cur_len):
                    new_label.append(0 if self.non_ignore else self.ignore_index)
            if label is not None and len(label) > 0:
                begin_idx = 0
                end_idx = cur_len
                if self.prepend_bos:
                    begin_idx = 1
                if self.append_eos:
                    end_idx = cur_len - 1
                for item in label:
                    for idx in range(item[0], item[1] + 1, 1):
                        idx += begin_idx
                        if idx >= end_idx:
                            break
                        if output_mode in ["multi_label", "multi-label"]:
                            new_label[idx][item[2]] = 1
                        else:
                            new_label[idx] = item[2]
        elif task_level_type in ["seq_level"]:
            if output_mode in ["multi_label", "multi-label"]:
                # N * label_size
                new_label = []
                for _ in range(label_size):
                    new_label.append(0 if self.non_ignore else self.ignore_index)
            else:
                # N * 1
                new_label = [0 if self.non_ignore else self.ignore_index]
            if output_mode in ["multi_label", "multi-label"]:
                if label is not None and len(label) > 0:
                    for v in label:
                        new_label[int(v)] = 1
            else:
                if label is not None and len(str(label)) > 0:
                    if isinstance(label, str):
                        new_label = [int(label)]
                    elif isinstance(label, list):
                        new_label = [int(label[0])]
                    else:
                        new_label = [label]
        else:
            raise Exception("Not support task_level_type=%s" % task_level_type)
        return new_label

    def __atom_parse_label__(self, max_length, task_level_type, label_size, output_mode, label):
        if isinstance(label, str):
            label = eval(label)
        '''
        print("label:")
        print(label)
        '''
        # 需要是padding长度
        cur_len = max_length
        if task_level_type in ["token_level", "structure_level"]:
            if output_mode in ["multi_label", "multi-label"]:
                # N * seq_len * label_size
                new_label = []
                for _ in range(cur_len):
                    tmp = []
                    for _ in range(label_size):
                        tmp.append(0 if self.non_ignore else self.ignore_index)
                    new_label.append(tmp)
            else:
                # N * seq_len
                new_label = []
                for _ in range(cur_len):
                    new_label.append(0 if self.non_ignore else self.ignore_index)
            if label is not None and len(label) > 0:
                begin_idx = 0
                end_idx = cur_len
                if self.atom_prepend_bos:
                    begin_idx = 1
                if self.atom_append_eos:
                    end_idx = cur_len - 1
                for idx, item in enumerate(label):
                    idx += begin_idx
                    if idx >= end_idx:
                        break
                    if output_mode in ["multi_label", "multi-label"]:
                        for v in item:
                            new_label[idx][v] = 1
                    else:
                        new_label[idx] = item
        elif task_level_type == "span_level":
            if output_mode in ["multi_label", "multi-label"]:
                # N * seq_len * label_size
                new_label = []
                for _ in range(cur_len):
                    tmp = []
                    for _ in range(label_size):
                        tmp.append(0 if self.non_ignore else self.ignore_index)
                    new_label.append(tmp)
            else:
                # N * seq_len
                new_label = []
                for _ in range(cur_len):
                    new_label.append(0 if self.non_ignore else self.ignore_index)
            if label is not None and len(label) > 0:
                begin_idx = 0
                end_idx = cur_len
                if self.atom_prepend_bos:
                    begin_idx = 1
                if self.atom_append_eos:
                    end_idx = cur_len - 1
                for item in label:
                    for idx in range(item[0], item[1] + 1, 1):
                        idx += begin_idx
                        if idx >= end_idx:
                            break
                        if output_mode in ["multi_label", "multi-label"]:
                            new_label[idx][item[2]] = 1
                        else:
                            new_label[idx] = item[2]
        elif task_level_type in ["seq_level"]:
            if output_mode in ["multi_label", "multi-label"]:
                # N * label_size
                new_label = []
                for _ in range(label_size):
                    new_label.append(0 if self.non_ignore else self.ignore_index)
            else:
                # N * 1
                new_label = [0 if self.non_ignore else self.ignore_index]
            if output_mode in ["multi_label", "multi-label"]:
                if label is not None and len(label) > 0:
                    for v in label:
                        new_label[int(v)] = 1
            else:
                if label is not None and len(str(label)) > 0:
                    if isinstance(label, str):
                        new_label = [int(label)]
                    elif isinstance(label, list):
                        new_label = [int(label[0])]
                    else:
                        new_label = [label]
        else:
            raise Exception("Not support task_level_type=%s" % task_level_type)

        return new_label

    def __mask_tokens__(self, input_ids, seq_len):
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        # 特殊字符处为1
        special_tokens_mask = [
            1 if v in self.all_special_token_idx_list else 0 for v in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        # 将特殊字符处填充为0.0
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # 非特殊字符的位置
        masked_indices = torch.bernoulli(probability_matrix).bool()
        # 特殊字符处为-100
        labels[~masked_indices] = self.ignore_index  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with alphabet.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_idx

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.seq_tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        if torch.any(labels != self.ignore_index):
            return input_ids, labels
        else:
            # non [MASK]， random one position, convect to [MASK]
            rand_idx = random.randint(int(self.prepend_bos), seq_len + int(self.prepend_bos) - 1)
            labels[rand_idx] = input_ids[rand_idx]
            input_ids[rand_idx] = self.mask_idx
            return input_ids, labels

    def __atom_mask_tokens__(self, input_ids, seq_len):
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        # 特殊字符处为1
        special_tokens_mask = [
            1 if v in self.atom_all_special_token_idx_list else 0 for v in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        # 将特殊字符处填充为0.0
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # 非特殊字符的位置
        masked_indices = torch.bernoulli(probability_matrix).bool()
        # 特殊字符处为-100
        labels[~masked_indices] = self.ignore_index  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with alphabet.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.atom_mask_idx

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.atom_tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        if torch.any(labels != self.ignore_index):
            return input_ids, labels
        else:
            # non [MASK]， random one position, convect to [MASK]
            rand_idx = random.randint(int(self.atom_prepend_bos), seq_len + int(self.atom_prepend_bos) - 1)
            labels[rand_idx] = input_ids[rand_idx]
            input_ids[rand_idx] = self.atom_mask_idx
            return input_ids, labels

    def __seq_encode__(self, batch_size, seq_types, seqs):
        '''
        该函数不加特殊字符[CLS]与[SEP]
        :param batch_size:
        :param seq_types:
        :param seqs:
        :return:
        '''
        if self.seq_subword:
            seq_encoded_list = []
            for seq_str in seqs:
                seq_to_list = self.seq_subword.process_line(seq_str.upper()).split(" ")
                seq = " ".join(seq_to_list)
                inputs = self.seq_tokenizer.encode_plus(
                    seq,
                    None,
                    add_special_tokens=False,
                    max_length=self.truncation_seq_length,
                    truncation=True
                )
                seq_encoded_list.append(inputs["input_ids"])
        else:
            seq_encoded_list = [self.seq_tokenizer.encode(seq_type=seq_type, seq=seq_str.upper()) for seq_type, seq_str in zip(seq_types, seqs)]
            # 该长度已经减去了需要增加的特殊字符的个数
            if self.truncation_seq_length:
                seq_encoded_list = [encoded[:self.truncation_seq_length] for encoded in seq_encoded_list]
        max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
        max_len = max_len + int(self.prepend_bos) + int(self.append_eos)
        # for input
        input_ids = torch.empty(
            (
                batch_size,
                max_len,
            ),
            dtype=torch.int64,
        )
        input_ids.fill_(self.padding_idx)

        position_ids = None
        if not self.no_position_embeddings:
            position_ids = torch.empty(
                (
                    batch_size,
                    max_len,
                ),
                dtype=torch.int64,
            )
            position_ids.fill_(self.padding_idx)

        token_type_ids = None
        if not self.no_position_embeddings:
            token_type_ids = torch.empty(
                (
                    batch_size,
                    max_len,
                ),
                dtype=torch.int64,
            )
            token_type_ids.fill_(self.padding_idx)
        attention_masks = torch.empty(
            (
                batch_size,
                max_len,
            ),
            dtype=torch.int64,
        )
        attention_masks.fill_(0)

        return seq_encoded_list, input_ids, position_ids, token_type_ids, attention_masks, max_len

    def __express_encode__(self, batch_size, express_list):
        '''
        该函数不加特殊字符[CLS]与[SEP]
        :param batch_size:
        :param express_list:
        :return:
        '''
        max_len = max(len(express) for express in express_list)
        if self.matrix_add_special_token:
            max_len -= 2
        if self.truncation_matrix_length:
            max_len = min(max_len, self.truncation_matrix_length)
        if self.matrix_add_special_token:
            max_len += 2
        else:
            max_len = max_len + int(self.prepend_bos) + int(self.append_eos)

        express_encoded_list = express_list
        # 该长度已经减去了需要增加的特殊字符的个数
        if self.truncation_matrix_length:
            express_encoded_list = [encoded[:self.truncation_matrix_length] for encoded in express_encoded_list]

        max_len = max_len + int(self.prepend_bos) + int(self.append_eos)
        # for input
        express_input_ids = torch.empty(
            (
                batch_size,
                max_len,
            ),
            dtype=torch.int64,
        )
        express_input_ids.fill_(self.padding_idx)
        for sample_idx in range(batch_size):
            real_len = len(express_encoded_list[sample_idx])
            cur_express_encoded = torch.tensor(express_encoded_list[sample_idx], dtype=torch.long)
            if self.prepend_bos:
                express_input_ids[sample_idx, 0] = self.cls_idx
            express_input_ids[sample_idx, int(self.prepend_bos):real_len + int(self.prepend_bos)] = cur_express_encoded
            cur_len = real_len + int(self.prepend_bos) + int(self.append_eos)
            if self.append_eos:
                express_input_ids[sample_idx, cur_len] = self.eos_idx

        return express_input_ids

    def __multi_seq_encode__(self, batch_size, seq_types, seqs):
        '''
        该函数是多sentence的表征器，每个sentence都加[CLS]与[SEP]
        :param batch_size:
        :param seq_types:
        :param seqs:
        :return:
        '''
        assert hasattr(self, "max_sentences") and hasattr(self, "max_sentence_length")
        max_sentence_len = 0
        max_sentence_num = 0
        if self.seq_subword:
            seq_encoded_list = []
            for cur_sample_seqs in seqs:
                cur_seq_encoded_list = []
                if len(cur_sample_seqs) > self.max_sentences:
                    # 每个样本最多cur_sample_seqs个
                    if self.trunc_type == "left":
                        cur_sample_seqs = cur_sample_seqs[-self.max_sentences:]
                    else:
                        cur_sample_seqs = cur_sample_seqs[:self.max_sentences]
                if max_sentence_num < len(cur_sample_seqs):
                    max_sentence_num = len(cur_sample_seqs)
                for seq_idx, seq_str in enumerate(cur_sample_seqs):
                    seq_to_list = self.seq_subword.process_line(seq_str.upper()).split(" ")
                    seq = " ".join(seq_to_list)
                    inputs = self.seq_tokenizer.encode_plus(
                        seq,
                        None,
                        add_special_tokens=False,
                        max_length=self.max_sentence_length,
                        truncation=True
                    )
                    if self.prepend_bos:
                        inputs["input_ids"] = [self.cls_idx] + inputs["input_ids"]
                    if self.append_eos:
                        inputs["input_ids"] = inputs["input_ids"] + [self.eos_idx]
                    if max_sentence_len < len(inputs["input_ids"]):
                        max_sentence_len = len(inputs["input_ids"])
                    cur_seq_encoded_list.append(inputs["input_ids"])
                seq_encoded_list.append(cur_seq_encoded_list)
        else:
            seq_encoded_list = []
            for cur_sample_idx, cur_sample_seqs in enumerate(seqs):
                cur_seq_encoded_list = []
                if len(cur_sample_seqs) > self.max_sentences:
                    # 每个样本最多cur_sample_seqs个
                    if self.trunc_type == "left":
                        cur_sample_seqs = cur_sample_seqs[-self.max_sentences:]
                    else:
                        cur_sample_seqs = cur_sample_seqs[:self.max_sentences]
                if max_sentence_num < len(cur_sample_seqs):
                    max_sentence_num = len(cur_sample_seqs)
                for seq_idx, seq_str in enumerate(cur_sample_seqs):
                    if len(seq_str) > self.max_sentence_length:
                        if self.trunc_type == "left":
                            seq_str = seq_str[-self.max_sentence_length:]
                        else:
                            seq_str = seq_str[:self.max_sentence_length]

                    inputs = self.seq_tokenizer.encode(seq_type=seq_types[cur_sample_idx], seq=seq_str.upper())
                    # print("len:%d, %s" % (len(seq_str), seq_str.upper()))
                    if self.prepend_bos:
                        inputs = [self.cls_idx] + inputs
                    if self.append_eos:
                        inputs = inputs + [self.eos_idx]
                    # print("inputs:%d, " %len(inputs), inputs)
                    cur_seq_encoded_list.append(inputs)
                    if max_sentence_len < len(inputs):
                        max_sentence_len = len(inputs)
            seq_encoded_list.append(cur_seq_encoded_list)
        # for input
        input_ids = torch.empty(
            (
                batch_size,
                max_sentence_num,
                max_sentence_len,
            ),
            dtype=torch.int64,
        )
        input_ids.fill_(self.padding_idx)

        position_ids = None
        if not self.no_position_embeddings:
            position_ids = torch.empty(
                (
                    batch_size,
                    max_sentence_num,
                    max_sentence_len
                ),
                dtype=torch.int64,
            )
            position_ids.fill_(self.padding_idx)

        token_type_ids = None
        if not self.no_position_embeddings:
            token_type_ids = torch.empty(
                (
                    batch_size,
                    max_sentence_num,
                    max_sentence_len
                ),
                dtype=torch.int64,
            )
            token_type_ids.fill_(self.padding_idx)
        attention_masks = torch.empty(
            (
                batch_size,
                max_sentence_num,
                max_sentence_len
            ),
            dtype=torch.int64,
        )
        attention_masks.fill_(0)

        return seq_encoded_list, input_ids, position_ids, token_type_ids, attention_masks, max_sentence_num, max_sentence_len

    def __atom_seq_encode__(self, batch_size, seq_types, seqs):
        '''
        该函数不加特殊字符[CLS]与[SEP]
        :param batch_size:
        :param seq_types:
        :param seqs:
        :return:
        '''
        seq_encoded_list = []
        for seq_idx, cur_seq in enumerate(seqs):
            if isinstance(cur_seq, str): # smiles
                cur_seq_encoded = self.atom_tokenizer.encode_smi(
                    seq_types[seq_idx],
                    cur_seq,
                    prepend_bos=False,
                    append_eos=False
                )
            elif isinstance(cur_seq, list): # atom list
                cur_seq_encoded = self.atom_tokenizer.encode(
                    seq_types[seq_idx],
                    cur_seq,
                    prepend_bos=False,
                    append_eos=False
                )
            else:
                raise Exception("not support molecule input type:", type(cur_seq))
            # 该长度已经减去了需要增加的特殊字符的个数
            if self.atom_truncation_seq_length:
                cur_seq_encoded = cur_seq_encoded[:self.atom_truncation_seq_length]
            seq_encoded_list.append(cur_seq_encoded)
        max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
        max_len = max_len + int(self.atom_prepend_bos) + int(self.atom_append_eos)
        # for input
        input_ids = torch.empty(
            (
                batch_size,
                max_len,
            ),
            dtype=torch.int64,
        )
        input_ids.fill_(self.atom_padding_idx)

        position_ids = None
        if not self.no_position_embeddings:
            position_ids = torch.empty(
                (
                    batch_size,
                    max_len,
                ),
                dtype=torch.int64,
            )
            position_ids.fill_(self.atom_padding_idx)

        token_type_ids = None
        if not self.no_position_embeddings:
            token_type_ids = torch.empty(
                (
                    batch_size,
                    max_len,
                ),
                dtype=torch.int64,
            )
            token_type_ids.fill_(self.atom_padding_idx)
        attention_masks = torch.empty(
            (
                batch_size,
                max_len,
            ),
            dtype=torch.int64,
        )
        attention_masks.fill_(0)

        return seq_encoded_list, input_ids, position_ids, token_type_ids, attention_masks, max_len

    def __vector_encode__(self, batch_size, vectors):
        embedding_vector_dim = vectors[0].shape[0]
        filled_vectors = torch.empty(
            (
                batch_size,
                embedding_vector_dim
            ),
            dtype=torch.float32,
        )
        filled_vectors.fill_(0.0)
        return filled_vectors, 1

    def __atom_vector_encode__(self, batch_size, vectors):
        return self.__vector_encode__(batch_size, vectors)

    def __multi_vector_encode__(self, batch_size, vectors):
        embedding_vector_dim = vectors[0][0].shape[0]
        filled_vectors = torch.empty(
            (
                batch_size,
                self.max_sentences,
                embedding_vector_dim
            ),
            dtype=torch.float32,
        )
        filled_vectors.fill_(0.0)
        return filled_vectors, self.max_sentences, 1

    def __matrix_encode__(self, batch_size, matrices):
        '''
        该函数不加特殊字符[CLS]与[SEP]的向量
        :param batch_size:
        :param matrices:
        :return:
        '''
        max_len = max(matrix.shape[0] for matrix in matrices)
        if self.matrix_add_special_token:
            max_len -= 2
        if self.truncation_matrix_length:
            max_len = min(max_len, self.truncation_matrix_length)
        if self.matrix_add_special_token:
            max_len += 2
        else:
            max_len = max_len + int(self.prepend_bos) + int(self.append_eos)
        embedding_vector_dim = matrices[0].shape[1]
        # for input
        filled_matrices = torch.empty(
            (
                batch_size,
                max_len,
                embedding_vector_dim
            ),
            dtype=torch.float32,
        )
        filled_matrices.fill_(0.0)
        attention_masks = torch.empty(
            (
                batch_size,
                max_len,
            ),
            dtype=torch.int64,
        )
        attention_masks.fill_(0)
        return filled_matrices, attention_masks, max_len

    def __atom_matrix_encode__(self, batch_size, matrices):
        '''
        该函数不加特殊字符[CLS]与[SEP]的向量
        :param batch_size:
        :param matrices:
        :return:
        '''
        max_len = max(matrix.shape[0] for matrix in matrices)
        if self.matrix_add_special_token:
            max_len -= 2
        if self.atom_truncation_matrix_length:
            max_len = min(max_len, self.atom_truncation_matrix_length)
        if self.matrix_add_special_token:
            max_len += 2
        else:
            max_len = max_len + int(self.atom_prepend_bos) + int(self.atom_append_eos)
        embedding_vector_dim = matrices[0].shape[1]
        # for input
        filled_matrices = torch.empty(
            (
                batch_size,
                max_len,
                embedding_vector_dim
            ),
            dtype=torch.float32,
        )
        filled_matrices.fill_(0.0)
        attention_masks = torch.empty(
            (
                batch_size,
                max_len,
            ),
            dtype=torch.int64,
        )
        attention_masks.fill_(0)
        return filled_matrices, attention_masks, max_len

    def __multi_matrix_encode__(self, batch_size, matrices):
        '''
        该函数不加特殊字符[CLS]与[SEP]的向量
        :param batch_size:
        :param matrices:
        :return:
        '''
        max_sentence_num = max(len(cur_matrix) for cur_matrix in matrices)
        max_sentence_num = min(max_sentence_num, self.max_sentences)
        if self.trunc_type == "left":
            max_sentence_len = max(max(matrix.shape[0] for matrix in cur_matrix[-max_sentence_num:]) for cur_matrix in matrices)
        else:
            max_sentence_len = max(max(matrix.shape[0] for matrix in cur_matrix[:max_sentence_num]) for cur_matrix in matrices)
        # print("encoder max_sentence_num:%d, max_sentence_len: %d" % (max_sentence_num, max_sentence_len))
        if self.matrix_add_special_token:
            max_sentence_len -= 2
        max_sentence_len = min(max_sentence_len, self.max_sentence_length)
        # print("encoder max_sentence_num:%d, max_sentence_len: %d" % (max_sentence_num, max_sentence_len))
        if self.matrix_add_special_token:
            max_sentence_len += 2
        else:
            max_sentence_len = max_sentence_len + int(self.prepend_bos) + int(self.append_eos)
        # print("encoder max_sentence_num:%d, max_sentence_len: %d" % (max_sentence_num, max_sentence_len))
        # print("self.max_sentence_length: %d" % self.max_sentence_length)
        # print("max_sentence_len: %d" % max_sentence_len)
        embedding_vector_dim = matrices[0][0].shape[1]
        # for input
        filled_matrices = torch.empty(
            (
                batch_size,
                max_sentence_num,
                max_sentence_len,
                embedding_vector_dim
            ),
            dtype=torch.float32,
        )
        filled_matrices.fill_(0.0)
        attention_masks = torch.empty(
            (
                batch_size,
                max_sentence_num,
                max_sentence_len
            ),
            dtype=torch.int64,
        )
        attention_masks.fill_(0)
        return filled_matrices, attention_masks, max_sentence_num, max_sentence_len

    def __call_single__(self, batch_size, seq_types, seqs, vectors, matrices, labels):
        max_length = sys.maxsize
        input_ids, position_ids, token_type_ids, seq_attention_masks = None, None, None, None
        seq_part_of_input = False
        molecule_flag = False
        multi_seq_flag = False
        if seqs and (self.input_type is None or "seq" in self.input_type):
            new_seqs = []
            for seq_idx, seq_type in enumerate(seq_types):
                if seq_type == "gene":
                    new_seqs.append(gene_seq_replace(seqs[seq_idx].upper()))
                elif seq_type == "molecule":
                    if isinstance(seqs[seq_idx], str):
                        new_seqs.append(AlphabetAtom.smiles_2_atom_seq(seqs[seq_idx]))
                    else:
                        new_seqs.append(seqs[seq_idx])
                    molecule_flag = True
                elif seq_type == "multi_gene":
                    new_seqs.append([gene_seq_replace(seq).upper() for seq in seqs[seq_idx].split(",")])
                    multi_seq_flag = True
                elif seq_type == "multi_prot":
                    new_seqs.append([seq.upper() for seq in seqs[seq_idx].split(",")])
                    multi_seq_flag = True
                else:
                    new_seqs.append(seqs[seq_idx].upper())
            if molecule_flag:
                # seq_encoded_list没有加特殊字符，input_ids标志位来占位， seq_max_length 根据标志位来加特殊字符长度
                seq_encoded_list, input_ids, position_ids, token_type_ids, seq_attention_masks, seq_max_length = self.__atom_seq_encode__(
                    batch_size=batch_size,
                    seq_types=seq_types,
                    seqs=new_seqs
                )

            elif multi_seq_flag:
                # seq_encoded_list根据标志位来加特殊字符，input_ids根据标志位来加特殊字符， seq_max_len 根据标志位来加特殊字符长度
                seq_encoded_list, input_ids, position_ids, token_type_ids, seq_attention_masks, seq_max_num, seq_max_len = self.__multi_seq_encode__(
                    batch_size=batch_size,
                    seq_types=seq_types,
                    seqs=new_seqs
                )
                '''
                print("seq_max_num: %d" % seq_max_num)
                print("seq_max_len: %d" % seq_max_len)
                print(input_ids.shape)
                print("len(seq_encoded_list): %d" % len(seq_encoded_list))
                for input_id in input_ids:
                    print(len(input_id))
                    for matrix in input_id:
                        print(matrix.shape)
                    print("*****")
                '''
            else:
                # seq_encoded_list没有加特殊字符，input_ids标志位来占位， seq_max_length 根据标志位来加特殊字符长度
                seq_encoded_list, input_ids, position_ids, token_type_ids, seq_attention_masks, seq_max_length = self.__seq_encode__(
                    batch_size=batch_size,
                    seq_types=seq_types,
                    seqs=new_seqs
                )
            if multi_seq_flag:
                max_length = min(max_length, seq_max_num * seq_max_len)
            else:
                max_length = min(max_length, seq_max_length)
            seq_part_of_input = True

        encoded_vectors = None
        vector_part_of_input = False
        if vectors is not None and len(vectors) > 0:
            if multi_seq_flag:
                encoded_vectors, vector_max_num, vector_max_len = self.__multi_vector_encode__(batch_size=batch_size, vectors=vectors)
            elif molecule_flag:
                encoded_vectors, vector_max_length = self.__atom_vector_encode__(batch_size=batch_size, vectors=vectors)
            else:
                encoded_vectors, vector_max_length = self.__vector_encode__(batch_size=batch_size, vectors=vectors)
            # max_length = min(max_length, vector_max_length)
            vector_part_of_input = True

        encoded_matrices, matrix_attention_masks = None, None
        matrix_part_of_input = False
        # print("multi_seq_flag:", multi_seq_flag)
        if matrices is not None and len(matrices) > 0:
            if multi_seq_flag:
                # 根据标记位填充，根据标记位填充，句子数量，根据标记位是否加上特殊字符长度
                encoded_matrices, matrix_attention_masks, matrix_max_num, matrix_max_len = self.__multi_matrix_encode__(
                    batch_size=batch_size,
                    matrices=matrices
                )
                '''
                print("matrix_max_num: %d" % matrix_max_num)
                print("matrix_max_len: %d" % matrix_max_len)
                print(encoded_matrices.shape)
                print("len(matrices): %d" % len(matrices))
                for matrix_array in matrices:
                    print(len(matrix_array))
                    for matrix in matrix_array:
                        print(matrix.shape)
                    print("*****")
                '''
            elif molecule_flag:
                # 根据标记位填充，根据标记位填充，句子数量，根据标记位是否加上特殊字符长度
                encoded_matrices, matrix_attention_masks, matrix_max_length = self.__atom_matrix_encode__(
                    batch_size=batch_size,
                    matrices=matrices
                )
            else:
                # 根据标记位填充，根据标记位填充，句子数量，根据标记位是否加上特殊字符长度
                encoded_matrices, matrix_attention_masks, matrix_max_length = self.__matrix_encode__(
                    batch_size=batch_size,
                    matrices=matrices
                )
            if multi_seq_flag:
                max_length = min(max_length, matrix_max_num * matrix_max_len)
            else:
                max_length = min(max_length, matrix_max_length)
            matrix_part_of_input = True
        has_label = False
        if labels:
            has_label = True

        new_labels = []
        num_sentences = 1
        sentence_length = 1
        for sample_idx in range(batch_size):
            # seq
            if seq_part_of_input:
                if multi_seq_flag:
                    # cls_idx 已经添加
                    pass
                elif not molecule_flag and self.prepend_bos:
                    input_ids[sample_idx, 0] = self.cls_idx
                elif molecule_flag and self.atom_prepend_bos:
                    input_ids[sample_idx, 0] = self.atom_cls_idx

                seq_encoded = seq_encoded_list[sample_idx]
                real_seq_len = len(seq_encoded)

                # seq_tensor = torch.tensor(seq_encoded, dtype=torch.int64)
                # print("seq_encoded：")
                # print(seq_encoded)
                if multi_seq_flag:
                    cur_seq_num = min(len(seq_encoded), seq_max_num)
                    if len(seq_encoded) > cur_seq_num:
                        if self.trunc_type == "left":
                            seq_encoded = seq_encoded[-cur_seq_num:]
                        else:
                            seq_encoded = seq_encoded[cur_seq_num:]
                    if num_sentences < cur_seq_num:
                        num_sentences = cur_seq_num
                    # print("cur_seq_num: %d" % len(seq_encoded))
                    for seq_idx in range(cur_seq_num):
                        cur_seq = seq_encoded[seq_idx]
                        cur_seq_len = min(len(cur_seq), seq_max_len)
                        '''
                        print("cur_seq:")
                        print(cur_seq_len)
                        print("input_ids:")
                        print(input_ids.shape)
                        '''
                        input_ids[sample_idx, seq_idx, :cur_seq_len] = torch.tensor(cur_seq[:cur_seq_len], dtype=torch.int64)
                        seq_attention_masks[sample_idx, seq_idx, :cur_seq_len] = 1
                        if cur_seq_len > sentence_length:
                            sentence_length = cur_seq_len
                elif molecule_flag:
                    seq_tensor = torch.tensor(seq_encoded, dtype=torch.int64)
                    input_ids[sample_idx, int(self.atom_prepend_bos): real_seq_len + int(self.atom_prepend_bos)] = seq_tensor
                    cur_sentence_length = int(self.atom_prepend_bos) + real_seq_len + int(self.atom_prepend_bos)
                    if cur_sentence_length > sentence_length:
                        sentence_length = cur_sentence_length
                else:
                    seq_tensor = torch.tensor(seq_encoded, dtype=torch.int64)
                    input_ids[sample_idx, int(self.prepend_bos): real_seq_len + int(self.prepend_bos)] = seq_tensor
                    cur_sentence_length = int(self.prepend_bos) + real_seq_len + int(self.prepend_bos)
                    if cur_sentence_length > sentence_length:
                        sentence_length = cur_sentence_length

                if multi_seq_flag:
                    # eos_idx 已经添加
                    pass
                elif not molecule_flag and self.append_eos:
                    input_ids[sample_idx, real_seq_len + int(self.prepend_bos)] = self.eos_idx
                elif molecule_flag and self.atom_append_eos:
                    input_ids[sample_idx, real_seq_len + int(self.atom_prepend_bos)] = self.atom_eos_idx

                if multi_seq_flag:
                    cur_len = num_sentences * sentence_length
                elif molecule_flag:
                    cur_len = int(self.atom_prepend_bos) + real_seq_len + int(self.atom_append_eos)
                else:
                    cur_len = int(self.prepend_bos) + real_seq_len + int(self.append_eos)

                if not self.no_position_embeddings:
                    if multi_seq_flag:
                        for pos_idx in range(0, cur_len):
                            position_ids[sample_idx, pos_idx//sentence_length, pos_idx % sentence_length] = pos_idx % sentence_length
                    else:
                        for pos_idx in range(0, cur_len):
                            position_ids[sample_idx, pos_idx] = pos_idx

                if not self.no_token_type_embeddings:
                    seq_type = seq_types[sample_idx]
                    if seq_type == "gene":
                        type_value = 0
                    else:
                        type_value = 1
                    if multi_seq_flag:
                        for pos_idx in range(0, cur_len):
                            token_type_ids[sample_idx, pos_idx//sentence_length, pos_idx % sentence_length] = type_value
                    else:
                        for pos_idx in range(0, cur_len):
                            token_type_ids[sample_idx, pos_idx] = type_value

                if multi_seq_flag:
                    pass
                else:
                    seq_attention_masks[sample_idx, 0: cur_len] = 1

            # vector
            if vector_part_of_input:
                if multi_seq_flag:
                    cur_vector_num = min(len(vectors[sample_idx]), vector_max_num)
                    if num_sentences < cur_vector_num:
                        num_sentences = cur_vector_num
                    for vector_idx in range(cur_vector_num):
                        encoded_vectors[sample_idx, vector_idx, :] = torch.tensor(vectors[sample_idx][vector_idx], dtype=torch.float32)
                else:
                    encoded_vectors[sample_idx, :] = torch.tensor(vectors[sample_idx], dtype=torch.float32)

            # matrix
            if matrix_part_of_input:
                '''
                matrix_encoded = matrices[sample_idx]
                if self.matrix_add_special_token:
                    real_seq_len = matrix_encoded.shape[0] - 2
                else:
                    real_seq_len = matrix_encoded.shape[0]
                if multi_seq_flag:
                    pass
                elif molecule_flag:
                    # real_seq_len = real_seq_len - int(self.atom_prepend_bos) - int(self.atom_append_eos)
                    real_seq_len = min(real_seq_len, self.atom_truncation_matrix_length)
                else:
                    # real_seq_len = real_seq_len - int(self.prepend_bos) - int(self.append_eos)
                    real_seq_len = min(real_seq_len, self.truncation_matrix_length)
                # print("real_seq_len: %d" % real_seq_len)
                '''
                if multi_seq_flag:
                    # 多序列matrix
                    matrix_encoded_list = matrices[sample_idx]
                    cur_matrix_num = min(len(matrix_encoded_list), matrix_max_num)
                    if len(matrix_encoded_list) > cur_matrix_num:
                        if self.trunc_type == "left":
                            matrix_encoded_list = matrix_encoded_list[:cur_matrix_num]
                        else:
                            matrix_encoded_list = matrix_encoded_list[-cur_matrix_num:]
                    if num_sentences < cur_matrix_num:
                        num_sentences = cur_matrix_num
                    # print("matrix_encoded_list: %d" % len(matrix_encoded_list))
                    for matrix_idx in range(cur_matrix_num):
                        # print("matrix_idx: %d" % matrix_idx)
                        cur_matrix = matrix_encoded_list[matrix_idx]
                        cur_matrix = torch.tensor(cur_matrix, dtype=torch.float32)
                        cur_matrix_len = min(cur_matrix.shape[0], matrix_max_len)
                        if self.matrix_add_special_token:
                            encoded_matrices[sample_idx, matrix_idx, 0: cur_matrix_len - 1] = cur_matrix[0:cur_matrix_len - 1]
                            encoded_matrices[sample_idx, matrix_idx, cur_matrix_len - 1] = cur_matrix[-1]
                            matrix_attention_masks[sample_idx, matrix_idx, 0:cur_matrix_len] = 1
                        else:
                            encoded_matrices[sample_idx, matrix_idx, int(self.prepend_bos): cur_matrix_len + int(self.prepend_bos)] = cur_matrix[0:cur_matrix_len]
                            matrix_attention_masks[sample_idx, matrix_idx, 0: int(self.prepend_bos) + cur_matrix_len + int(self.append_eos)] = 1
                            cur_matrix_len = int(self.prepend_bos) + cur_matrix_len + int(self.append_eos)
                        if sentence_length < cur_matrix_len:
                            sentence_length = cur_matrix_len
                else:
                    matrix_encoded = matrices[sample_idx]
                    if self.matrix_add_special_token:
                        real_seq_len = matrix_encoded.shape[0] - 2
                    else:
                        real_seq_len = matrix_encoded.shape[0]
                    if molecule_flag:
                        # real_seq_len = real_seq_len - int(self.atom_prepend_bos) - int(self.atom_append_eos)
                        real_seq_len = min(real_seq_len, self.atom_truncation_matrix_length)
                        # matrix = torch.tensor(matrix_encoded, dtype=torch.float32)
                        matrix = matrix_encoded.clone().detach()
                        if self.matrix_add_special_token:
                            encoded_matrices[sample_idx, 0: real_seq_len + 2] \
                                = matrix[0: real_seq_len + 2]
                            matrix_attention_masks[sample_idx, 0: real_seq_len + 2] = 1
                            cur_sentence_length = real_seq_len + 2
                        else:
                            encoded_matrices[sample_idx, int(self.atom_prepend_bos): real_seq_len + int(self.atom_prepend_bos)] \
                                = matrix[0: real_seq_len]
                            # matrix_attention_masks[sample_idx, int(self.atom_prepend_bos): real_seq_len + int(self.atom_prepend_bos)] = 1
                            matrix_attention_masks[sample_idx, 0: int(self.atom_prepend_bos) + real_seq_len + int(self.atom_append_eos)] = 1
                            cur_sentence_length = int(self.atom_prepend_bos) + real_seq_len + int(self.atom_prepend_bos)
                        if cur_sentence_length > sentence_length:
                            sentence_length = cur_sentence_length
                    else:
                        # real_seq_len = real_seq_len - int(self.prepend_bos) - int(self.append_eos)
                        real_seq_len = min(real_seq_len, self.truncation_matrix_length)
                        # matrix = torch.tensor(matrix_encoded, dtype=torch.float32)
                        matrix = matrix_encoded.clone().detach()
                        if self.matrix_add_special_token:
                            encoded_matrices[sample_idx, 0: real_seq_len + 2] = matrix[0: real_seq_len + 2]
                            matrix_attention_masks[sample_idx, 0: real_seq_len + 2] = 1
                            cur_sentence_length = real_seq_len + 2
                        else:
                            encoded_matrices[sample_idx, int(self.prepend_bos): real_seq_len + int(self.prepend_bos)] = matrix[0: real_seq_len]
                            # matrix_attention_masks[sample_idx, int(self.prepend_bos): real_seq_len + int(self.prepend_bos)] = 1
                            matrix_attention_masks[sample_idx, 0: int(self.prepend_bos) + real_seq_len + int(self.append_eos)] = 1
                            cur_sentence_length = int(self.prepend_bos) + real_seq_len + int(self.prepend_bos)
                        if cur_sentence_length > sentence_length:
                            sentence_length = cur_sentence_length

            if has_label:
                if multi_seq_flag:
                    # to do
                    new_labels.append(
                        self.__parse_label__(max_length, self.task_level_type,
                                             self.label_size, self.output_mode, labels[sample_idx]))
                elif molecule_flag:
                    new_labels.append(
                        self.__atom_parse_label__(max_length, self.task_level_type,
                                                  self.label_size, self.output_mode, labels[sample_idx]))
                else:
                    new_labels.append(
                        self.__parse_label__(max_length, self.task_level_type,
                                             self.label_size, self.output_mode, labels[sample_idx]))
        if new_labels is not None and new_labels:
            if self.output_mode in ["regression"]:
                labels = torch.tensor(new_labels, dtype=torch.float32)
            else:
                labels = torch.tensor(new_labels, dtype=torch.int64)
        else:
            labels = None
        '''
        print(input_ids.shape)
        print("encoded_matrices:")
        print(encoded_matrices.shape)
        print("num_sentences:%d" % num_sentences)
        print("sentence_length:%d" % sentence_length)
        if labels is not None:
            print("labels:")
            print(labels.shape)
        '''

        if multi_seq_flag:
            if seq_part_of_input:
                input_ids = torch.reshape(input_ids, (input_ids.shape[0], -1))
            if matrix_part_of_input:
                encoded_matrices = torch.reshape(encoded_matrices, (encoded_matrices.shape[0], -1, encoded_matrices.shape[-1]))
            if position_ids is not None:
                position_ids = torch.reshape(position_ids, (position_ids.shape[0], -1))
            if token_type_ids is not None:
                token_type_ids = torch.reshape(token_type_ids, (token_type_ids.shape[0], -1))
            if seq_attention_masks is not None:
                seq_attention_masks = torch.reshape(seq_attention_masks, (seq_attention_masks.shape[0], -1))
            if matrix_attention_masks is not None:
                matrix_attention_masks = torch.reshape(matrix_attention_masks, (matrix_attention_masks.shape[0], -1))
        '''
        print(input_ids.shape)
        print("encoded_matrices:")
        print(encoded_matrices.shape)
        print("num_sentences:%d" % num_sentences)
        print("sentence_length:%d" % sentence_length)
        if labels is not None:
            print("labels:")
            print(labels.shape)
        print("-" * 50)
        '''

        return input_ids, \
               position_ids, \
               token_type_ids, \
               seq_attention_masks, \
               encoded_vectors, \
               encoded_matrices, \
               matrix_attention_masks, \
               num_sentences, \
               sentence_length, \
               labels

    def __call__(self, raw_batch: Sequence[dict]):
        batch_size = len(raw_batch)
        # pair
        if "seq_id_a" in raw_batch[0] and "seq_id_b" in raw_batch[0]:
            res = {}
            # seq_ids_a = []
            seq_types_a = []
            seqs_a = []
            vectors_a = []
            matrices_a = []

            # seq_ids_b = []
            seq_types_b = []
            seqs_b = []
            vectors_b = []
            matrices_b = []

            labels = []
            for item in raw_batch:
                # seq_ids_a.append(item["seq_id_a"])
                seq_types_a.append(item["seq_type_a"])
                if item["seq_a"] is not None:
                    seqs_a.append(item["seq_a"])
                if item["vector_a"] is not None:
                    vectors_a.append(item["vector_a"])
                if item["matrix_a"] is not None:
                    matrices_a.append(item["matrix_a"])

                # seq_ids_b.append(item["seq_id_b"])
                seq_types_b.append(item["seq_type_b"])
                if item["seq_b"] is not None:
                    seqs_b.append(item["seq_b"])
                if item["vector_b"] is not None:
                    vectors_b.append(item["vector_b"])
                if item["matrix_b"] is not None:
                    matrices_b.append(item["matrix_b"])
                if "label" in item and item["label"] is not None:
                    labels.append(item["label"])
            input_ids_a, position_ids_a, token_type_ids_a, seq_attention_masks_a, \
            encoded_vectors_a, encoded_matrices_a, matrix_attention_masks_a, \
            num_sentences_a, sentence_length_a, labels \
                = self.__call_single__(batch_size, seq_types_a, seqs_a, vectors_a, matrices_a, labels)
            if not hasattr(self, "max_sentences") or self.max_sentences is None:
                res.update({
                    "input_ids_a": input_ids_a,
                    "position_ids_a": position_ids_a,
                    "token_type_ids_a": token_type_ids_a,
                    "seq_attention_masks_a": seq_attention_masks_a,
                    "vectors_a": encoded_vectors_a,
                    "matrices_a": encoded_matrices_a,
                    "matrix_attention_masks_a": matrix_attention_masks_a,
                    "labels": labels if labels is not None and len(labels) > 0 else None
                })
            else:
                res.update({
                    "input_ids_a": input_ids_a,
                    "position_ids_a": position_ids_a,
                    "token_type_ids_a": token_type_ids_a,
                    "seq_attention_masks_a": seq_attention_masks_a,
                    "vectors_a": encoded_vectors_a,
                    "matrices_a": encoded_matrices_a,
                    "matrix_attention_masks_a": matrix_attention_masks_a,
                    "num_sentences_a": num_sentences_a,
                    "sentence_length_a": sentence_length_a,
                    "labels": labels if labels is not None and len(labels) > 0 else None
                })
            input_ids_b, position_ids_b, token_type_ids_b, seq_attention_masks_b, \
            encoded_vectors_b, encoded_matrices_b, matrix_attention_masks_b, \
            num_sentences_b, sentence_length_b,  _ \
                = self.__call_single__(batch_size, seq_types_b, seqs_b, vectors_b, matrices_b, labels=None)
            if not hasattr(self, "max_sentences") or self.max_sentences is None:
                res.update({
                    "input_ids_b": input_ids_b,
                    "position_ids_b": position_ids_b,
                    "token_type_ids_b": token_type_ids_b,
                    "seq_attention_masks_b": seq_attention_masks_b,
                    "vectors_b": encoded_vectors_b,
                    "matrices_b": encoded_matrices_b,
                    "matrix_attention_masks_b": matrix_attention_masks_b
                })
            else:
                res.update({
                    "input_ids_b": input_ids_b,
                    "position_ids_b": position_ids_b,
                    "token_type_ids_b": token_type_ids_b,
                    "seq_attention_masks_b": seq_attention_masks_b,
                    "vectors_b": encoded_vectors_b,
                    "matrices_b": encoded_matrices_b,
                    "num_sentences_b": num_sentences_b,
                    "sentence_length_b": sentence_length_b,
                    "matrix_attention_masks_b": matrix_attention_masks_b
                })
            if "express_list_a" in raw_batch[0] and raw_batch[0]["express_list_a"] is not None:
                express_list_a = []
                for item in raw_batch:
                    express_list_a.append(item["express_list_a"])
                res.update({
                    "express_input_ids_a": self.__express_encode__(batch_size=batch_size, express_list=express_list_a)
                })
            if "express_list_b" in raw_batch[0] and raw_batch[0]["express_list_b"] is not None:
                express_list_b = []
                for item in raw_batch:
                    express_list_b.append(item["express_list_b"])
                res.update({
                    "express_input_ids_b": self.__express_encode__(batch_size=batch_size, express_list=express_list_b)
                })
            return res
        else:
            res = {}
            # seq_ids = []
            seq_types = []
            seqs = []
            vectors = []
            matrices = []
            labels = []
            for item in raw_batch:
                # seq_ids.append(item["seq_id"])
                seq_types.append(item["seq_type"])
                if item["seq"] is not None:
                    seqs.append(item["seq"])
                if item["vector"] is not None:
                    vectors.append(item["vector"])
                if item["matrix"] is not None:
                    matrices.append(item["matrix"])
                if item["label"] is not None:
                    labels.append(item["label"])
            '''
            print("seqs:")
            print(seqs)
            print([len(seq) for seq in seqs])
            print("matrices:")
            print(matrices)
            print([matrix.shape for matrix in matrices])
            print("labels:")
            print(labels)
            print([len(eval(label)) for label in labels])
            '''
            input_ids, position_ids, token_type_ids, seq_attention_masks, encoded_vectors, \
            encoded_matrices, matrix_attention_masks, num_sentences, sentence_length, labels = self.__call_single__(
                batch_size, seq_types, seqs, vectors, matrices, labels=labels)

            if not hasattr(self, "max_sentences") or self.max_sentences is None:
                res.update({
                    "input_ids": input_ids,
                    "position_ids": position_ids,
                    "token_type_ids": token_type_ids,
                    "seq_attention_masks": seq_attention_masks,
                    "vectors": encoded_vectors,
                    "matrices": encoded_matrices,
                    "matrix_attention_masks": matrix_attention_masks,
                    "labels": labels if labels is not None and len(labels) > 0 else None
                })
            else:
                res.update({
                    "input_ids": input_ids,
                    "position_ids": position_ids,
                    "token_type_ids": token_type_ids,
                    "seq_attention_masks": seq_attention_masks,
                    "vectors": encoded_vectors,
                    "matrices": encoded_matrices,
                    "matrix_attention_masks": matrix_attention_masks,
                    "num_sentences": num_sentences,
                    "sentence_length": sentence_length,
                    "labels": labels if labels is not None and len(labels) > 0 else None
                })
            if "express_list" in raw_batch[0] and raw_batch[0]["express_list"] is not None:
                express_list = []
                for item in raw_batch:
                    express_list.append(item["express_list"])
                res.update({
                    "express_input_ids": self.__express_encode__(batch_size=batch_size, express_list=express_list)
                })
            '''
            for item in res.items():
                key_name = item[0]
                print(key_name, ":")
                if item[1] is not None:
                    print(item[1])
                    print(item[1].shape)
                else:
                    print("None")
            '''
            return res

