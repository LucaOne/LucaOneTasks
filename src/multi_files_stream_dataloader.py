#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/4/21 10:54
@project: LucaOneTasks
@file: stream_dataloader
@desc: dataset stream dataloader for training downstream models
'''
import os
import sys
import random
import shutil
import numpy as np
from datetime import datetime
sys.path.append(".")
sys.path.append("..")
sys.path.append("../src")
try:
    from file_operator import *
except ImportError:
    from src.file_operator import *
csv.field_size_limit(sys.maxsize)


class MultiFilesStreamLoader(object):
    def __init__(self,
                 filepaths,
                 batch_size,
                 buffer_size,
                 parse_row_func,
                 batch_data_func,
                 task_level_type,
                 input_mode,
                 input_type,
                 output_mode,
                 label_size,
                 dataset_type="train",
                 vector_dirpath=None,
                 matrix_dirpath=None,
                 inference=False,
                 header=True,
                 shuffle=False,
                 seed=1221):
        if buffer_size % batch_size != 0:
            raise Exception("buffer_size must be evenly div by batch_size")
        self.shuffle = shuffle
        self.dataset_type = dataset_type
        if self.dataset_type == "train":
            self.shuffle = True
        self.cached = False

        self.filepaths = filepaths
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.parse_row_func = parse_row_func
        self.batch_data_func = batch_data_func
        self.task_level_type = task_level_type
        self.input_mode = input_mode
        self.input_type = input_type
        self.output_mode = output_mode
        self.vector_dirpath = vector_dirpath
        self.matrix_dirpath = matrix_dirpath
        self.inference = inference
        self.label_size = label_size

        self.filepaths = self.input_file_process(self.filepaths)
        if self.shuffle:
            for _ in range(5):
                random.shuffle(self.filepaths)
        self.header = header
        self.rnd = np.random.RandomState(seed)
        self.ptr = 0  # cur index of buffer
        self.total_filename_num = len(self.filepaths)
        self.cur_file_idx = 0
        print("filepath: %s" % self.filepaths[self.cur_file_idx % self.total_filename_num])
        self.cur_fin = file_reader(self.filepaths[self.cur_file_idx % self.total_filename_num],
                                   header_filter=True, header=self.header)
        # memory buffer
        self.buffer = []
        self.epoch_over = False
        self.enough_flag = False
        self.reload_buffer()

    def delete_cache(self):
        if self.shuffle and self.cached:
            for filepath in self.filepaths:
                dirpath = os.path.dirname(filepath)
                if os.path.exists(dirpath):
                    shutil.rmtree(dirpath)
                    print("deleted cache dirpath=%s" % dirpath)

    def input_file_process(self, filepaths):
        if isinstance(filepaths, str):
            if os.path.isdir(filepaths):
                # 是一个目录
                if self.shuffle:
                    filepaths = self.cache(filepaths)
                    self.cached = True
                new_filepaths = []
                for filename in os.listdir(filepaths):
                    if not filename.startswith("."):
                        new_filepaths.append(os.path.join(filepaths, filename))
                filepaths = new_filepaths
            else:
                # 是一个文件
                if self.shuffle:
                    dirpath = os.path.dirname(filepaths)
                    filename = os.path.basename(filepaths)
                    cached_dirpath = self.cache(dirpath)
                    cached_filepaths = os.path.join(cached_dirpath, filename)
                    filepaths = cached_filepaths
                    self.cached = True
                filepaths = [filepaths]
        else:
            # 是一个列表，必须是文件列表
            if self.shuffle:
                new_filepaths = []
                for filepath in filepaths:
                    if not os.path.exists(filepath) or not os.path.isfile(filepath):
                        continue
                    dirpath = os.path.dirname(filepath)
                    filename = os.path.basename(filepath)
                    cached_dirpath = self.cache(dirpath)
                    cached_filepath = os.path.join(cached_dirpath, filename)
                    self.cached = True
                    new_filepaths.append(cached_filepath)
                filepaths = new_filepaths
        return filepaths

    @staticmethod
    def cache(dirpath):
        # 对于训练，因为每个epoch内，需要将文件内容打乱然后重新写入文件，也就是会修改文件内容，那么则cache，只修改cache中的内容
        now = datetime.now()
        time_str = now.strftime("%Y%m%d%H%M%S%f")
        dirpath = dirpath.strip()
        suffix = "_" + time_str + "_cached"
        if dirpath[-1] == "/":
            cached_dirpath = dirpath[:-1] + suffix
        else:
            cached_dirpath = dirpath + suffix
        if not os.path.exists(cached_dirpath):
            os.makedirs(cached_dirpath)
        for filename in os.listdir(dirpath):
            source_file = os.path.join(dirpath, filename)
            target_file = os.path.join(cached_dirpath, filename)
            # 仅复制文件，不复制文件夹
            if os.path.isfile(source_file):
                shutil.copy2(source_file, target_file)
        print("cached dirpath: %s -> %s" % (dirpath, cached_dirpath))
        return cached_dirpath

    def next_file_reader(self):
        # self.cur_fin.close()
        self.cur_file_idx += 1
        self.cur_fin = file_reader(self.filepaths[self.cur_file_idx % self.total_filename_num],
                                   header_filter=True, header=self.header)

    def reset_file_reader(self):
        # self.cur_fin.close()
        if self.shuffle:
            random.shuffle(self.filepaths)
        self.cur_file_idx = 0
        filepath = self.filepaths[self.cur_file_idx % self.total_filename_num]
        if self.shuffle:
            # 读取文件内容
            with open(filepath, 'r') as file:
                lines = file.readlines()
            # 打乱行的顺序
            contents = lines[1:]
            for _ in range(5):
                random.shuffle(contents)
            contents = lines[0:1] + contents
            # 将打乱后的内容写回文件
            with open(filepath, 'w') as file:
                file.writelines(contents)
        self.cur_fin = file_reader(filepath, header_filter=True, header=self.header)

    def read_one_line(self):
        try:
            row = self.cur_fin.__next__()
            if self.input_mode == "pair":
                if len(row) == 5:
                    seq_id_a, seq_id_b, seq_a, seq_b, label = row[0:5]
                    seq_type_a, seq_type_b, vector_filename_a, vector_filename_b, matrix_filename_a, matrix_filename_b = None, None, None, None, None, None
                    express_list_a, express_list_b = None, None
                elif len(row) == 7:
                    seq_id_a, seq_id_b, seq_type_a, seq_type_b, seq_a, seq_b, label = row[0:7]
                    vector_filename_a, vector_filename_b, matrix_filename_a, matrix_filename_b = None, None, None, None
                    express_list_a, express_list_b = None, None
                elif len(row) == 11:
                    seq_id_a, seq_id_b, seq_type_a, seq_type_b, seq_a, seq_b, vector_filename_a, vector_filename_b, matrix_filename_a, matrix_filename_b, label = row[0:11]
                    express_list_a, express_list_b = None, None
                elif len(row) == 13:
                    seq_id_a, seq_id_b, seq_type_a, seq_type_b, seq_a, seq_b, vector_filename_a, vector_filename_b, matrix_filename_a, matrix_filename_b, express_list_a, express_list_b, label = row[0:13]
                else:
                    raise Exception("the cols num not in [5, 7, 11]")
                res = {
                    "seq_id_a": seq_id_a,
                    "seq_id_b": seq_id_b,
                    "seq_type_a": seq_type_a,
                    "seq_type_b": seq_type_b,
                    "seq_a": seq_a.upper(),
                    "seq_b": seq_b.upper()
                }
                if not self.inference:
                    res.update({
                        "label": label
                    })
                if "vector" in self.input_type:
                    res.update({
                        "vector_filename_a": vector_filename_a,
                        "vector_filename_b": vector_filename_b
                    })
                if "matrix" in self.input_type:
                    res.update({
                        "matrix_filename_a": matrix_filename_a,
                        "matrix_filename_b": matrix_filename_b
                    })
                if "express" in self.input_type:
                    res.update({
                        "express_list_a": eval(express_list_a) if express_list_a else None,
                        "express_list_b": eval(express_list_b) if express_list_b else None
                    })

            else:
                if len(row) == 3:
                    seq_id, seq, label = row[0:3]
                    seq_type, vector_filename, matrix_filename = "prot", None, None
                    express_list = None
                elif len(row) == 4:
                    seq_id, seq_type, seq, label = row[0:4]
                    vector_filename, matrix_filename = None, None
                    express_list = None
                elif len(row) == 6:
                    seq_id, seq_type, seq, vector_filename, matrix_filename, label = row[0:6]
                    express_list = None
                elif len(row) == 7:
                    seq_id, seq_type, seq, vector_filename, matrix_filename, express_list, label = row[0:7]
                else:
                    raise Exception("the cols num not in [3, 4, 6]")
                res = {
                    "seq_id": seq_id,
                    "seq_type": seq_type,
                    "seq": seq.upper()
                }
                if not self.inference:
                    res.update({
                        "label": label
                    })
                if "vector" in self.input_type:
                    res.update({
                        "vector_filename": vector_filename
                    })
                if "matrix" in self.input_type:
                    res.update({
                        "matrix_filename": matrix_filename
                    })
                if "express" in self.input_type:
                    res.update({
                        "express_list": eval(express_list) if express_list else None
                    })
            return res
        except Exception as e:
            print(e)
            return None

    def encode_line(self, line):
        return self.parse_row_func(**line)

    def reload_buffer(self):
        self.buffer = []
        self.ptr = 0
        ct = 0  # number of lines read
        while ct < self.buffer_size:
            line = self.read_one_line()
            # print("self.cur_file_idx :%d" % self.cur_file_idx )
            # cur file over
            if line is None or len(line) == 0:
                # read next file
                if self.cur_file_idx < self.total_filename_num - 1:
                    self.next_file_reader()
                    # line = self.read_one_line()
                    # self.buffer.append(self.encode_line(line))
                    # ct += 1
                else: # reset
                    # print("file index %d" % (self.cur_file_idx % self.total_filename_num), end="", flush=True)
                    # one epoch over(all files readed)
                    self.epoch_over = True
                    # next epoch
                    self.reset_file_reader()
                    break
            else:
                # done one line
                self.buffer.append(self.encode_line(line))
                ct += 1
        print("\nBuffer size: %d" % len(self.buffer))
        if not self.enough_flag and self.buffer_size == len(self.buffer):
            self.enough_flag = True
        if self.shuffle:
            for _ in range(5):
                self.rnd.shuffle(self.buffer)  # in-place

    def get_batch(self, start, end):
        """
        :param start:
        :param end:
        :return:
        """
        cur_batch = self.buffer[start:end]
        batch_input = self.batch_data_func(cur_batch)
        return batch_input

    def __iter__(self):
        return self

    def __next__(self):
        """
        next batch
        :return:
        """
        if self.enough_flag:
            if self.epoch_over and self.ptr < len(self.buffer):
                start = self.ptr
                end = min(len(self.buffer), self.ptr + self.batch_size)
                self.ptr = end
                # print("ok1:", self.epoch_over, self.ptr, len(self.buffer))
                return self.get_batch(start, end)
            elif self.epoch_over:
                # init for next epoch only for train dataset
                if self.dataset_type == "train":
                    self.reload_buffer()
                    self.epoch_over = False
                # print("ok2:", self.epoch_over, self.ptr, len(self.buffer))
                raise StopIteration
            elif self.ptr + self.batch_size > len(self.buffer):  # less than a batch
                start = self.ptr
                end = len(self.buffer)
                batch_input = self.get_batch(start, end)
                self.reload_buffer()
                # print("ok3:", self.epoch_over, self.ptr, len(self.buffer))
                return batch_input
            else: # more than batch
                start = self.ptr
                end = self.ptr + self.batch_size
                batch_input = self.get_batch(start, end)
                self.ptr += self.batch_size
                if self.ptr == len(self.buffer):
                    self.reload_buffer()
                # print("ok4:", self.epoch_over, self.ptr, len(self.buffer))
                return batch_input
        else:
            if self.ptr < len(self.buffer):
                start = self.ptr
                end = min(len(self.buffer), self.ptr + self.batch_size)
                self.ptr = end
                # print("ok1:", self.epoch_over, self.ptr, len(self.buffer))
                return self.get_batch(start, end)
            else:
                # init for next epoch only for train dataset
                if self.dataset_type == "train":
                    self.reload_buffer()
                # print("ok2:", self.epoch_over, self.ptr, len(self.buffer))
                raise StopIteration

