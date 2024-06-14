#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/4/9 13:46
@project: LucaOneTasks
@file: file_operator
@desc: file operator for LucaOneTasks
'''

import csv,sys
import io, textwrap, itertools
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
csv.field_size_limit(sys.maxsize)


common_nucleotide_set = {'A', 'T', 'C', 'G', 'U', 'N'}

# not {'O', 'U', 'Z', 'J', 'B'}
# Common amino acids
common_amino_acid_set = {'R', 'X', 'S', 'G', 'W', 'I', 'Q', 'A', 'T', 'V', 'K', 'Y', 'C', 'N', 'L', 'F', 'D', 'M', 'P', 'H', 'E'}


def clean_seq(protein_id, seq):
    seq = seq.upper()
    new_seq = ""
    has_invalid_char = False
    invalid_char_set = set()
    for ch in seq:
        if 'A' <= ch <= 'Z' and ch not in ['J']:
            new_seq += ch
        else:
            invalid_char_set.add(ch)
            has_invalid_char = True
    if has_invalid_char:
        print("id: %s. Seq: %s" % (protein_id, seq))
        print("invalid char set:", invalid_char_set)
    return new_seq


def file_reader(filename, header=True, header_filter=True):
    if filename.endswith(".fa") or filename.endswith(".fas") or filename.endswith(".fasta"):
        return fasta_reader(filename)
    elif filename.endswith(".csv"):
        return csv_reader(filename, header=True, header_filter=True)
    elif filename.endswith(".tsv"):
        return tsv_reader(filename, header=True, header_filter=True)
    else:
        return txt_reader(filename, header=header, header_filter=header_filter)


def txt_reader(handle, header=True, header_filter=True):
    '''
    csv 读取器，适合大文件
    :param handle:
    :param header:
    :param header_filter: 返回结果是否去掉头
    :return:
    '''
    handle = handle if isinstance(handle, io.TextIOWrapper) else open(handle, 'r')
    try:
        cnt = 0
        for line in handle:
            cnt += 1
            if header and header_filter and cnt == 1:
                continue
            yield line.strip()
    except Exception as e:
        raise StopIteration
    finally:
        if not handle.closed:
            handle.close()


def tsv_reader(handle, header=True, header_filter=True):
    '''
    csv 读取器，适合大文件
    :param handle:
    :param header:
    :param header_filter: 返回结果是否去掉头
    :return:
    '''
    handle = handle if isinstance(handle, io.TextIOWrapper) else open(handle, 'r')
    try:
        reader = csv.reader(handle, delimiter="\t")
        cnt = 0
        for row in reader:
            cnt += 1
            if header and header_filter and cnt == 1:
                continue
            yield row
    except Exception as e:
        raise StopIteration
    finally:
        if not handle.closed:
            handle.close()


def csv_reader(handle, header=True, header_filter=True):
    '''
    csv 读取器，适合大文件
    :param handle:
    :param header:
    :param header_filter: 返回结果是否去掉头
    :return:
    '''
    handle = handle if isinstance(handle, io.TextIOWrapper) else open(handle, 'r')
    try:
        # data = csv.reader((line.replace('\0','') for line in data_initial), delimiter=",")
        # reader = csv.reader(handle)
        reader = csv.reader((line.replace('\0', '') for line in handle))
        cnt = 0
        for row in reader:
            cnt += 1
            if header and header_filter and cnt == 1:
                continue
            yield row
    except Exception as e:
        raise StopIteration
    finally:
        if not handle.closed:
            handle.close()


def txt_writer(dataset, filepath, header=None):
    '''
    txt 写
    :param dataset: 数据
    :param handle: 文件
    :param header: 头
    :return:
    '''
    '''
    handle = handle if isinstance(handle, io.TextIOWrapper) else open(handle, 'w')
    try:
        if header:
            if isinstance(header, list):
                handle.write(",".join(header) + "\n")
            else:
                handle.write(header + "\n")
        print("header: %s" %header)
        for row in dataset:
            handle.write(str(row) + "\n")
    except Exception as e:
        raise e
    finally:
        if not handle.closed:
            handle.close()
    '''
    with open(filepath, "w") as wfp:
        if header:
            if isinstance(header, list):
                wfp.write(",".join(header) + "\n")
            else:
                wfp.write(header + "\n")
        for row in dataset:
            wfp.write(str(row) + "\n")


def csv_writer(dataset, handle, header):
    '''
    csv 写，适合大文件
    :param dataset: 数据
    :param handle: 文件
    :param header: 头
    :return:
    '''
    handle = handle if isinstance(handle, io.TextIOWrapper) else open(handle, 'w')
    try:
        writer = csv.writer(handle)
        if header:
            writer.writerow(header)
        for row in dataset:
            writer.writerow(row)
    except Exception as e:
        raise e
    finally:
        if not handle.closed:
            handle.close()


def fasta_reader(handle, width=None):
    """
    Reads a FASTA file, yielding header, sequence pairs for each sequence recovered 适合大文件
    args:
        :handle (str, pathliob.Path, or file pointer) - fasta to read from
        :width (int or None) - formats the sequence to have max `width` character per line.
                               If <= 0, processed as None. If None, there is no max width.
    yields:
        :(header, sequence) tuples
    returns:
        :None
    """
    FASTA_STOP_CODON = "*"

    handle = handle if isinstance(handle, io.TextIOWrapper) else open(handle, 'r')
    width = width if isinstance(width, int) and width > 0 else None
    try:
        for is_header, group in itertools.groupby(handle, lambda line: line.startswith(">")):
            if is_header:
                header = group.__next__().strip()
            else:
                seq = ''.join(line.strip() for line in group).strip().rstrip(FASTA_STOP_CODON)
                if width is not None:
                    seq = textwrap.fill(seq, width)
                yield header, seq
    except Exception as e:
        raise StopIteration
    finally:
        if not handle.closed:
            handle.close()


def write_fasta(filepath, sequences):
    '''
    write fasta file
    :param filepath: savepath
    :param sequences: fasta sequence(each item: [id, seq])
    :return:
    '''

    if sequences:
        with open(filepath, "w") as output_handle:
            if len(sequences[0]) > 1 and isinstance(sequences[0][0], str):
                for row in sequences:
                    protein_id = row[0]
                    seq = row[1]
                    sequence = SeqRecord(Seq(seq, None), id=protein_id[1:] if protein_id and protein_id[0] == ">" else protein_id, description="")
                    SeqIO.write(sequence, output_handle, "fasta")
            else:
                for sequence in sequences:
                    SeqIO.write(sequence, output_handle, "fasta")


