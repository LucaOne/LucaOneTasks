#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.**@**.com
@tel: 137****6540
@datetime: 2023/3/20 13:23
@project: LucaOneTasks
@file: inference_embedding
@desc: embedding inference for DNABert2
'''

import sys
import torch
sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../src")
try:
    from file_operator import fasta_reader
    from utils import clean_seq_luca
except ImportError:
    from src.file_operator import fasta_reader
    from src.utils import clean_seq_luca
from transformers import AutoTokenizer, AutoModel

model_id = 'zhihan1996/DNABERT-2-117M'

dnabert2_global_model, dnabert2_global_alphabet, dnabert2_global_version = None, None, None


def predict_embedding(
        sample,
        trunc_type,
        embedding_type,
        repr_layers=[-1],
        truncation_seq_length=4094,
        device=None,
        version="dnabert2",
        matrix_add_special_token=False
):
    '''
    use sequence to predict protein embedding matrix or vector(bos)
    :param sample: [protein_id, protein_sequence]
    :param trunc_type:
    :param embedding_type: bos or representations
    :param repr_layers: [-1]
    :param truncation_seq_length: [4094,2046,1982,1790,1534,1278,1150,1022]
    :param device:
    :param version:
    :param matrix_add_special_token:
    :return: embedding, processed_seq_len
    '''
    global dnabert2_global_model, dnabert2_global_alphabet, dnabert2_global_version
    assert "bos" in embedding_type or "representations" in embedding_type \
           or "matrix" in embedding_type or "vector" in embedding_type or "contacts" in embedding_type
    if len(sample) > 2:
        seq_id, seq = sample[0], sample[2]
    else:
        seq_id, seq = sample[0], sample[1]
    processed_seq = clean_seq_luca(seq_id, seq)
    if len(processed_seq) > truncation_seq_length:
        if trunc_type == "left":
            processed_seq = processed_seq[-truncation_seq_length:]
        else:
            processed_seq = processed_seq[:truncation_seq_length]
    if dnabert2_global_model is None or dnabert2_global_alphabet is None or dnabert2_global_version is None or dnabert2_global_version != version:
        if version == "dnabert2":
            dnabert2_global_alphabet = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            dnabert2_global_model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        else:
            raise Exception("not support this version=%s" % version)
        dnabert2_global_version = version
    '''
    if torch.cuda.is_available() and device is not None:
        dnabert2_global_model = dnabert2_global_model.to(device)
    elif torch.cuda.is_available():
        dnabert2_global_model = dnabert2_global_model.cuda()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("llm use cpu")
    '''
    if device is None:
        device = next(dnabert2_global_model.parameters()).device
    else:
        model_device = next(dnabert2_global_model.parameters()).device
        if device != model_device:
            dnabert2_global_model = dnabert2_global_model.to(device)
    dnabert2_global_model.eval()

    inputs = dnabert2_global_alphabet(processed_seq, return_tensors='pt')["input_ids"]
    embeddings = {}
    with torch.no_grad():
        # if torch.cuda.is_available():
        inputs = inputs.to(device=device, non_blocking=True)
        try:
            out = dnabert2_global_model(inputs)
            truncate_len = min(truncation_seq_length, inputs.shape[1])
            if "representations" in embedding_type or "matrix" in embedding_type:
                if matrix_add_special_token:
                    embedding = out[0].to(device="cpu")[0, 0: truncate_len + 2].clone().numpy()
                else:
                    embedding = out[0].to(device="cpu")[0, 1: truncate_len + 1].clone().numpy()
                embeddings["representations"] = embedding
            if "bos" in embedding_type or "vector" in embedding_type:
                embedding = out[0].to(device="cpu")[0, 0].clone().numpy()
                embeddings["bos_representations"] = embedding
            if "contacts" in embedding_type:
                # to do
                embeddings["contacts"] = None
            if len(embeddings) > 1:
                return embeddings, processed_seq
            elif len(embeddings) == 1:
                return list(embeddings.items())[0][1], processed_seq
            else:
                return None, None
        except RuntimeError as e:
            if e.args[0].startswith("CUDA out of memory"):
                print(f"Failed (CUDA out of memory) on sequence {sample[0]} of length {len(sample[1])}.")
                print("Please reduce the 'truncation_seq_length'")
            raise Exception(e)
    return None, None
