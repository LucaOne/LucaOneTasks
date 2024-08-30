#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.**@**.com
@tel: 137****6540
@datetime: 2023/3/20 13:23
@project: LucaOneTasks
@file: predict_embedding
@desc: embedding inference for ESM2
'''

import os
import sys
import esm
import torch
import random, argparse
from timeit import default_timer as timer
from esm import BatchConverter, pretrained
from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel
from torch.distributed.fsdp.wrap import enable_wrap, wrap
sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../src")
try:
    from file_operator import fasta_reader, csv_reader
    from utils import clean_seq, available_gpu_id, calc_emb_filename_by_seq_id
except ImportError:
    from src.file_operator import fasta_reader, csv_reader
    from src.utils import clean_seq, available_gpu_id, calc_emb_filename_by_seq_id


def enable_cpu_offloading(model):
    torch.distributed.init_process_group(
        backend="nccl", init_method="tcp://localhost:%d" % (7000 + random.randint(0, 1000)), world_size=1, rank=0
    )
    wrapper_kwargs = dict(cpu_offload=CPUOffload(offload_params=True))

    with enable_wrap(wrapper_cls=FullyShardedDataParallel, **wrapper_kwargs):
        for layer_name, layer in model.layers.named_children():
            wrapped_layer = wrap(layer)
            setattr(model.layers, layer_name, wrapped_layer)
        model = wrap(model)

    return model


def init_model_on_gpu_with_cpu_offloading(model):
    model = model.eval()
    model_esm = enable_cpu_offloading(model.esm)
    del model.esm
    model.cuda()
    model.esm = model_esm
    return model


def predict_pdb(sample, trunc_type, num_recycles=4, truncation_seq_length=4096, chunk_size=64, cpu_type="cpu-offload"):
    '''
    use sequence to predict protein 3D-structure
    :param sample:
    :param trunc_type:
    :param num_recycles:
    :param truncation_seq_length:
    :param chunk_size:
    :param cpu_type:
    :return: pdb, mean_plddt, ptm, processed_seq_len
    '''
    assert cpu_type is None or cpu_type in ["cpu-offload", "cpu-only"]
    model = esm.pretrained.esmfold_v1()
    model = model.eval()
    model.set_chunk_size(chunk_size)
    if cpu_type == "cpu_only":
        model.esm.float()  # convert to fp32 as ESM-2 in fp16 is not supported on CPU
        model.cpu()
    elif cpu_type == "cpu_offload":
        model = init_model_on_gpu_with_cpu_offloading(model)
    else:
        model.cuda()
    start = timer()
    protein_id, protein_seq = sample[0], sample[1]
    if len(protein_seq) > truncation_seq_length:
        if trunc_type == "left":
            protein_seq = protein_seq[-truncation_seq_length:]
        else:
            protein_seq = protein_seq[:truncation_seq_length]
    cur_seq_len = len(protein_seq)
    processed_seq = protein_seq[:truncation_seq_length] if cur_seq_len > truncation_seq_length else protein_seq
    with torch.no_grad():
        try:
            output = model.infer([processed_seq], num_recycles=num_recycles)
            output = {key: value.cpu() for key, value in output.items()}
            mean_plddt = output["mean_plddt"][0]
            ptm = output["ptm"][0]
            pdb = model.output_to_pdb(output)[0]
            use_time = timer() - start
            print("predict pdb use time: %f" % use_time)
            return pdb, mean_plddt, ptm, processed_seq
        except RuntimeError as e:
            if e.args[0].startswith("CUDA out of memory"):
                print(f"Failed (CUDA out of memory) on sequence {sample[0]} of length {len(sample[1])}.")
            else:
                print(e)
    return None, None, None, None


esm_global_model, esm_global_alphabet, esm_global_version, esm_global_layer_size = None, None, None, None


def predict_embedding(sample, trunc_type, embedding_type, repr_layers=[-1], truncation_seq_length=4094, device=None, version="3B", matrix_add_special_token=False):
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
    global esm_global_model, esm_global_alphabet, esm_global_version, esm_global_layer_size
    assert "bos" in embedding_type or "representations" in embedding_type \
           or "matrix" in embedding_type or "vector" in embedding_type or "contacts" in embedding_type
    if len(sample) > 2:
        protein_id, protein_seq = sample[0], sample[2]
    else:
        protein_id, protein_seq = sample[0], sample[1]
    protein_seq = clean_seq(protein_id, protein_seq)
    if len(protein_seq) > truncation_seq_length:
        if trunc_type == "left":
            protein_seq = protein_seq[-truncation_seq_length:]
        else:
            protein_seq = protein_seq[:truncation_seq_length]
    if esm_global_model is None or esm_global_alphabet is None or esm_global_version is None or esm_global_version != version or esm_global_layer_size is None:
        if version == "15B":
            llm_name = "esm2_t48_15B_UR50D"
            esm_global_model, esm_global_alphabet = pretrained.load_model_and_alphabet("esm2_t48_15B_UR50D")
            esm_global_layer_size = 48
        elif version == "3B":
            llm_name = "esm2_t36_3B_UR50D"
            esm_global_model, esm_global_alphabet = pretrained.load_model_and_alphabet("esm2_t36_3B_UR50D")
            esm_global_layer_size = 36
        elif version == "650M":
            llm_name = "esm2_t33_650M_UR50D"
            esm_global_model, esm_global_alphabet = pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
            esm_global_layer_size = 33
        elif version == "150M":
            llm_name = "esm2_t30_150M_UR50D"
            esm_global_model, esm_global_alphabet = pretrained.load_model_and_alphabet("esm2_t30_150M_UR50D")
            esm_global_layer_size = 30
        else:
            raise Exception("not support this version=%s" % version)
        print("LLM: %s, version: %s, layer_idx: %d, device: %s" % (llm_name, version, esm_global_layer_size, str(device)))
        esm_global_version = version
    '''
    if torch.cuda.is_available() and device is not None:
        esm_global_model = esm_global_model.to(device)
    elif torch.cuda.is_available():
        esm_global_model = esm_global_model.cuda()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("llm use cpu")
    '''
    if device is None:
        device = next(esm_global_model.parameters()).device
    else:
        model_device = next(esm_global_model.parameters()).device
        if device != model_device:
            esm_global_model = esm_global_model.to(device)
    # print("llm device:", device)
    assert all(-(esm_global_model.num_layers + 1) <= i <= esm_global_model.num_layers for i in repr_layers)
    repr_layers = [(i + esm_global_model.num_layers + 1) % (esm_global_model.num_layers + 1) for i in repr_layers]
    esm_global_model.eval()

    converter = BatchConverter(esm_global_alphabet, truncation_seq_length)
    protein_ids, raw_seqs, tokens = converter([[protein_id, protein_seq]])
    embeddings = {}
    with torch.no_grad():
        # if torch.cuda.is_available():
        tokens = tokens.to(device=device, non_blocking=True)
        try:
            out = esm_global_model(tokens, repr_layers=repr_layers, return_contacts=False)
            truncate_len = min(truncation_seq_length, len(raw_seqs[0]))
            if "representations" in embedding_type or "matrix" in embedding_type:
                if matrix_add_special_token:
                    embedding = out["representations"][esm_global_layer_size].to(device="cpu")[0, 0: truncate_len + 2].clone().numpy()
                else:
                    embedding = out["representations"][esm_global_layer_size].to(device="cpu")[0, 1: truncate_len + 1].clone().numpy()
                embeddings["representations"] = embedding
            if "bos" in embedding_type or "vector" in embedding_type:
                embedding = out["representations"][esm_global_layer_size].to(device="cpu")[0, 0].clone().numpy()
                embeddings["bos_representations"] = embedding
            if "contacts" in embedding_type:
                embedding = out["contacts"][esm_global_layer_size].to(device="cpu")[0, :, :].clone().numpy()
                embeddings["contacts"] = embedding
            if len(embeddings) > 1:
                return embeddings, protein_seq
            elif len(embeddings) == 1:
                return list(embeddings.items())[0][1], protein_seq
            else:
                return None, None
        except RuntimeError as e:
            if e.args[0].startswith("CUDA out of memory"):
                print(f"Failed (CUDA out of memory) on sequence {sample[0]} of length {len(sample[1])}.")
                print("Please reduce the 'truncation_seq_length'")
            raise Exception(e)
    return None, None


def get_args():
    parser = argparse.ArgumentParser(description='ESM/SRM Embedding')
    # for logging
    parser.add_argument("--llm_type", type=str, default="lucaone_gplm", choices=["esm", "ESM", "lucaone_gplm"],  help="llm type")
    parser.add_argument("--llm_version", type=str, default="3B", choices=["15B", "3B", "650M", "150M"], help="llm version")
    parser.add_argument("--embedding_type", type=str, default="matrix", choices=["matrix", "vector", "contact"], help="llm embedding type.")
    parser.add_argument("--trunc_type", type=str, default="right", choices=["left", "right"], help="llm trunc type of seq.")
    parser.add_argument("--truncation_seq_length", type=int, default=4094, help="truncation seq length.")
    parser.add_argument('--gpu', type=int, default=-1, help="gpu idx.")
    parser.add_argument("--input_file", type=str, default=None, help="the input filepath(.fasta or .csv)")
    parser.add_argument("--seq", type=str, default=None, help="the input seq")
    parser.add_argument("--seq_type", type=str, default=None, required=True, choices=["gene", "prot"], help="seq type")
    parser.add_argument("--save_path", type=str, default=None, help="embedding file save path")
    parser.add_argument("--matrix_add_special_token", action="store_true", help="Whether to add special token embedding in seq representation matrix")
    input_args = parser.parse_args()
    return input_args


def main(args):
    if args.gpu >= 0:
        gpu_id = args.gpu
    else:
        gpu_id = available_gpu_id()
        print("gpu_id: ", gpu_id)
    if gpu_id is None or gpu_id == -1:
        args.device = None
    else:
        args.device = torch.device("cuda:%d" % gpu_id if gpu_id > -1 else "cpu")
    # model.to(args.device)
    assert args.input_file is not None or args.seq is not None
    print("input seq type: %s" % args.seq_type)
    print("args device: %s" % args.device)
    embedding_type = args.embedding_type
    save_path = args.save_path
    seq_type = args.seq_type
    emb_save_path = os.path.join(save_path, args.llm_type, args.llm_version)
    print("emb save dir: %s" % emb_save_path)
    if seq_type not in ["gene", "prot"]:
        seq_type = "prot"
    if not os.path.exists(emb_save_path):
        os.makedirs(emb_save_path)

    if args.input_file:
        done = 0
        file_reader = fasta_reader
        if args.input_file.endswith(".csv"):
            file_reader = csv_reader

        for row in file_reader(args.input_file):
            if args.id_idx is None or args.seq_idx is None:
                if len(row) > 2:
                    seq_id, seq = row[0].strip(), row[2].upper()
                else:
                    seq_id, seq = row[0].strip(), row[1].upper()
            else:
                seq_id, seq = row[args.id_idx].strip(), row[args.seq_idx].upper()
            emb_filename = calc_emb_filename_by_seq_id(seq_id=seq_id, embedding_type=embedding_type)
            embedding_filepath = os.path.join(emb_save_path, emb_filename)
            emb, processed_seq_len = predict_embedding([seq_id, seq_type, seq],
                                                       args.trunc_type,
                                                       embedding_type,
                                                       repr_layers=[-1],
                                                       truncation_seq_length=args.truncation_seq_length,
                                                       device=args.device,
                                                       version=args.llm_version,
                                                       matrix_add_special_token=args.matrix_add_special_token
                                                       )
            # print("seq_len: %d" % len(seq))
            # print("emb shape:", embedding_info.shape)
            torch.save(emb, embedding_filepath)
            done += 1
            if done % 1000 == 0:
                print("embedding done: %d" % done)
        print("embedding over, done: %d" % done)
    elif args.seq:
        print("input seq length: %d" % len(args.seq))
        emb, processed_seq_len = predict_embedding([args.seq_id, seq_type, args.seq],
                                                   args.trunc_type,
                                                   embedding_type,
                                                   repr_layers=[-1],
                                                   truncation_seq_length=args.truncation_seq_length,
                                                   device=args.device,
                                                   version=args.llm_version,
                                                   matrix_add_special_token=args.matrix_add_special_token)
        print("done seq length: %d" % processed_seq_len)
        print(emb)
        print(emb.shape)


if __name__ == "__main__":
    args = get_args()
    main(args)




