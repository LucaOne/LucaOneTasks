#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/7/24 11:00
@project: LucaOneTasks
@file: alphabet.py
@desc: alphabet for DNA, RNA, Protein
'''
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Sequence, List


atom_standard_toks = ['C', 'N', 'O', 'S', 'H', 'Cl', 'F', 'Br', 'I',
                      'Si', 'P', 'B', 'Na', 'K', 'Al', 'Ca', 'Sn', 'As',
                      'Hg', 'Fe', 'Zn', 'Cr', 'Se', 'Gd', 'Au', 'Li'
                      ]

atom_prepend_toks = ['[PAD]', '[UNK]', '[CLS]']

atom_append_toks = ['[SEP]', '[MASK]']


class AlphabetAtom(object):
    def __init__(
            self,
            standard_toks: Sequence[str] = atom_standard_toks,
            prepend_toks: Sequence[str] = atom_prepend_toks,
            append_toks: Sequence[str] = atom_append_toks,
            prepend_bos: bool = True,
            append_eos: bool = True
    ):
        self.standard_toks = list(standard_toks)
        self.prepend_toks = list(prepend_toks)
        self.append_toks = list(append_toks)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos

        self.all_toks = list(self.prepend_toks)
        self.all_toks.extend(self.append_toks)
        self.all_toks.extend(self.standard_toks)

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

        self.unk_idx = self.tok_to_idx["[UNK]"]
        self.padding_idx = self.get_idx("[PAD]")
        self.pad_idx = self.get_idx("[PAD]")
        self.pad_token_id = self.padding_idx
        self.cls_idx = self.get_idx("[CLS]")
        self.mask_idx = self.get_idx("[MASK]")
        self.eos_idx = self.get_idx("[SEP]")
        self.all_special_tokens = prepend_toks + append_toks
        self.all_special_token_idx_list = [self.tok_to_idx[v] for v in self.all_special_tokens]
        self.unique_no_split_tokens = self.all_toks
        self.vocab_size = self.__len__()

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind):
        return self.all_toks[ind]

    def to_dict(self):
        return self.tok_to_idx.copy()

    def get_batch_converter(self, task_level_type, label_size, output_mode, no_position_embeddings,
                            no_token_type_embeddings, truncation_seq_length: int = None, ignore_index: int = -100, mlm_probability=0.15):
        '''
        return BatchConverter(
            task_level_type,
            label_size,
            output_mode,
            seq_subword=False,
            seq_tokenizer=self,
            no_position_embeddings=no_position_embeddings,
            no_token_type_embeddings=no_token_type_embeddings,
            truncation_seq_length=truncation_seq_length,
            truncation_matrix_length=truncation_seq_length,
            ignore_index=ignore_index,
            mlm_probability=mlm_probability,
            prepend_bos=self.prepend_bos,
            append_eos=self.append_eos)
        '''
        pass

    @classmethod
    def smiles_2_atom_seq(cls, smi):
        mol = Chem.MolFromSmiles(smi)
        mol = AllChem.AddHs(mol)
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]  # after add H
        return atoms

    @classmethod
    def from_predefined(cls, name: str = "atom_v1"):
        if name.lower() == "atom_v1":
            standard_toks = atom_standard_toks
        else:
            raise Exception("Not support tokenizer name: %s" % name)

        prepend_toks = atom_prepend_toks
        append_toks = atom_append_toks
        prepend_bos = True
        append_eos = True

        return cls(standard_toks, prepend_toks, append_toks, prepend_bos, append_eos)

    @classmethod
    def from_pretrained(cls, dir_path):
        import os, pickle
        return pickle.load(open(os.path.join(dir_path, "alphabet_atom.pkl"), "rb"))

    def save_pretrained(self, save_dir):
        import os, pickle
        with open(os.path.join(save_dir, "alphabet_atom.pkl"), 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    def save_pretrained(self, save_dir):
        import os, pickle
        with open(os.path.join(save_dir, "alphabet_atom.pkl"), 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    def tokenize(self, smi, prepend_bos, append_eos) -> List[str]:
        seq = AlphabetAtom.smiles_2_atom_seq(smi)
        if prepend_bos:
            seq = [self.get_tok(self.cls_idx)] + seq
        if append_eos:
            seq = seq + [self.get_tok(self.eos_idx)]
        return seq

    def encode(self, atom_list, prepend_bos, append_eos):
        idx_list = [self.get_idx(tok) for tok in atom_list]
        if prepend_bos:
            idx_list = [self.cls_idx] + idx_list
        if append_eos:
            idx_list = idx_list + [self.eos_idx]
        return idx_list

    def encode_smi(self, smi, prepend_bos, append_eos):
        atom_list = self.smiles_2_atom_seq(smi)
        return self.encode(atom_list, prepend_bos, append_eos)


if __name__ == "__main__":
    print("std len: %d" % len(atom_standard_toks))
    obj = AlphabetAtom.from_predefined("atom_v1")
    print("std len: %d" % len(obj.all_toks))
    toks = obj.tokenize("Cc1nc(CN2CCN(c3c(Cl)cnc4[nH]c(-c5cn(C)nc5C)nc34)CC2)no1", True, True)
    print(len(toks))
    print(toks)
    ids = obj.encode(obj.tokenize("Cc1nc(CN2CCN(c3c(Cl)cnc4[nH]c(-c5cn(C)nc5C)nc34)CC2)no1", False, False), True, True)
    print(len(ids))
    print(ids)
    ids = obj.encode_smi("Cc1nc(CN2CCN(c3c(Cl)cnc4[nH]c(-c5cn(C)nc5C)nc34)CC2)no1", True, True)
    print(len(ids))
    print(ids)

    ids = obj.tokenize("Cc1nc(CN2CCN(c3c(Cl)cnc4[nH]c(-c5cn(C)nc5C)nc34)CC2)no1", False, False)
    print(len(ids))
    print(ids)
