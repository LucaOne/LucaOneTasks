#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/8/1 22:30
@project: LucaOneTasks
@file: model_config.py
@desc: model config
'''
from transformers.configuration_utils import PretrainedConfig


class LucaConfig(PretrainedConfig):
    def __init__(self,
                 num_labels: int = 2,
                 vocab_size: int = 39,
                 pad_token_id: int = 0,
                 seq_fc_size: int = 1024,
                 vector_fc_size: int = 1024,
                 matrix_fc_size: int = 1024,
                 loss_reduction="mean",
                 max_position_embeddings: int = 2048,
                 type_vocab_size: int = 2,
                 num_hidden_layers: int = 12,
                 directionality="bidi",
                 initializer_range=0.02,
                 intermediate_size=4096,
                 hidden_act="gelu",
                 hidden_size: int = 1024,
                 num_attention_heads: int = 16,
                 no_token_embeddings: bool = False,
                 no_position_embeddings: bool = False,
                 no_token_type_embeddings: bool = False,
                 fc_activate_func="tanh",
                 classifier_activate_func="tanh",
                 classifier_size=1024,
                 alphabet: str = "gene_prot",
                 token_dropout: bool = True,
                 attention_probs_dropout_prob=0.1,
                 hidden_dropout_prob=0.1,
                 classifier_dropout_prob=0.1,
                 ignore_index=-100,
                 pos_weight=1.0,
                 layer_norm_eps=1e-12,
                 position_embedding_type="absolute",
                 self_atten=True,
                 cross_atten=True,
                 use_luca_layer_norm_v2=True,
                 kernel_size=7,
                 **kwargs):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.num_labels = num_labels
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.seq_fc_size = seq_fc_size
        self.vector_fc_size = vector_fc_size
        self.matrix_fc_size = matrix_fc_size
        self.loss_reduction = loss_reduction
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.num_hidden_layers = num_hidden_layers
        self.directionality = directionality
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.no_token_embeddings = no_token_embeddings
        self.no_position_embeddings = no_position_embeddings
        self.no_token_type_embeddings = no_token_type_embeddings
        self.fc_activate_func = fc_activate_func
        self.classifier_size = classifier_size
        self.alphabet = alphabet
        self.token_dropout = token_dropout
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.classifier_dropout_prob = classifier_dropout_prob
        self.ignore_index = ignore_index
        self.pos_weight = pos_weight
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.classifier_activate_func = classifier_activate_func
        self.self_atten = self_atten
        self.cross_atten = cross_atten
        self.use_luca_layer_norm_v2 = use_luca_layer_norm_v2
        self.kernel_size = kernel_size

