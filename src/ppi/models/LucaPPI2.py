#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/6/21 17:32
@project: LucaOneTasks
@file: LucaPPI2
@desc: LucaPPI2 for heterogeneous double sequence
'''

import sys
import logging
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
sys.path.append("../../../src")
try:
    from ....common.pooling import *
    from ....common.loss import *
    from ....utils import *
    from ....common.multi_label_metrics import *
    from ....common.modeling_bert import BertModel, BertPreTrainedModel
    from ....common.metrics import *
except ImportError:
    from src.common.pooling import *
    from src.common.loss import *
    from src.utils import *
    from src.common.multi_label_metrics import *
    from src.common.metrics import *
    from src.common.modeling_bert import BertModel, BertPreTrainedModel
logger = logging.getLogger(__name__)


class LucaPPI2(BertPreTrainedModel):
    def __init__(self, config, args):
        super(LucaPPI2, self).__init__(config)
        # seq, matrix, vector, seq+matrix, seq+vector
        self.input_type = args.input_type
        self.num_labels = config.num_labels
        self.fusion_type = args.fusion_type if hasattr(args, "fusion_type") and args.fusion_type else "concat"
        self.output_mode = args.output_mode
        self.task_level_type = args.task_level_type
        self.prepend_bos = args.prepend_bos
        self.append_eos = args.append_eos
        if self.task_level_type not in ["seq_level"]:
            assert self.input_type not in ["vector", "seq_vector"]
            assert self.fusion_type == "add"

        self.seq_encoder_a, self.seq_pooler_a, self.matrix_encoder_a, self.matrix_pooler_a = None, None, None, None
        self.seq_encoder_b, self.seq_pooler_b, self.matrix_encoder_b, self.matrix_pooler_b = None, None, None, None
        # for input a
        self.encoder_type_list_a = [False, False, False]
        self.input_size_list_a = [0, 0, 0]
        self.linear_idx_a = [-1, -1, -1]
        # for input b
        self.encoder_type_list_b = [False, False, False]
        self.input_size_list_b = [0, 0, 0]
        self.linear_idx_b = [-1, -1, -1]
        if self.input_type == "seq":
            # input_a and input_b both are seq
            # seq -> bert -> (pooler) -> fc * -> classifier
            self.input_size_list_a[0] = config.hidden_size
            config.max_position_embeddings = config.seq_max_length_a
            self.seq_encoder_a = BertModel(config, use_pretrained_embedding=False, add_pooling_layer=(args.seq_pooling_type is None or args.seq_pooling_type == "none") and self.task_level_type in ["seq_level"])
            self.input_size_list_b[0] = config.hidden_size
            config.max_position_embeddings = config.seq_max_length_b
            self.seq_encoder_b = BertModel(config, use_pretrained_embedding=False, add_pooling_layer=(args.seq_pooling_type is None or args.seq_pooling_type == "none") and self.task_level_type in ["seq_level"])
            if self.task_level_type in ["seq_level"]:
                self.seq_pooler_a = create_pooler(pooler_type="seq", config=config, args=args)
                self.seq_pooler_b = create_pooler(pooler_type="seq", config=config, args=args)
            self.encoder_type_list_a[0] = True
            self.encoder_type_list_b[0] = True
            self.linear_idx_a[0] = 0
            self.linear_idx_b[0] = 0
        elif self.input_type == "matrix":
            # input_a and input_b both are embedding matrix
            # emb matrix -> (encoder) - > (pooler) -> fc * -> classifier
            if args.matrix_encoder:
                matrix_encoder_config_a = copy.deepcopy(config)
                matrix_encoder_config_a.no_position_embeddings = True
                matrix_encoder_config_a.no_token_type_embeddings = True
                matrix_encoder_config_a.embedding_input_size = config.embedding_input_size_a
                matrix_encoder_config_a.max_position_embeddings = config.matrix_max_length_a
                matrix_encoder_config_b = copy.deepcopy(config)
                matrix_encoder_config_b.no_position_embeddings = True
                matrix_encoder_config_b.no_token_type_embeddings = True
                matrix_encoder_config_b.embedding_input_size = config.embedding_input_size_b
                matrix_encoder_config_b.max_position_embeddings = config.matrix_max_length_b
                if args.matrix_encoder_act:
                    self.matrix_encoder_a = nn.ModuleList([
                        nn.Linear(config.embedding_input_size_a, config.hidden_size),
                        create_activate(config.emb_activate_func),
                        # nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
                        BertModel(matrix_encoder_config_a, use_pretrained_embedding=True,
                                  add_pooling_layer=(args.matrix_pooling_type is None or args.matrix_pooling_type == "none") and self.task_level_type in ["seq_level"])
                    ])
                    self.matrix_encoder_b = nn.ModuleList([
                        nn.Linear(config.embedding_input_size_b, config.hidden_size),
                        create_activate(config.emb_activate_func),
                        # nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
                        BertModel(matrix_encoder_config_b, use_pretrained_embedding=True,
                                  add_pooling_layer=(args.matrix_pooling_type is None or args.matrix_pooling_type == "none") and self.task_level_type in ["seq_level"])
                    ])

                else:
                    self.matrix_encoder_a = nn.ModuleList([
                        # nn.Linear(config.embedding_input_size, config.hidden_size),
                        # nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
                        BertModel(matrix_encoder_config_a, use_pretrained_embedding=True, add_pooling_layer=(args.matrix_pooling_type is None or args.matrix_pooling_type == "none") and self.task_level_type in ["seq_level"])
                    ])
                    self.matrix_encoder_b = nn.ModuleList([
                        # nn.Linear(config.embedding_input_size, config.hidden_size),
                        # nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
                        BertModel(matrix_encoder_config_b, use_pretrained_embedding=True, add_pooling_layer=(args.matrix_pooling_type is None or args.matrix_pooling_type == "none") and self.task_level_type in ["seq_level"])
                    ])
                ori_embedding_input_size = config.embedding_input_size
                config.embedding_input_size = config.hidden_size
                if self.task_level_type in ["seq_level"]:
                    self.matrix_pooler_a = create_pooler(pooler_type="matrix", config=config, args=args)
                    self.matrix_pooler_b = create_pooler(pooler_type="matrix", config=config, args=args)
                self.input_size_list_a[1] = config.embedding_input_size
                self.input_size_list_b[1] = config.embedding_input_size
                config.embedding_input_size = ori_embedding_input_size
            else:
                self.input_size_list_a[1] = config.embedding_input_size_a
                self.input_size_list_b[1] = config.embedding_input_size_b
                if self.task_level_type in ["seq_level"]:
                    ori_embedding_input_size = config.embedding_input_size
                    config.embedding_input_size = config.embedding_input_size_a
                    self.matrix_pooler_a = create_pooler(pooler_type="matrix", config=config, args=args)
                    config.embedding_input_size = config.embedding_input_size_b
                    self.matrix_pooler_b = create_pooler(pooler_type="matrix", config=config, args=args)
                    config.embedding_input_size = ori_embedding_input_size
            self.encoder_type_list_a[1] = True
            self.encoder_type_list_b[1] = True
            self.linear_idx_a[1] = 0
            self.linear_idx_b[1] = 0
        elif self.input_type == "vector":
            # input_a and input_b both are seq + embedding vector
            # emb vector -> fc * -> classifier
            self.input_size_list_a[2] = config.embedding_input_size_a
            self.encoder_type_list_a[2] = True
            self.linear_idx_a[2] = 0
            self.input_size_list_b[2] = config.embedding_input_size_b
            self.encoder_type_list_b[2] = True
            self.linear_idx_b[2] = 0
        elif self.input_type == "seq_matrix":
            # input_a and input_b both are seq + embedding matrix
            # seq + matrix
            self.input_size_list_a[0] = config.hidden_size
            self.input_size_list_b[0] = config.hidden_size
            config.max_position_embeddings = config.seq_max_length_a
            self.seq_encoder_a = BertModel(config, use_pretrained_embedding=False, add_pooling_layer=(args.seq_pooling_type is None or args.seq_pooling_type == "none") and self.task_level_type in ["seq_level"])
            config.max_position_embeddings = config.seq_max_length_b
            self.seq_encoder_b = BertModel(config, use_pretrained_embedding=False, add_pooling_layer=(args.seq_pooling_type is None or args.seq_pooling_type == "none") and self.task_level_type in ["seq_level"])
            if self.task_level_type in ["seq_level"]:
                ori_embedding_input_size = config.embedding_input_size
                config.embedding_input_size = config.hidden_size
                self.seq_pooler_a = create_pooler(pooler_type="seq", config=config, args=args)
                config.embedding_input_size = config.hidden_size
                self.seq_pooler_b = create_pooler(pooler_type="seq", config=config, args=args)
                config.embedding_input_size = ori_embedding_input_size
            self.encoder_type_list_a[0] = True
            self.encoder_type_list_b[0] = True
            if args.matrix_encoder:
                matrix_encoder_config_a = copy.deepcopy(config)
                matrix_encoder_config_a.no_position_embeddings = True
                matrix_encoder_config_a.no_token_type_embeddings = True
                matrix_encoder_config_a.max_position_embeddings = config.matrix_max_length_a
                matrix_encoder_config_b = copy.deepcopy(config)
                matrix_encoder_config_b.no_position_embeddings = True
                matrix_encoder_config_b.no_token_type_embeddings = True
                matrix_encoder_config_b.max_position_embeddings = config.matrix_max_length_b
                if args.matrix_encoder_act and config.embedding_input_size_a != config.embedding_input_size_b:
                    self.matrix_encoder_a = nn.ModuleList([
                        nn.Linear(config.embedding_input_size_a, config.hidden_size),
                        create_activate(config.emb_activate_func),
                        # nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
                        BertModel(matrix_encoder_config_a, use_pretrained_embedding=True, add_pooling_layer=(args.matrix_pooling_type is None or args.matrix_pooling_type == "none") and self.task_level_type in ["seq_level"])
                    ])
                    self.matrix_encoder_b = nn.ModuleList([
                        nn.Linear(config.embedding_input_size_b, config.hidden_size),
                        create_activate(config.emb_activate_func),
                        # nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
                        BertModel(matrix_encoder_config_b, use_pretrained_embedding=True, add_pooling_layer=(args.matrix_pooling_type is None or args.matrix_pooling_type == "none") and self.task_level_type in ["seq_level"])
                    ])
                else:
                    self.matrix_encoder_a = nn.ModuleList([
                        # nn.Linear(config.embedding_input_size, config.hidden_size),
                        # nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
                        BertModel(matrix_encoder_config_a, use_pretrained_embedding=True, add_pooling_layer=(args.matrix_pooling_type is None or args.matrix_pooling_type == "none") and self.task_level_type in ["seq_level"])
                    ])
                    self.matrix_encoder_b = nn.ModuleList([
                        # nn.Linear(config.embedding_input_size, config.hidden_size),
                        # nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
                        BertModel(matrix_encoder_config_b, use_pretrained_embedding=True, add_pooling_layer=(args.matrix_pooling_type is None or args.matrix_pooling_type == "none") and self.task_level_type in ["seq_level"])
                    ])
                ori_embedding_input_size = config.embedding_input_size
                config.embedding_input_size = config.hidden_size
                if self.task_level_type in ["seq_level"]:
                    self.matrix_pooler_a = create_pooler(pooler_type="matrix", config=config, args=args)
                    self.matrix_pooler_b = create_pooler(pooler_type="matrix", config=config, args=args)
                self.input_size_list_a[1] = config.hidden_size
                self.input_size_list_b[1] = config.hidden_size
                config.embedding_input_size = ori_embedding_input_size
            else:
                self.input_size_list_a[1] = config.embedding_input_size_a
                self.input_size_list_b[1] = config.embedding_input_size_b
                if self.task_level_type in ["seq_level"]:
                    ori_embedding_input_size = config.embedding_input_size
                    config.embedding_input_size = config.embedding_input_size_a
                    self.matrix_pooler_a = create_pooler(pooler_type="matrix", config=config, args=args)
                    config.embedding_input_size = config.embedding_input_size_b
                    self.matrix_pooler_b = create_pooler(pooler_type="matrix", config=config, args=args)
                    config.embedding_input_size = ori_embedding_input_size
            self.encoder_type_list_a[1] = True
            self.encoder_type_list_b[1] = True
            self.linear_idx_a[0] = 0
            self.linear_idx_b[0] = 0
            self.linear_idx_a[1] = 1
            self.linear_idx_b[1] = 1
        elif self.input_type == "seq_vector":
            # input_a and input_b both are seq + embedding vector
            # seq + vector
            self.input_size_list_a[0] = config.hidden_size
            self.input_size_list_b[0] = config.hidden_size
            config.max_position_embeddings = config.seq_max_length_a
            self.seq_encoder_a = BertModel(config, use_pretrained_embedding=False, add_pooling_layer=(args.seq_pooling_type is None or args.seq_pooling_type == "none") and self.task_level_type in ["seq_level"])
            config.max_position_embeddings = config.seq_max_length_b
            self.seq_encoder_b = BertModel(config, use_pretrained_embedding=False, add_pooling_layer=(args.seq_pooling_type is None or args.seq_pooling_type == "none") and self.task_level_type in ["seq_level"])
            config.embedding_input_size = config.hidden_size
            self.seq_pooler_a = create_pooler(pooler_type="seq", config=config, args=args)
            self.seq_pooler_b = create_pooler(pooler_type="seq", config=config, args=args)
            self.encoder_type_list_a[0] = True
            self.encoder_type_list_b[0] = True
            self.input_size_list_a[2] = config.embedding_input_size_a
            self.input_size_list_b[2] = config.embedding_input_size_b
            self.encoder_type_list_a[2] = True
            self.encoder_type_list_b[2] = True
            self.linear_idx_a[0] = 0
            self.linear_idx_b[0] = 0
            self.linear_idx_a[2] = 1
            self.linear_idx_b[2] = 1
        else:
            raise Exception("Not support input_type=%s" % self.input_type)
        fc_size_list = [config.seq_fc_size, config.matrix_fc_size, config.vector_fc_size]
        all_linear_list_a = [None, None, None]
        all_linear_list_b = [None, None, None]
        self.output_size_a = [0, 0, 0]
        self.output_size_b = [0, 0, 0]
        print("self.encoder_type_list:", self.encoder_type_list_a)
        print("self.encoder_type_list:", self.encoder_type_list_b)
        for encoder_idx, encoder_flag in enumerate(self.encoder_type_list_a):
            if not encoder_flag:
                continue
            fc_size = fc_size_list[encoder_idx]
            input_size_a = self.input_size_list_a[encoder_idx]
            input_size_b = self.input_size_list_b[encoder_idx]
            print("encoder_idx:", encoder_idx, ", input_size_a:", input_size_a, ", input_size_b:", input_size_b)
            if fc_size is not None and len(fc_size) > 0:
                if isinstance(fc_size, list):
                    fc_size = [int(v) for v in fc_size]
                else:
                    fc_size = [int(fc_size)]
                linear_list_a = []
                linear_list_b = []
                for idx in range(len(fc_size)):
                    linear_a = nn.Linear(input_size_a, fc_size[idx])
                    linear_list_a.append(linear_a)
                    linear_list_a.append(create_activate(config.fc_activate_func))
                    input_size_a = fc_size[idx]
                    linear_b = nn.Linear(input_size_b, fc_size[idx])
                    linear_list_b.append(linear_b)
                    linear_list_b.append(create_activate(config.fc_activate_func))
                    input_size_b = fc_size[idx]
                all_linear_list_a[encoder_idx] = nn.ModuleList(linear_list_a)
                all_linear_list_b[encoder_idx] = nn.ModuleList(linear_list_b)
                self.output_size_a[encoder_idx] = fc_size[-1]
                self.output_size_b[encoder_idx] = fc_size[-1]
            else: # 没有全连接层
                self.linear_idx_a[encoder_idx] = -1
                self.linear_idx_b[encoder_idx] = -1
                self.output_size_a[encoder_idx] = input_size_a
                self.output_size_b[encoder_idx] = input_size_b
        all_linear_list_a = [linear for linear in all_linear_list_a if linear is not None]
        all_linear_list_b = [linear for linear in all_linear_list_b if linear is not None]
        if all_linear_list_a is not None and len(all_linear_list_a) > 0:
            self.linear_a = nn.ModuleList(all_linear_list_a)
        if all_linear_list_b is not None and len(all_linear_list_b) > 0:
            self.linear_b = nn.ModuleList(all_linear_list_b)
        if self.fusion_type == "add":
            output_size_a = [v for v in self.output_size_a if v > 0]
            assert len(set(output_size_a)) == 1
            last_hidden_size_a = output_size_a[0]
            output_size_b = [v for v in self.output_size_b if v > 0]
            assert len(set(output_size_b)) == 1
            last_hidden_size_b = output_size_b[0]
        else:
            last_hidden_size_a = sum(self.output_size_a)
            last_hidden_size_b = sum(self.output_size_b)
        last_hidden_size = last_hidden_size_a + last_hidden_size_b
        self.dropout, self.hidden_layer, self.hidden_act, self.classifier, self.output, self.loss_fct = \
            create_loss_function(config,
                                 args,
                                 hidden_size=last_hidden_size,
                                 classifier_size=args.classifier_size,
                                 sigmoid=args.sigmoid,
                                 output_mode=args.output_mode,
                                 num_labels=self.num_labels,
                                 loss_type=args.loss_type,
                                 ignore_index=args.ignore_index,
                                 return_types=["dropout", "hidden_layer", "hidden_act", "classifier", "output", "loss"])
        self.post_init()

    def __forward_a__(
            self,
            input_ids,
            seq_attention_masks,
            token_type_ids,
            position_ids,
            vectors,
            matrices,
            matrix_attention_masks
    ):
        if input_ids is not None and self.seq_encoder_a is not None:
            seq_outputs = self.seq_encoder_a(
                input_ids,
                attention_mask=seq_attention_masks,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=False
            )
            if self.append_eos:
                seq_index = torch.sum(seq_attention_masks, dim=1, keepdim=True) - 1
                seq_attention_masks = seq_attention_masks.scatter(1, seq_index, 0)

            if self.prepend_bos:
                seq_attention_masks[:, 0] = 0

            if self.seq_pooler_a is not None:
                seq_vector = self.seq_pooler_a(seq_outputs[0], mask=seq_attention_masks)
            elif self.task_level_type in ["seq_level"]:
                seq_vector = seq_outputs[1]
            else:
                seq_vector = seq_outputs[0]
            seq_linear_idx = self.linear_idx_a[0]
            if seq_linear_idx != -1:
                for i, layer_module in enumerate(self.linear_a[seq_linear_idx]):
                    seq_vector = layer_module(seq_vector)
        if vectors is not None:
            vector_vector = vectors
            vector_linear_idx = self.linear_idx_a[2]
            if vector_linear_idx != -1:
                for i, layer_module in enumerate(self.linear_a[vector_linear_idx]):
                    vector_vector = layer_module(vector_vector)
        if matrices is not None:
            if self.matrix_encoder_a is not None:
                for module in self.matrix_encoder_a[:-1]:
                    matrices = module(matrices)
                matrices_output = self.matrix_encoder_a[-1](
                    input_ids=None,
                    attention_mask=matrix_attention_masks,
                    token_type_ids=None,
                    position_ids=None,
                    head_mask=None,
                    inputs_embeds=matrices,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=False
                )
                matrices = matrices_output[0]

            if self.matrix_pooler_a is not None:
                matrix_vector = self.matrix_pooler_a(matrices, mask=matrix_attention_masks)
            elif self.task_level_type in ["seq_level"]:
                tmp_mask = torch.unsqueeze(matrix_attention_masks, dim=-1)
                matrices = matrices.masked_fill(tmp_mask == 0, 0.0)
                # 均值pooling
                matrix_vector = torch.sum(matrices, dim=1)/(torch.sum(tmp_mask, dim=1) + 1e-12)
            else:
                matrix_vector = matrices
            matrix_linear_idx = self.linear_idx_a[1]
            if matrix_linear_idx != -1:
                for i, layer_module in enumerate(self.linear_a[matrix_linear_idx]):
                    matrix_vector = layer_module(matrix_vector)

        if self.input_type == "seq":
            concat_vector = seq_vector
        elif self.input_type == "matrix":
            concat_vector = matrix_vector
        elif self.input_type == "vector":
            concat_vector = vector_vector
        elif self.input_type == "seq_matrix":
            if self.fusion_type == "add":
                concat_vector = torch.add(seq_vector, matrix_vector)
            else:
                concat_vector = torch.cat([seq_vector, matrix_vector], dim=-1)
        elif self.input_type == "seq_vector":
            if self.fusion_type == "add":
                concat_vector = torch.add(seq_vector, vector_vector)
            else:
                concat_vector = torch.cat([seq_vector, vector_vector], dim=-1)
        else:
            raise Exception("Not support input_type=%s" % self.input_type)
        return concat_vector

    def __forward_b__(self,
                       input_ids,
                       seq_attention_masks,
                       token_type_ids,
                       position_ids,
                       vectors,
                       matrices,
                       matrix_attention_masks):
        if input_ids is not None and self.seq_encoder_b is not None:
            seq_outputs = self.seq_encoder_b(
                input_ids,
                attention_mask=seq_attention_masks,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=False
            )
            if self.append_eos:
                seq_index = torch.sum(seq_attention_masks, dim=1, keepdim=True) - 1
                seq_attention_masks = seq_attention_masks.scatter(1, seq_index, 0)

            if self.prepend_bos:
                seq_attention_masks[:, 0] = 0

            if self.seq_pooler_b is not None:
                seq_vector = self.seq_pooler_b(seq_outputs[0], mask=seq_attention_masks)
            elif self.task_level_type in ["seq_level"]:
                seq_vector = seq_outputs[1]
            else:
                seq_vector = seq_outputs[0]
            seq_linear_idx = self.linear_idx_b[0]
            if seq_linear_idx != -1:
                for i, layer_module in enumerate(self.linear_b[seq_linear_idx]):
                    seq_vector = layer_module(seq_vector)
        if vectors is not None:
            vector_vector = vectors
            vector_linear_idx = self.linear_idx_b[2]
            if vector_linear_idx != -1:
                for i, layer_module in enumerate(self.linear_b[vector_linear_idx]):
                    vector_vector = layer_module(vector_vector)
        if matrices is not None:
            if self.matrix_encoder_b is not None:
                for module in self.matrix_encoder_b[:-1]:
                    matrices = module(matrices)
                matrices_output = self.matrix_encoder_b[-1](
                    input_ids=None,
                    attention_mask=matrix_attention_masks,
                    token_type_ids=None,
                    position_ids=None,
                    head_mask=None,
                    inputs_embeds=matrices,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=False
                )
                matrices = matrices_output[0]

            if self.matrix_pooler_b is not None:
                matrix_vector = self.matrix_pooler_b(matrices, mask=matrix_attention_masks)
            elif self.task_level_type in ["seq_level"]:
                tmp_mask = torch.unsqueeze(matrix_attention_masks, dim=-1)
                matrices = matrices.masked_fill(tmp_mask == 0, 0.0)
                # 均值pooling
                matrix_vector = torch.sum(matrices, dim=1)/(torch.sum(tmp_mask, dim=1) + 1e-12)
            else:
                matrix_vector = matrices
            matrix_linear_idx = self.linear_idx_b[1]
            if matrix_linear_idx != -1:
                for i, layer_module in enumerate(self.linear_b[matrix_linear_idx]):
                    matrix_vector = layer_module(matrix_vector)

        if self.input_type == "seq":
            concat_vector = seq_vector
        elif self.input_type == "matrix":
            concat_vector = matrix_vector
        elif self.input_type == "vector":
            concat_vector = vector_vector
        elif self.input_type == "seq_matrix":
            if self.fusion_type == "add":
                concat_vector = torch.add(seq_vector, matrix_vector)
            else:
                concat_vector = torch.cat([seq_vector, matrix_vector], dim=-1)
        elif self.input_type == "seq_vector":
            if self.fusion_type == "add":
                concat_vector = torch.add(seq_vector, vector_vector)
            else:
                concat_vector = torch.cat([seq_vector, vector_vector], dim=-1)
        else:
            raise Exception("Not support input_type=%s" % self.input_type)
        return concat_vector

    def forward(self,
                input_ids_a=None, input_ids_b=None,
                position_ids_a=None, position_ids_b=None,
                token_type_ids_a=None, token_type_ids_b=None,
                seq_attention_masks_a=None, seq_attention_masks_b=None,
                vectors_a=None, vectors_b=None,
                matrices_a=None, matrices_b=None,
                matrix_attention_masks_a=None, matrix_attention_masks_b=None,
                labels=None
                ):
        representation_vector_a = self.__forward_a__(input_ids_a,
                                                    seq_attention_masks_a,
                                                    token_type_ids_a,
                                                    position_ids_a,
                                                    vectors_a,
                                                    matrices_a,
                                                    matrix_attention_masks_a)
        representation_vector_b = self.__forward_b__(input_ids_b,
                                                    seq_attention_masks_b,
                                                    token_type_ids_b,
                                                    position_ids_b,
                                                    vectors_b,
                                                    matrices_b,
                                                    matrix_attention_masks_b)

        concat_vector = torch.cat([representation_vector_a, representation_vector_b], dim=-1)
        if self.dropout is not None:
            concat_vector = self.dropout(concat_vector)
        if self.hidden_layer is not None:
            concat_vector = self.hidden_layer(concat_vector)
        if self.hidden_act is not None:
            concat_vector = self.hidden_act(concat_vector)
        logits = self.classifier(concat_vector)
        if self.output:
            output = self.output(logits)
        else:
            output = logits
        outputs = [logits, output]

        if labels is not None:
            if self.output_mode in ["regression"]:
                if self.task_level_type not in ["seq_level"] and self.loss_reduction == "meanmean":
                    # logits: N, seq_len, 1
                    # labels: N, seq_len
                    loss = self.loss_fct(logits, labels)
                else:
                    # logits: N * seq_len
                    # labels: N * seq_len
                    loss = self.loss_fct(logits.view(-1), labels.view(-1))
            elif self.output_mode in ["multi_label", "multi-label"]:
                if self.loss_reduction == "meanmean":
                    # logits: N , label_size
                    # labels: N , label_size
                    loss = self.loss_fct(logits, labels.float())
                else:
                    # logits: N , label_size
                    # labels: N , label_size
                    loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels).float())
            elif self.num_labels <= 2 or self.output_mode in ["binary_class", "binary-class"]:
                if self.task_level_type not in ["seq_level"] and self.loss_reduction == "meanmean":
                    # logits: N ,seq_len, 1
                    # labels: N, seq_len
                    loss = self.loss_fct(logits, labels.float())
                else:
                    # logits: N * seq_len * 1
                    # labels: N * seq_len
                    loss = self.loss_fct(logits.view(-1), labels.view(-1).float())
            elif self.output_mode in ["multi_class", "multi-class"]:
                if self.task_level_type not in ["seq_level"] and self.loss_reduction == "meanmean":
                    # logits: N ,seq_len, label_size
                    # labels: N , seq_len
                    loss = self.loss_fct(logits, labels)
                else:
                    # logits: N * seq_len, label_size
                    # labels: N * seq_len
                    loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = [loss, *outputs]
        return outputs



