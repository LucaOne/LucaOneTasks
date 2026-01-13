#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/6/21 17:32
@project: LucaOneTasks
@file: LucaPairHomo
@desc: LucaPairHomo for homogeneous pair input
'''

import sys
import logging
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
sys.path.append("../../../src")
try:
    from common.pooling import *
    from common.loss import *
    from utils import *
    from common.multi_label_metrics import *
    from common.metrics import *
    from common.modeling_bert import BertPreTrainedModel
    from common.cross_transformer import LucaTransformer
except ImportError:
    from src.common.pooling import *
    from src.common.loss import *
    from src.utils import *
    from src.common.multi_label_metrics import *
    from src.common.metrics import *
    from src.common.modeling_bert import BertPreTrainedModel
    from src.common.cross_transformer import LucaTransformer
logger = logging.getLogger(__name__)


class LucaSingle(BertPreTrainedModel):
    def __init__(self, config, args):
        super(LucaSingle, self).__init__(config)
        # seq, matrix, vector, matrix_express, matrix_variant
        self.input_type = args.input_type
        self.output_mode = args.output_mode
        self.num_labels = config.num_labels
        self.fusion_type = args.fusion_type
        self.task_level_type = args.task_level_type
        self.prepend_bos = args.prepend_bos
        self.append_eos = args.append_eos
        self.loss_reduction = args.loss_reduction

        if self.task_level_type not in ["seq_level"]:
            assert self.input_type not in ["vector", "seq_vector"]
            assert self.fusion_type == "add"
        self.seq_encoder, self.seq_pooler, self.matrix_encoder, self.matrix_pooler = None, None, None, None
        self.encoder_type_list = [False, False, False]
        self.input_size_list = [0, 0, 0]
        self.linear_idx = [-1, -1, -1]
        if self.input_type == "seq":
            # seq -> bert -> (pooler) -> fc * -> classifier
            self.input_size_list[0] = config.hidden_size
            config.max_position_embeddings = config.seq_max_length
            self.seq_encoder = LucaTransformer(
                config,
                use_pretrained_embedding=False,
                add_pooling_layer=(args.seq_pooling_type is None or args.seq_pooling_type == "none") and self.task_level_type in ["seq_level"]
            )
            if self.task_level_type in ["seq_level"]:
                self.seq_pooler = create_pooler(pooler_type="seq", config=config, args=args)
            self.encoder_type_list[0] = True
            self.linear_idx[0] = 0
        elif self.input_type == "seq_variant":
            # seq -> bert -> (pooler) -> fc * -> classifier
            self.input_size_list[0] = config.hidden_size
            variant_bin_size = args.variant_bin_size
            variant_embeddings = nn.Embedding(variant_bin_size + 5, config.embedding_input_size, padding_idx=config.pad_token_id)
            config.max_position_embeddings = config.seq_max_length
            self.seq_encoder = LucaTransformer(
                config,
                use_pretrained_embedding=False,
                add_pooling_layer=(args.seq_pooling_type is None or args.seq_pooling_type == "none") and self.task_level_type in ["seq_level"],
                extra_emb_layer=variant_embeddings,
                use_dropout=False,
                use_layernorm=False
            )
            if self.task_level_type in ["seq_level"]:
                self.seq_pooler = create_pooler(pooler_type="seq", config=config, args=args)
            self.encoder_type_list[0] = True
            self.linear_idx[0] = 0
        elif self.input_type == "matrix":
            # emb matrix -> (encoder) - > (pooler) -> fc * -> classifier
            if args.matrix_encoder:
                matrix_encoder_config = copy.deepcopy(config)
                matrix_encoder_config.no_position_embeddings = True
                matrix_encoder_config.no_token_type_embeddings = True
                matrix_encoder_config.max_position_embeddings = config.matrix_max_length
                if args.matrix_encoder_act:
                    self.matrix_encoder = nn.ModuleList([
                        nn.Linear(config.embedding_input_size, config.hidden_size),
                        create_activate(config.emb_activate_func),
                        # nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
                        LucaTransformer(matrix_encoder_config, use_pretrained_embedding=True, add_pooling_layer=False)
                    ])

                else:
                    self.matrix_encoder = nn.ModuleList([
                        LucaTransformer(matrix_encoder_config, use_pretrained_embedding=True, add_pooling_layer=False)
                    ])
                ori_embedding_input_size = config.embedding_input_size
                config.embedding_input_size = config.hidden_size
                if self.task_level_type in ["seq_level"]:
                    self.matrix_pooler = create_pooler(pooler_type="matrix", config=config, args=args)
                self.input_size_list[1] = config.embedding_input_size
                config.embedding_input_size = ori_embedding_input_size
            else:
                self.input_size_list[1] = config.embedding_input_size
                if self.task_level_type in ["seq_level"]:
                    self.matrix_pooler = create_pooler(pooler_type="matrix", config=config, args=args)
            self.encoder_type_list[1] = True
            self.linear_idx[1] = 0
        elif self.input_type == "vector":
            # emb vector -> fc * -> classifier
            self.input_size_list[2] = config.embedding_input_size
            self.encoder_type_list[2] = True
            self.linear_idx[2] = 0
        elif self.input_type == "seq_matrix":
            # seq + matrix
            self.input_size_list[0] = config.hidden_size
            config.max_position_embeddings = config.seq_max_length
            self.seq_encoder = LucaTransformer(
                config,
                use_pretrained_embedding=False,
                add_pooling_layer=(args.seq_pooling_type is None or args.seq_pooling_type == "none") and self.task_level_type in ["seq_level"]
            )
            if self.task_level_type in ["seq_level"]:
                self.seq_pooler = create_pooler(pooler_type="seq", config=config, args=args)
            self.encoder_type_list[0] = True
            if args.matrix_encoder:
                matrix_encoder_config = copy.deepcopy(config)
                matrix_encoder_config.no_position_embeddings = True
                matrix_encoder_config.no_token_type_embeddings = True
                matrix_encoder_config.max_position_embeddings = config.matrix_max_length
                if args.matrix_encoder_act:
                    self.matrix_encoder = nn.ModuleList([
                        nn.Linear(config.embedding_input_size, config.hidden_size),
                        create_activate(config.emb_activate_func),
                        # nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
                        LucaTransformer(matrix_encoder_config, use_pretrained_embedding=True, add_pooling_layer=False)
                    ])

                else:
                    self.matrix_encoder = nn.ModuleList([
                        LucaTransformer(matrix_encoder_config, use_pretrained_embedding=True, add_pooling_layer=False)
                    ])
                ori_embedding_input_size = config.embedding_input_size
                config.embedding_input_size = config.hidden_size
                if self.task_level_type in ["seq_level"]:
                    self.matrix_pooler = create_pooler(pooler_type="matrix", config=config, args=args)
                self.input_size_list[1] = config.embedding_input_size
                config.embedding_input_size = ori_embedding_input_size
            else:
                self.input_size_list[1] = config.embedding_input_size
                if self.task_level_type in ["seq_level"]:
                    self.matrix_pooler = create_pooler(pooler_type="matrix", config=config, args=args)
            self.encoder_type_list[1] = True
            self.linear_idx[0] = 0
            self.linear_idx[1] = 1
        elif self.input_type == "matrix_express":
            # matrix + express
            # matrix a has express
            express_bin_size = args.express_bin_size
            self.express_embeddings = nn.Embedding(express_bin_size + 5, config.embedding_input_size, padding_idx=config.pad_token_id)
            self.LayerNorm = nn.LayerNorm(config.embedding_input_size, eps=config.layer_norm_eps)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            if args.matrix_encoder:
                matrix_encoder_config = copy.deepcopy(config)
                matrix_encoder_config.no_position_embeddings = True
                matrix_encoder_config.no_token_type_embeddings = True
                matrix_encoder_config.max_position_embeddings = config.matrix_max_length
                if args.matrix_encoder_act:
                    self.matrix_encoder = nn.ModuleList([
                        nn.Linear(config.embedding_input_size, config.hidden_size),
                        create_activate(config.emb_activate_func),
                        # nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
                        LucaTransformer(matrix_encoder_config, use_pretrained_embedding=True, add_pooling_layer=False)
                    ])

                else:
                    self.matrix_encoder = nn.ModuleList([
                        LucaTransformer(matrix_encoder_config, use_pretrained_embedding=True, add_pooling_layer=False)
                    ])
                ori_embedding_input_size = config.embedding_input_size
                config.embedding_input_size = config.hidden_size
                if self.task_level_type in ["seq_level"]:
                    self.matrix_pooler = create_pooler(pooler_type="matrix", config=config, args=args)
                self.input_size_list[1] = config.embedding_input_size
                config.embedding_input_size = ori_embedding_input_size
            else:
                self.input_size_list[1] = config.embedding_input_size
                if self.task_level_type in ["seq_level"]:
                    self.matrix_pooler = create_pooler(pooler_type="matrix", config=config, args=args)
            self.encoder_type_list[1] = True
            self.linear_idx[1] = 0
        elif self.input_type == "matrix_variant":
            # matrix + variant"
            # matrix a has variant"
            variant_bin_size = args.variant_bin_size
            self.variant_embeddings = nn.Embedding(variant_bin_size + 5, config.embedding_input_size, padding_idx=config.pad_token_id)
            self.LayerNorm = nn.LayerNorm(config.embedding_input_size, eps=config.layer_norm_eps)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            if args.matrix_encoder:
                matrix_encoder_config = copy.deepcopy(config)
                matrix_encoder_config.no_position_embeddings = True
                matrix_encoder_config.no_token_type_embeddings = True
                matrix_encoder_config.max_position_embeddings = config.matrix_max_length
                if args.matrix_encoder_act:
                    self.matrix_encoder = nn.ModuleList([
                        nn.Linear(config.embedding_input_size, config.hidden_size),
                        create_activate(config.emb_activate_func),
                        # nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
                        LucaTransformer(matrix_encoder_config, use_pretrained_embedding=True, add_pooling_layer=False)
                    ])

                else:
                    self.matrix_encoder = nn.ModuleList([
                        LucaTransformer(matrix_encoder_config, use_pretrained_embedding=True, add_pooling_layer=False)
                    ])
                ori_embedding_input_size = config.embedding_input_size
                config.embedding_input_size = config.hidden_size
                if self.task_level_type in ["seq_level"]:
                    self.matrix_pooler = create_pooler(pooler_type="matrix", config=config, args=args)
                self.input_size_list[1] = config.embedding_input_size
                config.embedding_input_size = ori_embedding_input_size
            else:
                self.input_size_list[1] = config.embedding_input_size
                if self.task_level_type in ["seq_level"]:
                    self.matrix_pooler = create_pooler(pooler_type="matrix", config=config, args=args)
            self.encoder_type_list[1] = True
            self.linear_idx[1] = 0
        elif self.input_type == "seq_vector":
            # seq + vector
            self.input_size_list[0] = config.hidden_size
            config.max_position_embeddings = config.seq_max_length
            self.seq_encoder = LucaTransformer(
                config,
                use_pretrained_embedding=False,
                add_pooling_layer=(args.seq_pooling_type is None or args.seq_pooling_type == "none") and self.task_level_type in ["seq_level"]
            )
            self.seq_pooler = create_pooler(pooler_type="seq", config=config, args=args)
            self.encoder_type_list[0] = True
            self.input_size_list[2] = config.embedding_input_size
            self.encoder_type_list[2] = True
            self.linear_idx[0] = 0
            self.linear_idx[2] = 1
        else:
            raise Exception("Not support input_type=%s" % self.input_type)
        fc_size_list = [config.seq_fc_size, config.matrix_fc_size, config.vector_fc_size]
        all_linear_list = [None, None, None]
        self.output_size = [0, 0, 0]
        for encoder_idx, encoder_flag in enumerate(self.encoder_type_list):
            if not encoder_flag:
                continue
            fc_size = fc_size_list[encoder_idx]
            input_size = self.input_size_list[encoder_idx]
            if fc_size is not None:
                if isinstance(fc_size, list):
                    fc_size = [int(v) for v in fc_size]
                else:
                    fc_size = [int(fc_size)]
                linear_list = []
                for idx in range(len(fc_size)):
                    linear = nn.Linear(input_size, fc_size[idx])
                    linear_list.append(linear)
                    linear_list.append(create_activate(config.fc_activate_func))
                    input_size = fc_size[idx]
                all_linear_list[encoder_idx] = nn.ModuleList(linear_list)
                self.output_size[encoder_idx] = fc_size[-1]
            else:
                # 没有全连接层
                self.linear_idx[encoder_idx] = -1
                self.output_size[encoder_idx] = input_size
        all_linear_list = [linear for linear in all_linear_list if linear is not None]
        if all_linear_list is not None and len(all_linear_list) > 0:
            self.linear = nn.ModuleList(all_linear_list)
        if self.fusion_type == "add":
            output_size = [v for v in self.output_size if v > 0]
            assert len(set(output_size)) == 1
            last_hidden_size = output_size[0]
        else:
            last_hidden_size = sum(self.output_size)

        self.dropout, self.hidden_layer, self.hidden_act, self.classifier, self.output, self.loss_fct = \
            create_loss_function(
                config,
                args,
                hidden_size=last_hidden_size,
                classifier_size=config.classifier_size,
                sigmoid=args.sigmoid,
                output_mode=args.output_mode,
                num_labels=self.num_labels,
                loss_type=args.loss_type,
                ignore_index=args.ignore_index,
                return_types=["dropout", "hidden_layer", "hidden_act", "classifier", "output", "loss"]
            )
        self.post_init()

    def forward(
            self,
            input_ids=None,
            position_ids=None,
            token_type_ids=None,
            seq_attention_masks=None,
            vectors=None,
            matrices=None,
            matrix_attention_masks=None,
            express_input_ids=None,
            variant_input_ids=None,
            labels=None,
            **kwargs
    ):
        '''
        print("matrices:")
        print(matrices)
        input("continue:")
        print("express_input_ids:")
        print(express_input_ids)
        input("continue:")

        print("input_ids:")
        print(input_ids)
        input("continue:")
        print("variant_input_ids:")
        print(variant_input_ids)
        input("continue:")
        # 要排除的值
        exclude_vals = torch.tensor([2, 3, 5], device=variant_input_ids.device)
        positions = [torch.where(~torch.isin(row, exclude_vals))[0] for row in variant_input_ids]
        print(positions)
        '''
        sample_ids = kwargs["sample_ids"] if "sample_ids" in kwargs else None
        attention_scores_savepath = kwargs["attention_scores_savepath"] if "attention_scores_savepath" in kwargs else None
        attention_pooling_scores_savepath = kwargs["attention_pooling_scores_savepath"] if "attention_pooling_scores_savepath" in kwargs else None
        output_matrix_dirpath = kwargs["output_matrix_dirpath"] if "output_matrix_dirpath" in kwargs else None
        output_classification_vector_dirpath = kwargs["output_classification_vector_dirpath"] if "output_classification_vector_dirpath" in kwargs else None
        output_logits_dirpath = kwargs["output_logits_dirpath"] if "output_logits_dirpath" in kwargs else None
        return_attentions = sample_ids is not None and attention_scores_savepath is not None
        if input_ids is not None and self.seq_encoder is not None:
            seq_outputs = self.seq_encoder(
                input_ids,
                attention_mask=seq_attention_masks,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=None,
                return_attentions=return_attentions,
                extra_input_ids=express_input_ids if express_input_ids is not None else variant_input_ids
            )
            if return_attentions:
                output_seq_attentions = seq_outputs[2]
                for sample_idx, sample_id in enumerate(sample_ids):
                    cur_sample_output_seq_attentions = {}
                    for layer_idx, attn in output_seq_attentions.items():
                        cur_sample_output_seq_attentions[layer_idx] = attn[sample_idx].detach().cpu()
                filepath = os.path.join(attention_scores_savepath, "%s_seq_attention_scores.pt" % sample_id)
                torch.save(cur_sample_output_seq_attentions, filepath)
            if output_matrix_dirpath:
                for sample_idx, sample_id in enumerate(sample_ids):
                    filepath = os.path.join(output_matrix_dirpath, "%s_seq_representation_matrix.pt" % sample_id)
                    torch.save(seq_outputs[0][sample_idx].detach().cpu(), filepath)

            # 将CLS与SEP位置设置为0（mask掉）
            if self.append_eos:
                seq_index = torch.sum(seq_attention_masks, dim=1, keepdim=True) - 1
                seq_attention_masks = seq_attention_masks.scatter(1, seq_index, 0)
            if self.prepend_bos:
                seq_attention_masks[:, 0] = 0
            if self.seq_pooler is not None:
                seq_vector = self.seq_pooler(
                    seq_outputs[0],
                    mask=seq_attention_masks,
                    sample_ids=sample_ids,
                    prefix="seq",
                    attention_pooling_scores_savepath=attention_pooling_scores_savepath
                )
            elif self.task_level_type in ["seq_level"]:
                seq_vector = seq_outputs[1]
            else:
                seq_vector = seq_outputs[0]

            seq_linear_idx = self.linear_idx[0]
            if seq_linear_idx != -1:
                for i, layer_module in enumerate(self.linear[seq_linear_idx]):
                    seq_vector = layer_module(seq_vector)

        if vectors is not None:
            vector_vector = vectors
            vector_linear_idx = self.linear_idx[2]
            if vector_linear_idx != -1:
                for i, layer_module in enumerate(self.linear[vector_linear_idx]):
                    vector_vector = layer_module(vector_vector)
        if matrices is not None:
            if express_input_ids is not None:
                express_input_embeds = self.express_embeddings(express_input_ids)
                matrices = matrices + express_input_embeds
                matrices = self.LayerNorm(matrices)
                matrices = self.dropout(matrices)
            elif variant_input_ids is not None:
                variant_input_embeds = self.variant_embeddings(variant_input_ids)
                matrices = matrices + variant_input_embeds
                matrices = self.LayerNorm(matrices)
                matrices = self.dropout(matrices)
            if self.matrix_encoder is not None:
                for module in self.matrix_encoder[:-1]:
                    matrices = module(matrices)
                matrices_output = self.matrix_encoder[-1](
                    input_ids=None,
                    attention_mask=matrix_attention_masks,
                    token_type_ids=None,
                    position_ids=None,
                    inputs_embeds=matrices
                )
                matrices = matrices_output[0]
                if return_attentions:
                    output_matrix_attentions = matrices_output[2]
                    for sample_idx, sample_id in enumerate(sample_ids):
                        cur_sample_output_matrix_attentions = {}
                        for layer_idx, attn in output_matrix_attentions.items():
                            cur_sample_output_matrix_attentions[layer_idx] = attn[sample_idx].detach().cpu()
                        filepath = os.path.join(attention_scores_savepath, "%s_matrix_attention_scores.pt" % sample_id)
                        torch.save(cur_sample_output_matrix_attentions, filepath)
            if output_matrix_dirpath:
                for sample_idx, sample_id in enumerate(sample_ids):
                    filepath = os.path.join(output_matrix_dirpath, "%s_embedding_representation_matrix.pt" % sample_id)
                    torch.save(matrices[sample_idx].detach().cpu(), filepath)

            # matrix_attention_masks的特殊位置为0
            if self.matrix_pooler is not None:
                matrix_vector = self.matrix_pooler(
                    matrices,
                    mask=matrix_attention_masks,
                    sample_ids=sample_ids,
                    prefix="matrix",
                    attention_pooling_scores_savepath=attention_pooling_scores_savepath
                )
            elif self.task_level_type in ["seq_level"]:
                tmp_mask = torch.unsqueeze(matrix_attention_masks, dim=-1)
                matrices = matrices.masked_fill(tmp_mask == 0, 0.0)
                # 均值pooling
                matrix_vector = torch.sum(matrices, dim=1)/(torch.sum(tmp_mask, dim=1) + 1e-12)
            else:
                matrix_vector = matrices
            matrix_linear_idx = self.linear_idx[1]
            if matrix_linear_idx != -1:
                for i, layer_module in enumerate(self.linear[matrix_linear_idx]):
                    matrix_vector = layer_module(matrix_vector)

        if self.input_type in ["seq", "seq_variant"]:
            concat_vector = seq_vector
        elif self.input_type in ["matrix", "matrix_express", "matrix_variant"]:
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
        if self.dropout is not None:
            concat_vector = self.dropout(concat_vector)
        if self.hidden_layer is not None:
            concat_vector = self.hidden_layer(concat_vector)
        if self.hidden_act is not None:
            concat_vector = self.hidden_act(concat_vector)
        if output_classification_vector_dirpath and sample_ids:
            for sample_idx, sample_id in enumerate(sample_ids):
                if self.input_type == "seq":
                    output_classification_vector = {
                        "representation_vector": seq_vector[sample_idx].detach().cpu(),
                    }
                elif self.input_type in ["matrix", "matrix_express", "matrix_variant"]:
                    output_classification_vector = {
                        "representation_vector": matrix_vector[sample_idx].detach().cpu(),
                    }
                elif self.input_type == "vector":
                    output_classification_vector = {
                        "representation_vector": vector_vector[sample_idx].detach().cpu(),
                    }
                elif self.input_type == "seq_matrix":
                    output_classification_vector = {
                        "seq_representation_vector": seq_vector[sample_idx].detach().cpu(),
                        "matrix_representation_vector": matrix_vector[sample_idx].detach().cpu(),
                    }
                elif self.input_type == "seq_vector":
                    output_classification_vector = {
                        "seq_representation_vector": seq_vector[sample_idx].detach().cpu(),
                        "vector_representation_vector": vector_vector[sample_idx].detach().cpu(),
                    }
                else:
                    raise Exception("Not support input_type=%s" % self.input_type)
                output_classification_vector["classification_vector"] = concat_vector[sample_idx].detach().cpu()
                filepath = os.path.join(output_classification_vector_dirpath, "%s_classification_vector.pt" % sample_id)
                torch.save(output_classification_vector, filepath)
        logits = self.classifier(concat_vector)
        if self.output is not None:
            output = self.output(logits)
        else:
            output = logits
        outputs = [logits, output]
        if output_logits_dirpath and sample_ids:
            for sample_idx, sample_id in enumerate(sample_ids):
                cur_logit = logits[sample_idx]
                filepath = os.path.join(output_logits_dirpath, "%s_logit.pt" % sample_id)
                torch.save(cur_logit, filepath)

        '''
        print("logits shape:")
        print(logits.shape)
        print("labels shape:")
        print(labels.shape)
        '''
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




