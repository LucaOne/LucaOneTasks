#!/usr/bin/env python
# encoding: utf-8
"""
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/6/21 17:32
@project: LucaX
@file: LucaIntraInter
@desc: xxxx
"""

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
    from common.luca_pair import LucaPair
    from common.modeling_bert import BertPreTrainedModel
except ImportError:
    from src.common.pooling import *
    from src.common.loss import *
    from src.utils import *
    from src.common.multi_label_metrics import *
    from src.common.metrics import *
    from src.common.luca_pair import LucaPair
    from src.common.modeling_bert import BertPreTrainedModel
logger = logging.getLogger(__name__)


class LucaPairIntraInter(BertPreTrainedModel):
    def __init__(self, config, args):
        super(LucaPairIntraInter, self).__init__(config)
        self.input_type = args.input_type
        self.num_labels = config.num_labels
        self.output_mode = args.output_mode
        self.task_level_type = args.task_level_type
        self.fusion_type = args.fusion_type if hasattr(args, "fusion_type") and args.fusion_type else "concat"
        if config.hidden_size != config.embedding_input_size_a:
            self.linear_a = nn.Linear(config.embedding_input_size_a, config.hidden_size, bias=True)
        else:
            self.linear_a = None
        if config.hidden_size != config.embedding_input_size_b:
            self.linear_b = nn.Linear(config.embedding_input_size_b, config.hidden_size, bias=True)
        else:
            self.linear_b = None
        if self.input_type == "matrix_express_vs_matrix_express":
            # matrix a has express
            express_bin_size_a = args.express_bin_size_a
            self.express_embeddings_a = nn.Embedding(express_bin_size_a + 5, config.embedding_input_size_a, padding_idx=config.pad_token_id)
            self.LayerNorm_a = nn.LayerNorm(config.embedding_input_size_a, eps=config.layer_norm_eps)
            self.dropout_a = nn.Dropout(config.hidden_dropout_prob)

            express_bin_size_b = args.express_bin_size_b
            self.express_embeddings_b = nn.Embedding(express_bin_size_b + 5, config.embedding_input_size_b, padding_idx=config.pad_token_id)
            self.LayerNorm_b = nn.LayerNorm(config.embedding_input_size_b, eps=config.layer_norm_eps)
            self.dropout_b = nn.Dropout(config.hidden_dropout_prob)
        self.encoder = LucaPair(config)
        config.embedding_input_size = config.hidden_size
        self.pooler = nn.ModuleList([create_pooler(pooler_type="matrix", config=config, args=args) for _ in range(4)])
        self.dropout, self.hidden_layer, self.hidden_act, self.classifier, self.output, self.loss_fct = \
            create_loss_function(
                config,
                args,
                hidden_size=4 * config.hidden_size if self.fusion_type == "concat" else config.hidden_size,
                classifier_size=args.classifier_size,
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
            input_ids_a=None,
            input_ids_b=None,
            position_ids_a=None,
            position_ids_b=None,
            token_type_ids_a=None,
            token_type_ids_b=None,
            seq_attention_masks_a=None,
            seq_attention_masks_b=None,
            vectors_a=None,
            vectors_b=None,
            matrices_a=None,
            matrices_b=None,
            matrix_attention_masks_a=None,
            matrix_attention_masks_b=None,
            express_input_ids_a=None,
            express_input_ids_b=None,
            output_attentions=False,
            labels=None,
            **kwargs
    ):
        sample_ids = kwargs["sample_ids"] if "sample_ids" in kwargs else None
        attention_scores_savepath = kwargs["attention_scores_savepath"] if "attention_scores_savepath" in kwargs else None
        attention_pooling_scores_savepath = kwargs["attention_pooling_scores_savepath"] if "attention_pooling_scores_savepath" in kwargs else None
        output_classification_vector_dirpath = kwargs["output_classification_vector_dirpath"] if "output_classification_vector_dirpath" in kwargs else None
        output_attentions = output_attentions or (sample_ids and attention_scores_savepath)
        # 对称结构： intra-attention + inter-attention
        # matrices_a: seq_a_embedding, [B, seq_len_a, dim]
        # matrix_attention_masks_a: seq_a_mask, [B, seq_len_a]
        # matrices_b: seq_b_embedding, [B, seq_len_b, dim]
        # matrix_attention_masks_b: seq_b_mask, [B, seq_len_b]
        if self.linear_a is not None:
            # [B, seq_len_a, dim]->[B, seq_len_a, hidden_size]
            hidden_states_a = self.linear_a(matrices_a)
        else:
            hidden_states_a = matrices_a
        if express_input_ids_a is not None:
            express_input_embeds_a = self.express_embeddings_a(express_input_ids_a)
            hidden_states_a = hidden_states_a + express_input_embeds_a
            hidden_states_a = self.LayerNorm_a(hidden_states_a)
            hidden_states_a = self.dropout_a(hidden_states_a)
        if self.linear_b is not None:
            # [B, seq_len_b, dim]->[B, seq_len_b, hidden_size]
            hidden_states_b = self.linear_b(matrices_b)
        else:
            hidden_states_b = matrices_b
        if express_input_ids_b is not None:
            express_input_embeds_b = self.express_embeddings_b(express_input_ids_b)
            hidden_states_b = hidden_states_b + express_input_embeds_b
            hidden_states_b = self.LayerNorm_b(hidden_states_b)
            hidden_states_b = self.dropout_b(hidden_states_b)
        outputs = self.encoder(
            hidden_states_a=hidden_states_a,
            attention_mask_a=matrix_attention_masks_a,
            hidden_states_b=hidden_states_b,
            attention_mask_b=matrix_attention_masks_b,
            head_mask_a=None,
            cross_attn_head_mask_ab=None,
            head_mask_b=None,
            cross_attn_head_mask_ba=None,
            past_key_values=None,
            cross_past_key_values=None,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=True
        )
        last_hidden_states = outputs.last_hidden_state
        if output_attentions:
            self_attentions = outputs.attentions
            cross_attentions = outputs.cross_attentions
            for sample_idx, sample_id in enumerate(sample_ids):
                cur_self_attentions_a = {}
                cur_self_attentions_b = {}
                for layer_idx in range(len(self_attentions)):
                    cur_self_attentions_a[layer_idx] = self_attentions[layer_idx][0].detach().cpu()
                    cur_self_attentions_b[layer_idx] = self_attentions[layer_idx][1].detach().cpu()
                filepath_a = os.path.join(attention_scores_savepath, "%s_self_attention_scores_a.pt" % sample_id)
                torch.save(cur_self_attentions_a, filepath_a)
                filepath_b = os.path.join(attention_scores_savepath, "%s_self_attention_scores_b.pt" % sample_id)
                torch.save(cur_self_attentions_b, filepath_b)
                cur_cross_attentions_ab = {}
                cur_cross_attentions_ba = {}
                for layer_idx in range(len(cross_attentions)):
                    cur_cross_attentions_ab[layer_idx] = cross_attentions[layer_idx][0].detach().cpu()
                    cur_cross_attentions_ba[layer_idx] = cross_attentions[layer_idx][1].detach().cpu()
                filepath_ab = os.path.join(attention_scores_savepath, "%s_cross_attention_scores_ab.pt" % sample_id)
                torch.save(cur_cross_attentions_ab, filepath_ab)
                filepath_ba = os.path.join(attention_scores_savepath, "%s_cross_attention_scores_ba.pt" % sample_id)
                torch.save(cur_cross_attentions_ba, filepath_ba)

        if self.pooler is not None:
            last_hidden_states = [
                # a
                self.pooler[0](
                    last_hidden_states[0],
                    seq_attention_masks_a,
                    sample_ids=sample_ids,
                    prefix="matrix_a",
                    attention_pooling_scores_savepath=attention_pooling_scores_savepath
                ),
                # b
                self.pooler[1](
                    last_hidden_states[1],
                    seq_attention_masks_b,
                    sample_ids=sample_ids,
                    prefix="matrix_b",
                    attention_pooling_scores_savepath=attention_pooling_scores_savepath
                ),
                # ab
                self.pooler[2](
                    last_hidden_states[2],
                    seq_attention_masks_a,
                    sample_ids=sample_ids,
                    prefix="matrix_ab",
                    attention_pooling_scores_savepath=attention_pooling_scores_savepath
                ),
                # ba
                self.pooler[3](
                    last_hidden_states[3],
                    seq_attention_masks_b,
                    sample_ids=sample_ids,
                    prefix="matrix_ba",
                    attention_pooling_scores_savepath=attention_pooling_scores_savepath
                )
            ]
        if output_classification_vector_dirpath and sample_ids:
            for sample_idx, sample_id in enumerate(sample_ids):
                output_classification_vector = {
                    "representation_vector_a": last_hidden_states[0][sample_idx].detach().cpu(),
                    "representation_vector_b": last_hidden_states[1][sample_idx].detach().cpu(),
                    "representation_vector_ab": last_hidden_states[2][sample_idx].detach().cpu(),
                    "representation_vector_ba": last_hidden_states[3][sample_idx].detach().cpu(),
                }
                filepath = os.path.join(output_classification_vector_dirpath, "%s_classification_vector.pt" % sample_id)
                torch.save(output_classification_vector, filepath)
        if self.dropout is not None:
            last_hidden_states = [self.dropout(hidden) for hidden in last_hidden_states]
        if self.fusion_type == "add":
            last_hidden_states = torch.add(last_hidden_states)
        else:
            last_hidden_states = torch.cat(last_hidden_states, dim=-1)

        if self.hidden_layer is not None:
            last_hidden_states = self.hidden_layer(last_hidden_states)
        if self.hidden_act is not None:
            last_hidden_states = self.hidden_act(last_hidden_states)

        logits = self.classifier(last_hidden_states)
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
            else:
                raise Exception("not support the output_mode=%s" % self.output_mode)
            outputs = [loss, *outputs]
        return outputs
