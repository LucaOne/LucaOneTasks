#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.**@**.com
@tel: 137****6540
@datetime: 2022/12/5 16:49
@project: LucaOneTasks
@file: pooling
@desc: pooling strategies
'''
import sys
import copy
import torch
import torch.nn as nn
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../src")
try:
    from common.modeling_bert import BertPooler, BertModel, BertEncoder
except ImportError:
    from src.common.modeling_bert import BertPooler, BertModel, BertEncoder


def create_pooler(pooler_type, config, args):
    '''
    pooler building
    :param config:
    :param args:
    :return:
    '''
    if pooler_type == "seq":
        pooling_type = args.seq_pooling_type
        hidden_size = config.hidden_size
    else:
        pooling_type = args.matrix_pooling_type
        hidden_size = config.embedding_input_size

    if pooling_type == "first":
        return GlobalMaskFirstPooling1D()
    elif pooling_type == "last":
        return GlobalMaskLastPooling1D()
    elif pooling_type == "attentive":
        return GlobalMaskAttentivePooling1D(embed_size=hidden_size)
    elif pooling_type == "max":
        return GlobalMaskMaxPooling1D()
    elif pooling_type == "sum":
        return GlobalMaskSumPooling1D(axis=1)
    elif pooling_type == "avg":
        return GlobalMaskAvgPooling1D()
    elif pooling_type == "attention":
        return GlobalMaskContextAttentionPooling1D(embed_size=hidden_size)
    elif pooling_type == "context_attention":
        return GlobalMaskContextAttentionPooling1D(embed_size=hidden_size)
    elif pooling_type == "weighted_attention":
        return GlobalMaskWeightedAttentionPooling1D(embed_size=hidden_size)
    elif pooling_type == "value_attention":
        return GlobalMaskValueAttentionPooling1D(embed_size=hidden_size)
    elif pooling_type == "transformer":
        copy_config = copy.deepcopy(config)
        copy_config.hidden_size = hidden_size
        return GlobalMaskTransformerPooling1D(copy_config)
    else:
        return None


class GlobalMaskFirstPooling1D(nn.Module):
    def __init__(self):
        super(GlobalMaskFirstPooling1D, self).__init__()

    def forward(self, x, mask=None):
        # B * seq_len * emb_dim
        if mask is None:
            return x[:, 0, :]
        else:
            seq_len = mask.shape[1]
            batch_size = mask.shape[0]
            position_ids = torch.arange(1, seq_len + 1).expand((batch_size, -1)).to(mask.device)
            masks_position_ids = position_ids * torch.where(mask == 0, seq_len + 1, mask)
            indices = torch.min(masks_position_ids, dim=1)[0] - 1
            x_selected = x[range(indices.shape[0]), indices]
            return x_selected


class GlobalMaskLastPooling1D(nn.Module):
    def __init__(self):
        super(GlobalMaskLastPooling1D, self).__init__()

    def forward(self, x, mask=None):
        # B * seq_len * emb_dim
        if mask is None:
            return x[:, -1, :]
        else:
            seq_len = mask.shape[1]
            batch_size = mask.shape[0]
            position_ids = torch.arange(0, seq_len).expand((batch_size, -1)).to(mask.device)
            masks_position_ids = position_ids * mask
            indices = torch.max(masks_position_ids, dim=1)[0]
            x_selected = x[range(indices.shape[0]), indices]
            return x_selected


class GlobalMaskAttentivePooling1D(nn.Module):
    def __init__(self, embed_size, use_bias=False):
        super(GlobalMaskAttentivePooling1D, self).__init__()
        self.linear_proj = nn.Linear(embed_size, embed_size)
        self.weight = nn.Linear(embed_size, 1, bias=use_bias)

    def forward(self, x, mask=None):
        # B * seq_len * embed_size
        out = self.linear_proj(x)
        # B * seq_len
        attention_probs = torch.tanh(self.weight(out)).squeeze(-1)
        if mask is not None:
            attention_probs = torch.softmax(attention_probs + (1.0 - mask) * -10000, dim=-1)
        else:
            attention_probs = torch.softmax(attention_probs, dim=-1)
        return torch.sum(attention_probs.unsqueeze(-1) * x, 1)


class GlobalMaskMaxPooling1D(nn.Module):
    def __init__(self):
        super(GlobalMaskMaxPooling1D, self).__init__()

    def forward(self, x, mask=None):
        if mask is not None:
            # (B, Seq_len) -> (B, Seq_len, 1)
            mask = 1.0 - mask
            mask = mask * (-2**10 + 1)
            mask = torch.unsqueeze(mask, dim=-1)
            x += mask
        return torch.max(x, dim=1)[0]


class GlobalMaskMinPooling1D(nn.Module):
    def __init__(self):
        super(GlobalMaskMinPooling1D, self).__init__()

    def forward(self, x, mask=None):
        if mask is not None:
            # (B, Seq_len) -> (B, Seq_len, 1)
            mask = 1.0 - mask
            mask = mask * (2**10+1)
            mask = torch.unsqueeze(mask, dim=-1)
            x += mask
        return torch.min(x, dim=1)[0]


class GlobalMaskAvgPooling1D(nn.Module):
    def __init__(self):
        super(GlobalMaskAvgPooling1D, self).__init__()

    def forward(self, x, mask=None):
        if mask is not None:
            # (B, Seq_len) -> (B, Seq_len, 1)
            mask = torch.unsqueeze(mask, dim=-1)
            x *= mask
            return torch.sum(x, dim=1)/torch.sum(mask, dim=1)
        else:
            return torch.mean(x, dim=1)


class GlobalMaskSumPooling1D(nn.Module):
    def __init__(self, axis):
        '''
        sum pooling
        :param axis: axis=0, add all the rows of the matrix，axis=1, add all the cols of the matrix
        '''
        super(GlobalMaskSumPooling1D, self).__init__()
        self.axis = axis

    def forward(self, x, mask=None):
        if mask is not None:
            # (B, Seq_len) -> (B, Seq_len, 1)
            mask = torch.unsqueeze(mask, dim=-1)
            x *= mask
        return torch.sum(x, dim=self.axis)


class GlobalMaskWeightedAttentionPooling1D(nn.Module):
    def __init__(self, embed_size, use_bias=False):
        super(GlobalMaskWeightedAttentionPooling1D, self).__init__()
        self.embed_size = embed_size
        self.use_bias = use_bias

        self.W = nn.Parameter(torch.Tensor(self.embed_size))
        nn.init.trunc_normal_(self.W, std=0.01)
        if self.use_bias:
            self.b = nn.Parameter(torch.Tensor(1))
            nn.init.trunc_normal_(self.b, std=0.01)

    def forward(self, x, mask=None):
        # (B, Len, Embed) x (Embed,) = (B, Len)
        logits = torch.matmul(x, self.W)
        if self.use_bias:
            logits += self.b

        if mask is not None:
            attention_probs = nn.Softmax(dim=-1)(logits + (1.0 - mask) * -10000)
        else:
            attention_probs = nn.Softmax(dim=-1)(logits)
        x = torch.sum(torch.unsqueeze(attention_probs, dim=-1) * x, dim=1)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.embed_size) + (', bias=%r)' % self.use_bias)


class GlobalMaskContextAttentionPooling1D(nn.Module):
    def __init__(self, embed_size, units=None, use_additive_bias=False, use_attention_bias=False):
        super(GlobalMaskContextAttentionPooling1D, self).__init__()
        self.embed_size = embed_size
        self.use_additive_bias = use_additive_bias
        self.use_attention_bias = use_attention_bias
        self.units = units if units else embed_size

        self.U = nn.Parameter(torch.Tensor(self.embed_size, self.units))
        self.V = nn.Parameter(torch.Tensor(self.embed_size, self.units))
        if self.use_additive_bias:
            self.b1 = nn.Parameter(torch.Tensor(self.units))
            nn.init.trunc_normal_(self.b1, std=0.01)
        if self.use_attention_bias:
            self.b2 = nn.Parameter(torch.Tensor(1))
            nn.init.trunc_normal_(self.b2, std=0.01)

        self.c = nn.Parameter(torch.Tensor(self.units))

        nn.init.trunc_normal_(self.U, std=0.01)
        nn.init.trunc_normal_(self.V, std=0.01)
        nn.init.trunc_normal_(self.c, std=0.01)

    def forward(self, x, mask=None):
        # (B, Len, Embed) x (Embed, Units) = (B, Len, Units)
        q = torch.matmul(x, self.U)
        k = torch.matmul(x, self.V)
        if self.use_additive_bias:
            h = torch.tanh(q + k + self.b1)
        else:
            h = torch.tanh(q + k)

        if self.use_attention_bias:
            e = torch.matmul(h, self.c) + self.b2
        else:
            e = torch.matmul(h, self.c)
        if mask is not None:
            attention_probs = nn.Softmax(dim=-1)(e + (1.0 - mask) * -10000)
        else:
            attention_probs = nn.Softmax(dim=-1)(e)
        x = torch.sum(torch.unsqueeze(attention_probs, dim=-1) * x, dim=1)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.embed_size) + ' -> ' + str(self.units) + ', bias=(%r, %r))' % (self.use_additive_bias, self.use_attention_bias)


class GlobalMaskValueAttentionPooling1D(nn.Module):
    def __init__(self, embed_size, units=None, use_additive_bias=False, use_attention_bias=False):
        super(GlobalMaskValueAttentionPooling1D, self).__init__()
        self.embed_size = embed_size
        self.use_additive_bias = use_additive_bias
        self.use_attention_bias = use_attention_bias
        self.units = units if units else embed_size

        self.U = nn.Parameter(torch.Tensor(self.embed_size, self.units))
        self.V = nn.Parameter(torch.Tensor(self.embed_size, self.units))
        if self.use_additive_bias:
            self.b1 = nn.Parameter(torch.Tensor(self.units))
            nn.init.trunc_normal_(self.b1, std=0.01)
        if self.use_attention_bias:
            self.b2 = nn.Parameter(torch.Tensor(self.embed_size))
            nn.init.trunc_normal_(self.b2, std=0.01)

        self.W = nn.Parameter(torch.Tensor(self.units, self.embed_size))

        nn.init.trunc_normal_(self.U, std=0.01)
        nn.init.trunc_normal_(self.V, std=0.01)
        nn.init.trunc_normal_(self.W, std=0.01)

    def forward(
            self,
            x,
            mask=None,
            sample_ids=None,
            save_attention_scores=False
    ):
        # (B, Len, Embed) x (Embed, Units) = (B, Len, Units)
        q = torch.matmul(x, self.U)
        k = torch.matmul(x, self.V)
        if self.use_additive_bias:
            h = torch.tanh(q + k + self.b1)
        else:
            h = torch.tanh(q + k)

        # (B, Len, Units) x (Units, Embed) = (B, Len, Embed)
        if self.use_attention_bias:
            e = torch.matmul(h, self.W) + self.b2
        else:
            e = torch.matmul(h, self.W)
        if mask is not None:
            attention_probs = nn.Softmax(dim=1)(e + torch.unsqueeze((1.0 - mask) * -10000, dim=-1))
        else:
            attention_probs = nn.Softmax(dim=1)(e)
        x = torch.sum(attention_probs * x, dim=1)
        if save_attention_scores and sample_ids:
            import os
            dirpath = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "attention_scores")
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
            for sample_idx, sample_id in enumerate(sample_ids):
                filepath = os.path.join(dirpath, "%s.pt" % sample_id)
                if attention_probs is not None:
                    attention_probs_cpu = attention_probs.detach().cpu()
                    torch.save(attention_probs_cpu, filepath)

        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.embed_size) + ' -> ' + str(self.units) + ', bias=(%r, %r))' % (self.use_additive_bias, self.use_attention_bias)


class GlobalMaskTransformerPooling1D(nn.Module):
    def __init__(self, config):
        super(GlobalMaskTransformerPooling1D, self).__init__()
        self.embeddings = nn.Parameter(torch.Tensor(1, 1, config.hidden_size))
        nn.init.trunc_normal_(self.embeddings, std=0.02)
        config.num_hidden_layers = 2
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, x, mask=None):
        B, Seq_len, Enbed = x.size()
        cls_emb_batch = self.embeddings.expand(B, 1, Enbed)
        merged_output = torch.cat((cls_emb_batch, x), dim=1) # [B, Seq_len + 1, Enbed]
        if mask is not None:
            device = x.device
            cls_mask = torch.ones(B, 1).to(device)
            mask = torch.cat([cls_mask, mask], dim=1)
            mask = mask[:, None, None, :]

        sequence_output = self.encoder(merged_output,
                                       attention_mask=mask,
                                       head_mask=None,
                                       encoder_hidden_states=None,
                                       encoder_attention_mask=None,
                                       output_attentions=False,
                                       output_hidden_states=False,
                                       return_dict=False)[0]
        pooled_output = self.pooler(sequence_output)
        return pooled_output


class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()
        self.fc = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        x = torch.squeeze(x, dim=-1)
        return x


class GlobalAvgPool1d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool1d, self).__init__()
        self.fc = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        x = torch.squeeze(x, dim=-1)
        return x


class AttentionPool1d(nn.Module):
    def __init__(self, embed_size):
        super(AttentionPool1d, self).__init__()
        self.embed_size = embed_size
        self.W = nn.Parameter(torch.Tensor(self.embed_size, self.embed_size))
        self.b = nn.Parameter(torch.Tensor(self.embed_size))
        self.c = nn.Parameter(torch.Tensor(self.embed_size))
        nn.init.trunc_normal_(self.W, std=0.02)
        nn.init.trunc_normal_(self.b, std=0.02)
        nn.init.trunc_normal_(self.c, std=0.02)

    def forward(self, x):
        '''
        # x: (b, seq_len, embed)
        # mul: (b, seq_len)
        mul = torch.matmul(x, self.w)
        # b, seq_len
        attention_probs = nn.Softmax(dim=-1)(mul)
        # b, seq_len
        x = torch.sum(torch.unsqueeze(attention_probs, dim=-1) * x, dim=1)
        '''
        mul = torch.tanh(torch.matmul(x, self.W) + self.b)
        mul = torch.matmul(mul, self.c)
        attention_probs = nn.Softmax(dim=-1)(mul)
        x = torch.sum(torch.unsqueeze(attention_probs, dim=-1) * x, dim=1)
        return x


class TransformerPool1d(nn.Module):
    def __init__(self, config, embeddings, embed_size, num_transformer_layers=2, cls_id=2, device="cuda"):
        super(TransformerPool1d, self).__init__()
        if embeddings:
            self.embeddings = embeddings
        else:
            self.embeddings = nn.Parameter(torch.Tensor(1, 1, embed_size))
            nn.init.trunc_normal_(self.embeddings, std=0.02)
            # self.embeddings = BertEmbeddings(config)
        self.cls_id = cls_id
        self.device = device
        config.num_hidden_layers = num_transformer_layers
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, x):
        # x：(b, seq_len, embed)
        b, seq_len, embed = x.size()
        # b, 1
        # cls_emb_batch = self.embeddings(torch.tensor([[self.cLs_id]] * x.size()[0], dtype=torch.long).to(self.device))
        cls_emb_batch = self.embeddings.expand(b, 1, embed)
        merged_output = torch.cat((cls_emb_batch, x), dim=1)
        # [b, seq_len + 1, embed]
        sequence_output = self.encoder(merged_output,
                                       attention_mask=None,
                                       head_mask=None,
                                       encoder_hidden_states=None,
                                       encoder_attention_mask=None,
                                       output_attentions=False,
                                       output_hidden_states=False,
                                       return_dict=False)[0]
        pooled_output = self.pooler(sequence_output)
        return pooled_output


