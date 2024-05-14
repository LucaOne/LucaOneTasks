#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/8/30 09:55
@project: LucaOneTasks
@file: cross_transformer.py
@desc: cross transformer
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def utils_softmax(x, dim: int, onnx_trace: bool = False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    cos = cos[:, : x.shape[-2], :]
    sin = sin[:, : x.shape[-2], :]

    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim: int, *_, **__):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dimension=1):
        seq_len = x.shape[seq_dimension]

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dimension], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self._cos_cached = emb.cos()[None, :, :]
            self._sin_cached = emb.sin()[None, :, :]

        return self._cos_cached, self._sin_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=-2)

        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )


class LucaLayerNormV1(nn.Module):
    def __init__(self, hidden_size, eps=1e-12, affine=True):
        """
        Construct a layernorm layer in the TF style (eps inside the sqrt).
        """
        super().__init__()
        self.hidden_size = (hidden_size,) if isinstance(hidden_size, int) else tuple(hidden_size)
        self.eps = eps
        self.affine = bool(affine)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.weight, self.bias = None, None

    def forward(self, x):
        dims = tuple(-(i + 1) for i in range(len(self.hidden_size)))
        means = x.mean(dims, keepdim=True)
        x_zeromean = x - means
        variances = x_zeromean.pow(2).mean(dims, keepdim=True)
        x = x_zeromean / torch.sqrt(variances + self.eps)
        if self.affine:
            x = (self.weight * x) + self.bias
        return x


try:
    # Optimized LayerNorm
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm
    class LucaLayerNormV2(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)
except ImportError as e:
    print("import apex err:", e)
    from torch.nn import LayerNorm as LucaLayerNormV2


class LucaTransformerLayer(nn.Module):
    """
    LucaTransformer layer block.
    """
    def __init__(
            self,
            embed_dim,
            ffn_embed_dim,
            attention_heads,
            dropout=0.1,
            add_bias_kv=False,
            use_luca_layer_norm_v2=False,
            attention_type="self",
            use_rotary_embeddings: bool = True
    ):
        '''
        Tramsformer-Encoder 层
        :param embed_dim: token embedding dim
        :param ffn_embed_dim: fully connected layer dim
        :param attention_heads: heads num
        :param dropout: dropout
        :param add_bias_kv: key-value layer add bias
        :param use_luca_layer_norm_v2: whether to use layer norm v2
        :param attention_type: self, encoder-decoder, cross
        :param use_rotary_embeddings: whether to use rotary embedding
        '''
        super().__init__()
        assert attention_type in ["encoder-decoder", "self"]
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.dropout = dropout
        self.add_bias_kv = add_bias_kv
        self.attention_type = attention_type
        self.use_rotary_embeddings = use_rotary_embeddings
        self._init_submodules(add_bias_kv, use_luca_layer_norm_v2)

    def _init_submodules(self, add_bias_kv, use_luca_layer_norm_v2):
        LucaLayerNorm = LucaLayerNormV2 if use_luca_layer_norm_v2 else LucaLayerNormV1


        # pre layer norm
        self.pre_layer_norm = LucaLayerNorm(self.embed_dim)

        self.self_attn = LucaMultiHeadAttention(
            self.embed_dim,
            self.attention_heads,
            k_dim=self.embed_dim,
            v_dim=self.embed_dim,
            dropout=self.dropout,
            bias=True,
            add_bias_kv=add_bias_kv,
            add_zero_attn=False,
            attention_type=self.attention_type,
            use_rotary_embeddings=self.use_rotary_embeddings
        )

        # post layer norm
        self.post_layer_norm = LucaLayerNorm(self.embed_dim)

        # dimension increase by the fully connected layer
        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim, bias=True)

        # dimension reduction by the fully connected layer
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim, bias=True)

    def forward(
            self,
            x,
            self_attn_mask=None,
            self_attn_padding_mask=None,
            need_head_weights=False
    ):
        residual = x
        x = self.pre_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=True,
            need_head_weights=need_head_weights,
            attn_mask=self_attn_mask,
        )
        x = residual + x

        residual = x
        x = self.post_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        return x, attn


class LucaCrossTransformerLayer(nn.Module):
    """
    Luca Cross Transformer layer block.
    """
    def __init__(
            self,
            embed_dim,
            ffn_embed_dim,
            attention_heads,
            dropout=0.1,
            add_bias_kv=False,
            use_luca_layer_norm_v2=False,
            attention_type="cross",
            use_rotary_embeddings: bool = True
    ):
        '''
        Tramsformer-Encoder 层
        :param embed_dim: token embedding dim
        :param ffn_embed_dim: fully connected layer dim
        :param attention_heads: heads num
        :param add_bias_kv: key-value layer add bias
        :param use_luca_layer_norm_v2: whether to use layer norm v2
        :param attention_type: self, encoder-decoder, cross
        :param use_rotary_embeddings: whether to use rotary embedding
        '''
        super().__init__()
        assert attention_type == "cross"
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.dropout = dropout
        self.add_bias_kv = add_bias_kv
        self.use_rotary_embeddings = use_rotary_embeddings
        self.attention_type = attention_type
        self._init_submodules(add_bias_kv, use_luca_layer_norm_v2)

    def _init_submodules(self, add_bias_kv, use_luca_layer_norm_v2):
        LucaLayerNorm = LucaLayerNormV2 if use_luca_layer_norm_v2 else LucaLayerNormV1

        # pre layer norm
        self.pre_layer_norm = LucaLayerNorm(self.embed_dim)

        self.cross_attn = LucaMultiHeadAttention(
            self.embed_dim,
            self.attention_heads,
            k_dim=self.embed_dim,
            v_dim=self.embed_dim,
            dropout=self.dropout,
            bias=True,
            add_bias_kv=add_bias_kv,
            add_zero_attn=False,
            attention_type=self.attention_type,
            use_rotary_embeddings=self.use_rotary_embeddings
        )

        # post layer norm
        self.post_layer_norm = LucaLayerNorm(self.embed_dim)

        # dimension increase by the fully connected layer
        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim, bias=True)

        # dimension reduction by the fully connected layer
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim, bias=True)

    def forward(
            self,
            x1,
            x2,
            cross_attn_mask=None,
            cross_attn_padding_mask=None,
            need_head_weights=False
    ):
        residual = x1
        x1 = self.pre_layer_norm(x1)
        x1, attn = self.cross_attn(
            query=x1,
            key=x2,
            value=x2,
            key_padding_mask=cross_attn_padding_mask,
            need_weights=True,
            need_head_weights=need_head_weights,
            attn_mask=cross_attn_mask,
        )
        x1 = residual + x1

        residual = x1
        x1 = self.post_layer_norm(x1)
        x1 = gelu(self.fc1(x1))
        x1 = self.fc2(x1)
        x1 = residual + x1

        return x1, attn


class LucaMultiHeadAttention(nn.Module):
    """
    Cross Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(
            self,
            embed_dim,
            num_heads,
            k_dim=None,
            v_dim=None,
            dropout=0.0,
            bias=True,
            add_bias_kv: bool = False,
            add_zero_attn: bool = False,
            attention_type: str = "self",
            use_rotary_embeddings: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.k_dim = k_dim if k_dim is not None else embed_dim
        self.v_dim = v_dim if v_dim is not None else embed_dim
        self.qkv_same_dim = self.k_dim == embed_dim and self.v_dim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
                self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.attention_type = attention_type

        assert self.attention_type != "self" or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = nn.Linear(self.k_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.v_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False
        self.rot_emb = None
        if use_rotary_embeddings:
            self.rot_emb = RotaryEmbedding(dim=self.head_dim)

        self.enable_torch_version = False
        if hasattr(F, "multi_head_attention_forward"):
            self.enable_torch_version = True
        else:
            self.enable_torch_version = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        '''
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)
        '''
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1.0 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1.0 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1.0 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
            self,
            query,
            key: Optional[torch.Tensor],
            value: Optional[torch.Tensor],
            key_padding_mask: Optional[torch.Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]] = None,
            need_weights: bool = True,
            static_kv: bool = False,
            attn_mask: Optional[torch.Tensor] = None,
            before_softmax: bool = False,
            need_head_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Input shape: Time x Batch x Channel
        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if (
                not self.rot_emb
                and self.enable_torch_version
                and not self.onnx_trace
                and incremental_state is None
                and not static_kv
                # A workaround for quantization to work. Otherwise JIT compilation
                # treats bias in linear module as method.
                and not torch.jit.is_scripting()
                and not need_head_weights
        ):
            assert key is not None and value is not None
            return F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                torch.empty([0]),
                torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                self.training,
                key_padding_mask,
                need_weights,
                attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
            )
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.attention_type == "cross"
                    key = value = None
        else:
            saved_state = None
        if self.attention_type == "self":
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.attention_type == "cross":
            # cross attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        elif self.attention_type == "encoder_decoder":
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)

        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[torch.Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = self._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )
            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)

        assert k is not None
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask),
                    ],
                    dim=1,
                )

        if self.rot_emb:
            q, k = self.rot_emb(q, k)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = utils_softmax(attn_weights, dim=-1, onnx_trace=self.onnx_trace)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(
            attn_weights_float.type_as(attn_weights),
            p=self.dropout,
            training=self.training,
        )
        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[torch.Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).type_as(attn).transpose(1, 0)
            if not need_head_weights:
                # 每个头取平均值
                attn_weights = attn_weights.mean(dim=0)
        return attn, attn_weights

    @classmethod
    def _append_prev_key_padding_mask(
            cls,
            key_padding_mask: Optional[torch.Tensor],
            prev_key_padding_mask: Optional[torch.Tensor],
            batch_size: int,
            src_len: int,
            static_kv: bool,
    ) -> Optional[torch.Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - prev_key_padding_mask.size(1)),
                device=prev_key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), filler.float()], dim=1
            )
        elif key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - key_padding_mask.size(1)),
                device=key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat([filler.float(), key_padding_mask.float()], dim=1)
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    @torch.jit.export
    def reorder_incremental_state(
            self, incremental_state: Dict[str, Dict[str, Optional[torch.Tensor]]], new_order: torch.Tensor
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention and input_buffer_k.size(0) == new_order.size(
                            0
                    ):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
            self, incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]]
    ) -> Dict[str, Optional[torch.Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[torch.Tensor]] = {}
            return empty_result

    def _set_input_buffer(
            self,
            incremental_state: Dict[str, Dict[str, Optional[torch.Tensor]]],
            buffer: Dict[str, Optional[torch.Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    @classmethod
    def apply_sparse_mask(cls, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    @classmethod
    def upgrade_state_dict_named(cls, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim: 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim:]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][dim: 2 * dim]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim:]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value


class LucaTransformer(nn.Module):
    def __init__(self, config, emb_layer=None, use_pretrained_embedding=False, add_pooling_layer=True):
        super().__init__()
        self.config = config
        self.use_pretrained_embedding = use_pretrained_embedding
        self.add_pooling_layer = add_pooling_layer
        if hasattr(config, "padding_idx"):
            self.padding_idx = config.padding_idx
        if hasattr(config, "pad_token_id"):
            self.padding_idx = config.pad_token_id
        self.use_luca_layer_norm_v2 = True
        if hasattr(config, "use_luca_layer_norm_v2"):
            self.use_luca_layer_norm_v2 = config.use_luca_layer_norm_v2

        if emb_layer is not None:
            print("The intra attention module use the exists embedding layer!")
            self.embeddings = emb_layer
        elif use_pretrained_embedding:
            self.embeddings = nn.Linear(config.embedding_input_size, config.hidden_size)
        else:
            self.embeddings = LucaEmbeddings(config)

        self.encoder = nn.ModuleList(
            [
                LucaTransformerLayer(
                    self.config.hidden_size,
                    self.config.intermediate_size,
                    self.config.num_attention_heads,
                    dropout=self.config.attention_probs_dropout_prob,
                    add_bias_kv=False,
                    use_luca_layer_norm_v2=self.use_luca_layer_norm_v2,
                    attention_type="self",
                    use_rotary_embeddings=True if config.no_position_embeddings else False
                )
                for _ in range(self.config.num_hidden_layers)
            ]
        )
        self.layer_size = self.config.num_hidden_layers

        luca_layer_norm = LucaLayerNormV2 if self.use_luca_layer_norm_v2 else LucaLayerNormV1

        self.last_layer_norm = luca_layer_norm(self.config.hidden_size)

        self.pooler = LucaPooler(config) if add_pooling_layer else None

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
    ):
        if self.use_pretrained_embedding:
            x = self.embeddings(inputs_embeds)
        else:
            x = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds
            )
        if attention_mask is None:
            padding_mask = input_ids.eq(self.padding_idx)
        else:
            padding_mask = attention_mask.eq(self.padding_idx)

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        if not padding_mask.any():
            padding_mask = None

        # (B, L, E) => (L, B, E)
        x = x.transpose(0, 1)

        for layer_idx, layer in enumerate(self.encoder):
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                self_attn_mask=None,
                need_head_weights=True
            )

        x = self.last_layer_norm(x)
        x = x.transpose(0, 1)  # (L, B, E) => (B, L, E)
        matrix_output = x
        pooled_output = self.pooler(matrix_output) if self.pooler is not None else None
        return matrix_output, pooled_output


class LucaCrossTransformer(nn.Module):
    def __init__(self, config, emb_layer_a=None, emb_layer_b=None, shared=False, use_pretrained_embedding=False, add_pooling_layer=True):
        super().__init__()
        self.config = config
        self.use_pretrained_embedding = use_pretrained_embedding
        self.add_pooling_layer = add_pooling_layer
        if hasattr(config, "padding_idx"):
            self.padding_idx = config.padding_idx
        if hasattr(config, "pad_token_id"):
            self.padding_idx = config.pad_token_id
        self.use_luca_layer_norm_v2 = True
        if hasattr(config, "use_luca_layer_norm_v2"):
            self.use_luca_layer_norm_v2 = config.use_luca_layer_norm_v2
        self.shared = shared

        if emb_layer_a is not None:
            print("The inter attention module use the exists embedding layer!")
            self.embeddings_a = emb_layer_a
        elif use_pretrained_embedding:
            self.embeddings_a = nn.Linear(config.embedding_input_size, config.hidden_size)
        else:
            self.embeddings_a = LucaEmbeddings(config)
        if emb_layer_b is not None:
            print("The inter attention module use the exists embedding layer!")
            self.embeddings_b = emb_layer_b
        elif use_pretrained_embedding:
            self.embeddings_b = nn.Linear(config.embedding_input_size, config.hidden_size)
        else:
            self.embeddings_b = LucaEmbeddings(config)

        self.encoder_a = nn.ModuleList(
            [
                LucaCrossTransformerLayer(
                    self.config.hidden_size,
                    self.config.intermediate_size,
                    self.config.num_attention_heads,
                    dropout=self.config.attention_probs_dropout_prob,
                    add_bias_kv=False,
                    use_luca_layer_norm_v2=self.use_luca_layer_norm_v2,
                    attention_type="cross",
                    use_rotary_embeddings=True if config.no_position_embeddings else False
                    )
                for _ in range(self.config.num_hidden_layers)
            ]
        )
        if self.shared:
            self.encoder_b = self.encoder_a
        else:
            self.encoder_b = nn.ModuleList(
                [
                    LucaCrossTransformerLayer(
                        self.config.hidden_size,
                        self.config.intermediate_size,
                        self.config.num_attention_heads,
                        dropout=self.config.attention_probs_dropout_prob,
                        add_bias_kv=False,
                        use_luca_layer_norm_v2=self.use_luca_layer_norm_v2,
                        attention_type="cross",
                        use_rotary_embeddings=True if config.no_position_embeddings else False
                    )
                    for _ in range(self.config.num_hidden_layers)
                ]
            )
        self.layer_size = self.config.num_hidden_layers

        luca_layer_norm = LucaLayerNormV2 if self.use_luca_layer_norm_v2 else LucaLayerNormV1

        self.last_layer_norm_a = luca_layer_norm(self.config.hidden_size)
        if self.shared:
            self.last_layer_norm_b = self.last_layer_norm_a
        else:
            self.last_layer_norm_b = luca_layer_norm(self.config.hidden_size)

        self.pooler_a = LucaPooler(config) if add_pooling_layer else None
        if self.shared:
            self.pooler_b = self.pooler_a
        else:
            self.pooler_b = LucaPooler(config) if add_pooling_layer else None

    def forward(
            self,
            input_ids_a: Optional[torch.Tensor] = None,
            attention_mask_a: Optional[torch.Tensor] = None,
            token_type_ids_a: Optional[torch.Tensor] = None,
            position_ids_a: Optional[torch.Tensor] = None,
            inputs_embeds_a: Optional[torch.Tensor] = None,
            input_ids_b: Optional[torch.Tensor] = None,
            attention_mask_b: Optional[torch.Tensor] = None,
            token_type_ids_b: Optional[torch.Tensor] = None,
            position_ids_b: Optional[torch.Tensor] = None,
            inputs_embeds_b: Optional[torch.Tensor] = None,
    ):
        if self.use_pretrained_embedding:
            # a.shape and b.shape must be same
            max_seq_len = max(inputs_embeds_a.shape[1], inputs_embeds_b.shape[1])
            btz_a, seq_len_a, dim_a = inputs_embeds_a.shape
            if seq_len_a != max_seq_len:
                inputs_embeds_a = torch.cat([inputs_embeds_a, torch.zeros(btz_a, max_seq_len - seq_len_a, dim_a, dtype=torch.float32, device=inputs_embeds_a.device)], dim=1)
                if attention_mask_a is not None:
                    attention_mask_a = torch.cat([attention_mask_a, self.padding_idx + torch.zeros(btz_a, max_seq_len - seq_len_a, dtype=torch.long, device=attention_mask_a.device)], dim=1)
            btz_b, seq_len_b, dim_b = inputs_embeds_b.shape
            if seq_len_b != max_seq_len:
                inputs_embeds_b = torch.cat([inputs_embeds_b, torch.zeros(btz_b, max_seq_len - seq_len_b, dim_b, dtype=torch.float32, device=inputs_embeds_b.device)], dim=1)
                if attention_mask_b is not None:
                    attention_mask_b = torch.cat([attention_mask_b, self.padding_idx + torch.zeros(btz_b, max_seq_len - seq_len_b, dtype=torch.long, device=attention_mask_b.device)], dim=1)

            x_a = self.embeddings_a(inputs_embeds_a)
            x_b = self.embeddings_b(inputs_embeds_b)
        else:
            # a.shape and b.shape must be same
            max_seq_len = max(input_ids_a.shape[1], input_ids_b.shape[1])
            btz_a, seq_len_a = input_ids_a.shape
            if seq_len_a != max_seq_len:
                input_ids_a = torch.cat([input_ids_a, self.padding_idx + torch.zeros(btz_a, max_seq_len - seq_len_a, dtype=torch.long, device=input_ids_a.device)], dim=1)
                if attention_mask_a is not None:
                    attention_mask_a = torch.cat([attention_mask_a, self.padding_idx + torch.zeros(btz_a, max_seq_len - seq_len_a, dtype=torch.long, device=attention_mask_a.device)], dim=1)
            x_a = self.embeddings_a(
                input_ids=input_ids_a,
                position_ids=position_ids_a,
                token_type_ids=token_type_ids_a,
                inputs_embeds=inputs_embeds_a
            )
            btz_b, seq_len_b = input_ids_b.shape
            if seq_len_b != max_seq_len:
                input_ids_b = torch.cat([input_ids_b, self.padding_idx + torch.zeros(btz_b, max_seq_len - seq_len_b, dtype=torch.long, device=input_ids_b.device)], dim=1)
                if attention_mask_b is not None:
                    attention_mask_b = torch.cat([attention_mask_b, self.padding_idx + torch.zeros(btz_b, max_seq_len - seq_len_b, dtype=torch.long, device=attention_mask_b.device)], dim=1)
            x_b = self.embeddings_b(
                input_ids=input_ids_b,
                position_ids=position_ids_b,
                token_type_ids=token_type_ids_b,
                inputs_embeds=inputs_embeds_b
            )

        if attention_mask_a is None:
            padding_mask_a = input_ids_a.eq(self.padding_idx)
            attention_mask_a = ~padding_mask_a
        else:
            padding_mask_a = attention_mask_a.eq(self.padding_idx)

        if padding_mask_a is not None:
            x_a = x_a * (1 - padding_mask_a.unsqueeze(-1).type_as(x_a))

        if attention_mask_b is None:
            padding_mask_b = input_ids_b.eq(self.padding_idx)
            attention_mask_b = ~padding_mask_b
        else:
            padding_mask_b = attention_mask_b.eq(self.padding_idx)

        if padding_mask_b is not None:
            x_b = x_b * (1 - padding_mask_b.unsqueeze(-1).type_as(x_b))

        if not padding_mask_a.any():
            padding_mask_a = None

        if not padding_mask_b.any():
            padding_mask_b = None

        # (B, L, E) => (L, B, E)
        x_a = x_a.transpose(0, 1)
        # (B, L, E) => (L, B, E)
        x_b = x_b.transpose(0, 1)
        for layer_idx, layer in enumerate(self.encoder_a):
            x_a, attn_a = layer(
                x_a,
                x_b,
                cross_attn_padding_mask=padding_mask_b,
                cross_attn_mask=None,
                need_head_weights=True
            )
        for layer_idx, layer in enumerate(self.encoder_b):
            x_b, attn_b = layer(
                x_b,
                x_a,
                cross_attn_padding_mask=padding_mask_a,
                cross_attn_mask=None,
                need_head_weights=True
            )

        x_a = self.last_layer_norm_a(x_a)
        x_a = x_a.transpose(0, 1)  # (L, B, E) => (B, L, E)
        x_b = self.last_layer_norm_b(x_b)
        x_b = x_b.transpose(0, 1)  # (L, B, E) => (B, L, E)
        matrix_output_a = x_a
        pooled_output_a = self.pooler_a(matrix_output_a) if self.pooler_a is not None else None
        matrix_output_b = x_b
        pooled_output_b = self.pooler_b(matrix_output_b) if self.pooler_b is not None else None
        return matrix_output_a, pooled_output_a, matrix_output_b, pooled_output_b, attention_mask_a, attention_mask_b


class LucaPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class LucaEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        if hasattr(config, "no_token_embeddings"):
            self.no_token_embeddings = config.no_token_embeddings
        else:
            self.no_token_embeddings = False
        if not self.no_token_embeddings:
            self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        if hasattr(config, "no_position_embeddings"):
            self.no_position_embeddings = config.no_position_embeddings
        else:
            self.no_position_embeddings = False
        if hasattr(config, "no_token_type_embeddings"):
            self.no_token_type_embeddings = config.no_token_type_embeddings
        else:
            self.no_token_type_embeddings = False
        if not self.no_position_embeddings:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        if not self.no_token_type_embeddings:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        # self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        if not self.no_position_embeddings:
            self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        if not self.no_token_type_embeddings and not self.no_position_embeddings:
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long),
                persistent=False,
            )

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if not self.no_position_embeddings and position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if not self.no_token_type_embeddings and token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device if input_ids is not None else inputs_embeds.device)
        if self.no_token_embeddings and inputs_embeds is None:
            raise Exception("The model has not token_embeddings layer, the inputs_embeds cannot None")

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        embeddings = inputs_embeds

        if not self.no_token_type_embeddings:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings

        if not self.no_position_embeddings and self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        # embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


if __name__ == "__main__":
    def printf(value, visible=False):
        if visible:
            print(value)

    import numpy as np
    self = LucaTransformerLayer(embed_dim=4, ffn_embed_dim=4 * 4, attention_heads=1, use_luca_layer_norm_v2=True)
    print(self)
    x = torch.tensor(np.array([[[0.1, 0.2, 0.1, -0.1], [0.2, 0.1, 0.1, -0.1], [0.1, 0.2, 0.1, -0.1]], [[0.1, 0.2, 0.1, -0.1], [0.1, 0.2, 0.1, -0.1], [0.1, 0.2, 0.1, -0.1]]]), dtype=torch.float32)
    # self_attn_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long)
    self_attn_mask = None
    self_attn_padding_mask = torch.tensor([[0, 0, 1], [0, 0, 0]], dtype=torch.long)

    printf("input:")
    printf(x)
    printf(x.shape)
    x = x.transpose(0, 1)
    x, atten = self(x, self_attn_padding_mask=self_attn_padding_mask, self_attn_mask=self_attn_mask, need_head_weights=True)
    x = x.transpose(0, 1)
    printf("output:")
    printf(x)
    printf(x.shape)
    printf("atten:")
    printf(atten)
    printf(atten.shape)
    attentions = [atten.transpose(1, 0)]
    attentions = torch.stack(attentions, 1)
    printf("attentions:")
    printf(attentions)
    printf(attentions.shape)

    attention_mask = 1 - self_attn_padding_mask.type_as(attentions)
    printf("attention_mask:")
    printf(attention_mask)
    printf(attention_mask.shape)
    attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
    printf("attention_mask:")
    printf(attention_mask)
    printf(attention_mask.shape)
    attentions = attentions * attention_mask[:, None, None, :, :]
    printf("attentions:")
    printf(attentions)
    printf(attentions.shape)


    cross = LucaCrossTransformerLayer(embed_dim=4, ffn_embed_dim=4 * 4, attention_heads=1, use_luca_layer_norm_v2=True)
    print(cross)
    x1 = torch.tensor(np.array([[[0.1, 0.2, 0.1, -0.1], [0.2, 0.1, 0.1, -0.1], [0.1, 0.2, 0.1, -0.1]], [[0.1, 0.2, 0.1, -0.1], [0.1, 0.2, 0.1, -0.1], [0.1, 0.2, 0.1, -0.1]]]), dtype=torch.float32)
    x2 = torch.tensor(np.array([[[0.1, 0.2, 0.1, -0.1], [0.2, 0.1, 0.1, -0.1], [0.1, 0.2, 0.1, -0.1]], [[0.1, 0.2, 0.1, -0.1], [0.1, 0.2, 0.1, -0.1], [0.1, 0.2, 0.1, -0.1]]]), dtype=torch.float32)
    cross_attn_mask = None
    cross_attn_padding_mask = torch.tensor([[0, 0, 1], [0, 0, 0]], dtype=torch.long)

    printf("input:")
    printf(x1)
    printf(x1.shape)
    printf(x2)
    printf(x2.shape)
    x1 = x1.transpose(0, 1)
    x2 = x2.transpose(0, 1)
    x, atten = cross(x1, x2, cross_attn_padding_mask=cross_attn_padding_mask, cross_attn_mask=cross_attn_mask, need_head_weights=True)
    x = x.transpose(0, 1)
    printf("output:")
    printf(x)
    printf(x.shape)
    printf("atten:")
    printf(atten)
    printf(atten.shape)
    attentions = [atten.transpose(1, 0)]
    attentions = torch.stack(attentions, 1)
    printf("attentions:")
    printf(attentions)
    printf(attentions.shape)

    attention_mask = 1 - self_attn_padding_mask.type_as(attentions)
    printf("attention_mask:")
    printf(attention_mask)
    printf(attention_mask.shape)
    attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
    printf("attention_mask:")
    printf(attention_mask)
    printf(attention_mask.shape)
    attentions = attentions * attention_mask[:, None, None, :, :]
    printf("attentions:")
    printf(attentions)
    printf(attentions.shape)







