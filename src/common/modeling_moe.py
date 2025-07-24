#!/usr/bin/env python
# encoding: utf-8
"""
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2024/6/25 14:20
@project: LucaX
@file: modeling_moe
@desc: Two implementations of MoE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN


class FFNExpert(nn.Module):
    """
    An FFN expert similar to the FFN in Transformer Layers.
    """
    def __init__(self, d_model, hidden_dim, activation_function):
        super(FFNExpert, self).__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.activation_function = activation_function
        self.input_projection = nn.Linear(d_model, hidden_dim)
        self.activation_fn = ACT2FN[activation_function]
        self.output_projection = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.activation_fn(x)
        x = self.output_projection(x)
        return x


class Router(nn.Module):
    """
    Router to distribute tokens to experts.
    """
    def __init__(self, d_model, num_experts):
        super(Router, self).__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.layer = nn.Linear(d_model, num_experts)

    def forward(self, x):
        return F.softmax(self.layer(x), dim=-1)


class MoE1(nn.Module):
    """
    Mixture of Experts
    """
    def __init__(self, d_model, num_experts, hidden_dim, activation_function, top_k=2):
        super(MoE1, self).__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.activation_function = activation_function
        self.top_k = top_k
        self.router = Router(self.d_model, self.num_experts)
        self.experts = nn.ModuleList([FFNExpert(d_model=self.d_model,
                                                hidden_dim=self.hidden_dim,
                                                activation_function=self.activation_function) for _ in range(num_experts)])

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape

        # Flatten batch and sequence dimensions for the gate
        # shape: (batch_size * seq_len, d_model)
        x_flat = x.view(-1, d_model)

        # Gate calculations
        # shape: (batch_size * seq_len, num_experts)
        gate_weights = self.router(x_flat)

        # Select Top-K experts for each input
        # shape: (batch_size * seq_len, top_k)
        top_k_weights, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        top_k_weights_normalized = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # Calculate expert outputs for selected Top-K experts
        # Initialize aggregated expert outputs
        expert_outputs = torch.zeros(batch_size * seq_len, d_model, device=x.device)

        for i in range(self.top_k):
            # shape: (batch_size * seq_len,)
            expert_idx = top_k_indices[:, i]
            mask = torch.zeros_like(gate_weights)
            # 第topk的mask，True位置赋1
            # shape: (batch_size * seq_len, top_k)
            # 只有每个token对应的topk的mask为1
            mask[torch.arange(mask.size(0)), expert_idx] = 1
            # (batch_size * seq_len, top_k, 1) * (batch_size * seq_len, d_model)
            # print("mask.unsqueeze(2):")
            # [1482, 8, 1]
            # print("x_flat.unsqueeze(1):")
            # [1482, 1, 1280]
            # mask 操作
            masked_inputs = (mask.unsqueeze(2) * x_flat.unsqueeze(1)).sum(dim=1)
            # print("masked_inputs:")
            # [1482, 1280]
            expert_output = torch.zeros_like(masked_inputs)
            for j in range(self.num_experts):
                if (expert_idx == j).any():
                    # print("mask[:, j].unsqueeze(1)")
                    # [1482, 1]
                    # print("self.experts[j](masked_inputs) ")
                    # [1482, 1280]
                    expert_output += self.experts[j](masked_inputs) * mask[:, j].unsqueeze(1)
            # print("top_k_weights_normalized[:, i].unsqueeze(1):")
            # [1482, 1]
            expert_outputs += top_k_weights_normalized[:, i].unsqueeze(1) * expert_output

        # Reshape back to original shape
        expert_outputs = expert_outputs.view(batch_size, seq_len, d_model)

        return expert_outputs


class Expert(nn.Module):
    """
    Define the Expert class
    """
    def __init__(self, d_model, hidden_dim, activation_function):
        super(Expert, self).__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.activation_function = activation_function
        self.input_projection = nn.Linear(d_model, hidden_dim)
        self.activation_fn = ACT2FN[activation_function]
        self.output_projection = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        x = self.input_projection(x)
        x = self.activation_fn(x)
        x = self.output_projection(x)
        # -> shape: (batch_size, seq_len, embed_dim)
        return x


class Gate(nn.Module):
    """
    Define the Gating Network class
    """
    def __init__(self, d_model, num_experts):
        super(Gate, self).__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.layer = nn.Linear(d_model, num_experts)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        # -> (batch_size, seq_len, num_experts)
        return F.softmax(self.layer(x), dim=2)


class MoE(nn.Module):
    """
    Mixture of Experts
    """
    def __init__(self, d_model, hidden_dim, activation_function, num_experts, top_k=2):
        super(MoE, self).__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.activation_function = activation_function
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = Gate(d_model=self.d_model, num_experts=self.num_experts)
        self.experts = nn.ModuleList([Expert(d_model=self.d_model,
                                             hidden_dim=self.hidden_dim,
                                             activation_function=self.activation_function
                                             )
                                      for _ in range(self.num_experts)])

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        # gating_scores: (batch_size, seq_len, num_experts)
        gating_scores = self.gate(x)
        top_k_gating_scores, top_k_indices = gating_scores.topk(k=self.top_k, dim=2, sorted=False)
        # Create a mask to zero out the contributions of non-topk experts
        # mask: (batch_size, seq_len, num_experts)
        mask = torch.zeros_like(gating_scores).scatter_(2, top_k_indices, 1)
        # Use the mask to retain only the topk gating scores
        # gating_scores: (batch_size, seq_len, num_experts)
        gating_scores = gating_scores * mask
        # Normalize the gating scores to sum to 1 across the selected top experts
        # gating_scores: (batch_size, seq_len, num_experts)
        # mask掉的部分为0
        gating_scores = F.normalize(input=gating_scores, p=1, dim=2)
        # expert_outputs: (batch_size, num_experts, seq_len, d_model)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        # expert_outputs: (batch_size, seq_len, num_experts, d_model)
        expert_outputs = expert_outputs.transpose(1, 2)
        # (batch_size, seq_len, num_experts), (batch_size, seq_len, num_experts, d_model)
        # -> (batch_size, seq_len, d_model)
        output = torch.einsum('bte,bteo->bto', gating_scores, expert_outputs)
        return output


if __name__ == "__main__":
    """
    两种实现方式结果一样，MoE效率比MoE1高
    """
    moe = MoE(d_model=2,
              hidden_dim=4,
              activation_function='gelu',
              num_experts=8,
              top_k=2)
    input_x = torch.tensor([[[0.1, 0.2], [0.2, 0.4], [0.3, 0.6]],
                      [[0.2, 0.4], [0.4, 0.8], [0.8, 1.6]],
                      [[0.1, 0.2], [0.2, 0.4], [0.3, 0.6]],
                      [[0.2, 0.4], [0.4, 0.8], [0.8, 1.6]],
                      ], dtype=torch.float)
    print(moe(input_x))

    moe1 = MoE1(d_model=2,
                hidden_dim=4,
                activation_function='gelu',
                num_experts=8,
                top_k=2)
    moe1.router.layer.weight = moe.gate.layer.weight
    moe1.router.layer.bias = moe.gate.layer.bias
    for idx in range(8):
        moe1.experts[idx].input_projection.weight = moe.experts[idx].input_projection.weight
        moe1.experts[idx].input_projection.bias = moe.experts[idx].input_projection.bias
        moe1.experts[idx].output_projection.weight = moe.experts[idx].output_projection.weight
        moe1.experts[idx].output_projection.bias = moe.experts[idx].output_projection.bias
    print(moe1(input_x))