#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 Vladislav Lialin and Namrata Shivagunde 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, hidden, num_heads, causal=False):
        """Multi-head attention module which computes [softmax(xQ_h @ xK_h^T) @ xV: ...] @ U

        Can work as both self-attention or cross-attention (if kv is provided to .forward).

        Args:
            causal: use causal masking (do not allow target to look to the future or current token of source)
        """
        if hidden % num_heads:
            raise ValueError(f"hidden should be divisible by num_heads, "
                             f"but got hidden={hidden} and num_heads={num_heads}")
        super().__init__()

        self.k = nn.Linear(input_size, hidden)
        self.q = nn.Linear(input_size, hidden)
        self.v = nn.Linear(input_size, hidden)
        self.mix = nn.Linear(hidden, hidden)

        self.num_heads = num_heads
        self.head_size = hidden // num_heads
        self.scale = self.head_size ** 0.5
        self.causal = causal  # causal masking

    def forward(self, q, kv=None, key_padding_mask=None, return_attention=False):
        """[Softmax(source Q_1 @ target K_1^T) @ target V_1 : ... ) @ x V_heads] @ U

        Performs self-attention if kv is not specified.
        In this case, kv = q and kv_seq_len = query_seq_len.

        Args:
            q: FloatTensor[batch_size, query_seq_len, input_size]
            kv (target) : optional, FloatTensor[batch_size, kv_seq_len, input_size]
            key_padding_mask: BoolTensor[batch_size, kv_seq_len] 0 means unpadded, 1 means padded

        Returns:
            FloatTensor[batch_size, seq_len, hidden]
        """

        # Task 1.1 (1 point)
        # Update this function with cross-attention mechanism
        # If target is None, then target (kv) and source (q) will be same.
        # Define k, q, v using self.k, self.q and self.v based on if the target exists or not 
        # Note : Please write shape of each tensor for each line of code
        ## YOUR CODE STARTS HERE## ~ 2 lines code

        if kv == None:
            k = q = v= self.q(q) #shape [batch_size, source_seq_len, hidden]
        else:
            q = self.q(q) #shape [batch_size, source_seq_len, hidden]
            k = self.k(kv) #shape [batch_size, target_seq_len, hidden]
            v = self.v(kv) #shape [batch_size, target_seq_len, hidden]

        # YOUR CODE ENDS HERE

        bs, attending_seq, _ = q.shape
        attended_seq = k.shape[1]

        # [b, s, h] -> [b, h, s] -> [b * heads, h / heads, s] -> [b * heads, s, h / heads]
        k = k.transpose(1, 2).reshape(bs * self.num_heads, self.head_size, -1).transpose(1, 2).contiguous()  # [batch * num_heads, seq, hidden / num_heads]
        q = q.transpose(1, 2).reshape(bs * self.num_heads, self.head_size, -1).transpose(1, 2).contiguous()
        v = v.transpose(1, 2).reshape(bs * self.num_heads, self.head_size, -1).transpose(1, 2).contiguous()

        scores = q @ k.transpose(1, 2) / self.scale  # [batch * num_heads, attending_seq, attended_seq]
        assert scores.shape == (bs * self.num_heads, attending_seq, attended_seq)

        if key_padding_mask is not None:
            # Task 1.2 (1 point)
            # Padding
            # Set the scores corresponding to padded positions (key_padding_mask == 1) to -inf
            # 
            # You might need to reshape the scores to [batch_size, seq_len, seq_len]
            # in this case, remember to reshape them back
            # Our implementation is 3 lines
            # YOUR CODE STARTS HERE

            resized_mask = key_padding_mask.bool().unsqueeze(1)
            scores = scores.reshape(bs, self.num_heads * attending_seq, attended_seq)
            scores.masked_fill_(resized_mask, float('-inf'))
            scores = scores.reshape(bs * self.num_heads, attending_seq, attended_seq)

            # YOUR CODE ENDS HERE

        assert scores.size() == (bs * self.num_heads, attending_seq, attended_seq),\
            f"scores have wrong shape. Expected {(bs * self.num_heads, attending_seq, attended_seq)}, got {scores.size()}"

        if self.causal:
            causal_mask = torch.triu(torch.ones(attending_seq, attended_seq, dtype=torch.bool, device=scores.device), diagonal=1)
            scores.masked_fill_(causal_mask.bool().unsqueeze(0), float("-inf"))

        probs = torch.softmax(scores, dim=-1)  # [batch * num_heads, tgt_seq, src_seq]
        att = probs @ v  # [batch * num_heads, tgt_seq, hidden / num_heads]

        # [b * heads, s, h / heads] -> [b * heads, h / heads, s] -> [b, h, s] -> [b, s, h]
        att = att.transpose(1, 2).reshape(bs, -1, attending_seq).transpose(1, 2).contiguous()
    
        att = self.mix(att)
        
        if return_attention:
            return att, probs

        return att
