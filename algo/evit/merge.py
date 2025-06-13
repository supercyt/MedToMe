# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
from typing import Callable, Tuple
import torch
import torch.nn.functional as F


def do_nothing(x, mode=None):
    return x


def bipartite_soft_matching(
    metric: torch.Tensor,
    cls_attn: torch.Tensor,
    ratio:float = 1.0,
    class_token: bool = False,
    sigma: float = .0,
) -> Tuple[Callable, Callable]:
    
    protected = 0
    if class_token:
        protected += 1
    # if len(metric.shape) == 2:
    #     metric = metric[None,...]

    # We can only reduce by a maximum of 50% tokens
    T = metric.shape[1]

    if ratio < 1.0:
        r = math.floor(T- T*ratio)
    else:
        return do_nothing, do_nothing

    with torch.no_grad():
        sorted_cls_attn, idx = torch.sort(cls_attn, descending=True)
        topk_attn, topk_idx = sorted_cls_attn[..., :-r], idx[..., :-r]
        non_topk_attn, non_topk_idx = sorted_cls_attn[..., -r:], idx[..., -r:]


    def merge(x: torch.Tensor, mode="mean", training=False) -> torch.Tensor:
        # separate [CLS] token and other token
        cls_token = x[..., 0:1, :]
        x_without_cls = x[..., 1:, :]
        n, t1, c = x_without_cls.shape
        # obtain the attentive and inattentive tokens
        attentive_tokens = x_without_cls.gather(-2, topk_idx.unsqueeze(-1).expand(n, t1 - r, c))
        inattentive_tokens = x_without_cls.gather(-2, non_topk_idx.unsqueeze(-1).expand(n, r, c))
        fused_token = non_topk_attn.unsqueeze(-2) @ inattentive_tokens

        x_new = torch.concat([cls_token, attentive_tokens, fused_token], dim=-2)
        return x_new

    def unmerge():
        pass

    return merge, unmerge


def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")
    x = x / size

    return x, size 



def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source

def merge_attention_mask(
    merge, attention_mask: torch.Tensor
): 

    attention_mask = merge(attention_mask, mode="amax")
    return attention_mask 
