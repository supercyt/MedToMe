# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------


from typing import Tuple

import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer
from copy import copy

from ..merge import merge_source, merge_wavg, shallow_bsm, deep_bsm


class MedToMeBlock(Block):
    """
    Modifications:
     - Apply MedToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """
    def init_skip_lam(self, skip_lam=1):
        self.skip_lam = skip_lam

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_attn, metric, attn = self.attn(self.norm1(x), self._info["size"])
        x = x + self._drop_path1(x_attn) / self.skip_lam

        ratio = self._info["ratio"].pop(0)
        use_cls_metric = self._info["use_cls_metric"].pop(0)

        if ratio < 1.0:
            if not use_cls_metric:
                merge = shallow_bsm(
                    metric=metric,
                    ratio=ratio,
                    class_token=self._info["class_token"],
                )

                if self._info["trace_source"]:
                    self._info["source"] = merge_source(
                        merge, x, self._info["source"]
                    )

                x, self._info["size"] = merge_wavg(merge, x, self._info["size"])
            else:
                B, N, C = x.shape
                # importance metric
                cls_attn = attn[:, :, 0, 1:]
                cls_attn = cls_attn.mean(dim=1)  # [B, N-1]
                _, idx = torch.sort(cls_attn, descending=True)
                cls_index = torch.zeros((B, 1), device=idx.device).long()
                idx = torch.cat((cls_index, idx + 1), dim=1)
                # sorting
                x = torch.gather(x, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
                if self._info["size"] is not None:
                    self._info["size"] = torch.gather(self._info["size"], dim=1, index=idx.unsqueeze(-1))
                merge = deep_bsm(
                    metric=x.detach(),
                    ratio=ratio,
                    class_token=self._info["class_token"],
                )

                if self._info["trace_source"]:
                    self._info["source"] = merge_source(
                        merge, x, self._info["source"]
                    )

                x, self._info["size"] = merge_wavg(merge, x, self._info["size"])

            # if self._info["size"] is None:
            #     self._info["size"] = torch.ones_like(x[..., 0, None])
            #
            # x = merge(x, mode="max")
            # # x = merge(x * self._info["size"], mode="sum")
            # self._info["size"] = merge(self._info["size"], mode="sum")
            # # x = x / size

        x = x + self._drop_path2(self.mlp(self.norm2(x))) / self.skip_lam
        return x


class MedToMeAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply proportional attention
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Return k as well here
        return x, k.mean(1), attn

