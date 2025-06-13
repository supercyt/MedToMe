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
import torch.nn as nn
from timm.models.vision_transformer import Attention, Block, VisionTransformer
from .timm import PiToMeAttention, PiToMeBlock, PiToMeBlock
import math
from tlt.lvvit_layers import Block as LvBlock
from tlt.lvvit_layers import Attention as LvAttention



def make_pitome_class(transformer_class):
    class PiToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, x, return_flop=False) -> torch.Tensor:
      
            self._info["ratio"] = [self.ratio] * len(self.blocks)
            # self._info["use_bsm_pitome"] = [False] * (len(self.blocks)//2) + [True] * (len(self.blocks)//2)
            num_bsm_layers = math.ceil(len(self.blocks) * 0.5)
            self._info["use_bsm_pitome"] = [True] * (num_bsm_layers) + [False] * (len(self.blocks)-num_bsm_layers)
            self._info["size"] = None
            self._info["source"] = None
            self.total_flop = 0

            x = super().forward(x)
            if return_flop:
                return x, self.total_flop
            else:
                return x


        # def forward_features(self, x):
        #     x = self.patch_embed(x)
        #     cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        #     if self.dist_token is None:
        #         x = torch.cat((cls_token, x), dim=1)
        #     else:
        #         x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        #     x = self.pos_drop(x + self.pos_embed)
        #     for block in self.blocks:
        #         self.total_flop += self.calculate_block_flop(x.shape)
        #         x = block(x)
        #     x = self.norm(x)
        #     if self.dist_token is None:
        #         return self.pre_logits(x[:, 0])
        #     else:
        #         return x[:, 0], x[:, 1]
 
        def calculate_block_flop(self, shape):
            flops = 0
            _, N, C = shape
            mhsa_flops = 4*N*C*C + 2*N*N*C
            flops += mhsa_flops
            ffn_flops = 8*N*C*C
            flops += ffn_flops
            return flops


    return PiToMeVisionTransformer



def apply_patch(
   model: VisionTransformer, trace_source: bool = False, prop_attn: bool = False, skip_lam=1):

    PiToMeVisionTransformer = make_pitome_class(model.__class__)
    print('using', 'pitome')

    model.__class__ = PiToMeVisionTransformer
    model.ratio = 1.0
    
    model._info = {
        "ratio": model.ratio,
        "margin":  [],
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "distill_token": False,
    }
    current_layer = 0
    num_layers = len(model.blocks)
    margins = [.75 - .75*(i/num_layers) for i in range(num_layers)]

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, Block) or isinstance(module, LvBlock):
            module.__class__ = PiToMeBlock
            module.init_margin(margins[current_layer])
            module.init_skip_lam(skip_lam)
            module._info = model._info
            current_layer +=1
        elif isinstance(module, Attention) or isinstance(module, LvAttention):
            module.__class__ = PiToMeAttention
