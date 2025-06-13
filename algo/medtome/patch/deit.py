import math
from typing import Tuple

import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer
# from timm.models.helpers import checkpoint_seq 
from .timm import MedToMeBlock, MedToMeBlock, MedToMeAttention
from tlt.lvvit_layers import Block as LvBlock
from tlt.lvvit_layers import Attention as LvAttention


def make_medtome_class(transformer_class):
    class MedToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, x, return_flop=True) -> torch.Tensor:

            self._info["ratio"] = [self.ratio] * len(self.blocks)
            num_cls_layers = math.ceil(len(self.blocks) * 0.1)
            self._info["use_cls_metric"] = [False] * (num_cls_layers) + [True] * (len(self.blocks) - num_cls_layers)
            self._info["size"] = None
            self._info["source"] = None
            self.total_flop = 0

            x = super().forward(x)
            if return_flop:
                return x, self.total_flop
            else:
                return x

    return MedToMeVisionTransformer



def apply_patch(
   model: VisionTransformer, trace_source: bool = False, prop_attn: bool = False, skip_lam=1
):
    """
    Applies MedToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    MedToMeVisionTransformer = make_medtome_class(model.__class__)
    print('using', 'medtome')

    model.__class__ = MedToMeVisionTransformer
    model.r = 0
    model.ratio = 1.0 
    
    # model.compress_method = 'medtome' 
    model._info = {
        "ratio": model.ratio,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": True,
        "distill_token": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._info["distill_token"] = True

    for module in model.modules():

        if isinstance(module, Block) or isinstance(module, LvBlock):
            module.__class__ = MedToMeBlock
            module.init_skip_lam(skip_lam)
            module._info = model._info
        elif isinstance(module, Attention) or isinstance(module, LvAttention):
            module.__class__ = MedToMeAttention
