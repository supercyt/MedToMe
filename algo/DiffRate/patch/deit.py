
import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer
from .timm import DiffRateBlock, DiffRateAttention
from ..utils import ste_min
from tlt.lvvit_layers import Block as LvBlock
from tlt.lvvit_layers import Attention as LvAttention


def make_diffrate_class(transformer_class):
    class DiffRateVisionTransformer(transformer_class):
        def forward(self, x, return_flop=True) -> torch.Tensor:
            B = x.shape[0]
            self._info["size"] = torch.ones([B,self.patch_embed.num_patches+1,1], device=x.device)
            self._info["mask"] =  torch.ones((B,self.patch_embed.num_patches+1),device=x.device)
            self._info["prune_kept_num"] = []
            self._info["merge_kept_num"] = []
            if self._info["trace_source"]:
                self._info["source"] = torch.eye(self.patch_embed.num_patches+1, device=x.device)[None, ...].expand(B, self.patch_embed.num_patches+1, self.patch_embed.num_patches+1)
            x = super().forward(x)
            if return_flop:
                if self.training:
                    flops = self.calculate_flop_training()
                else:
                    flops = self.calculate_flop_inference()
                return x, flops
            else:
                return x
        
        def parameters(self, recurse=True):
            # original network parameter
            params = []
            for n, m in self.named_parameters():
                if n.find('ddp') > -1:
                    continue
                params.append(m)
            return iter(params)    
        
        def arch_parameters(self):
            params = []
            for n, m in self.named_parameters():
                if n.find('ddp') > -1:
                    params.append(m)
            return iter(params)    

        def get_kept_num(self):
            prune_kept_num = []
            merge_kept_num = []
            for block in self.blocks:
                prune_kept_num.append(int(block.prune_ddp.kept_token_number))
                merge_kept_num.append(int(block.merge_ddp.kept_token_number))
            return prune_kept_num, merge_kept_num
                

        def set_kept_num(self, prune_kept_numbers, merge_kept_numbers):
            assert len(prune_kept_numbers) == len(self.blocks) and len(merge_kept_numbers) == len(self.blocks)
            for block, prune_kept_number, merge_kept_number in zip(self.blocks, prune_kept_numbers, merge_kept_numbers):
                block.prune_ddp.kept_token_number = prune_kept_number
                block.merge_ddp.kept_token_number = merge_kept_number
        
        def init_kept_num_using_ratio(self, ratio):
            import math
            N = self.patch_embed.num_patches
            for block in self.blocks:
                r = math.floor(N - N*ratio)
                block.prune_ddp.kept_token_number = N - 0 
                block.merge_ddp.kept_token_number = N - r
                N -= r
            
        def init_kept_num_using_r(self, r):
            N = self.patch_embed.num_patches
            for block in self.blocks:
                r = min(r, N // 2)
                block.prune_ddp.kept_token_number = N - 0 
                block.merge_ddp.kept_token_number = N - r
                N -= r
        
        def calculate_flop_training(self):
            C = self.embed_dim
            patch_number = float(self.patch_embed.num_patches)
            N = torch.tensor(patch_number+1, device=self.blocks[0].prune_ddp.selected_probability.device)
            flops = 0
            patch_embedding_flops = N*C*(self.patch_embed.patch_size[0]*self.patch_embed.patch_size[1]*3)
            classifier_flops = C*self.num_classes
            with torch.cuda.amp.autocast(enabled=False):
                for prune_kept_number, merge_kept_number in zip(self._info["prune_kept_num"],self._info["merge_kept_num"]):
                    prune_kept_number = prune_kept_number.float()     
                    merge_kept_number = merge_kept_number.float()
                    mhsa_flops = 4*N*C*C + 2*N*N*C
                    flops += mhsa_flops
                    N = ste_min(N, prune_kept_number, merge_kept_number)
                    ffn_flops = 8*N*C*C
                    flops += ffn_flops
            flops += patch_embedding_flops
            flops += classifier_flops
            return flops

        def calculate_flop_inference(self):
            C = self.embed_dim
            patch_number = float(self.patch_embed.num_patches)
            N = torch.tensor(patch_number+1, device=self.blocks[0].prune_ddp.selected_probability.device)
            flops = 0
            patch_embedding_flops = N*C*(self.patch_embed.patch_size[0]*self.patch_embed.patch_size[1]*3)
            classifier_flops = C*self.num_classes
            with torch.cuda.amp.autocast(enabled=False):
                for block in (self.blocks):
                    prune_kept_number = block.prune_ddp.kept_token_number
                    merge_kept_number = block.merge_ddp.kept_token_number
                    mhsa_flops = 4*N*C*C + 2*N*N*C
                    flops += mhsa_flops
                    N = ste_min(N, prune_kept_number, merge_kept_number)
                    ffn_flops = 8*N*C*C
                    flops += ffn_flops
            flops += patch_embedding_flops
            flops += classifier_flops
            return flops
        

    return DiffRateVisionTransformer


def apply_patch(
    model: VisionTransformer, trace_source: bool = False,prune_granularity=1, merge_granularity=1, skip_lam = 1
):
    """
    Applies DiffRate to this transformer.
    """
    print('using', 'diffrate')
    DiffRateVisionTransformer = make_diffrate_class(model.__class__)

    model.__class__ = DiffRateVisionTransformer
    model._info = {
        "size": None,
        "mask": None,           # only for training
        "source": None,
        "class_token": model.cls_token is not None,
        "trace_source": trace_source,
    }

    block_index = 0
    non_compressed_block_index = [0]
    for module in model.modules():
        if isinstance(module, Block) or isinstance(module, LvBlock):
            module.__class__ = DiffRateBlock
            if block_index in non_compressed_block_index:
                module.introduce_diffrate(model.patch_embed.num_patches, model.patch_embed.num_patches+1, model.patch_embed.num_patches+1)
            else:
                module.introduce_diffrate(model.patch_embed.num_patches, prune_granularity, merge_granularity)
            module.init_skip_lam(skip_lam)
            block_index += 1
            module._info = model._info
        elif isinstance(module, Attention) or isinstance(module, LvAttention):
            module.__class__ = DiffRateAttention