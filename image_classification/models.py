# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
# from performer import PerformerVisionTransformer
from mgk import MGK2keyVisionTransformer
from gmm import GMMVisionTransformer
from resK import ResKVisionTransformer, GaussResKVisionTransformer, ResVVisionTransformer, ResQKVisionTransformer, DenseQKVisionTransformer
from residual import ResAVisionTransformer
from attention import HDPVisionTransformer
from xcit import XCiT, HDPXCiT


__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384', 'deit_tiny_mgk_2head_patch16_224',
    'deit_tiny_mgk_2head_scale3_patch16_224', 'deit_tiny_mgk_2head_scale3_patch16_224_dim72',
    'deit_tiny_gmm_patch16_224', 'deit_tiny_gmm_softe_patch16_224', 'deit_tiny_gmm_gd_patch16_224','deit_tiny_gmm_gd_sharekey_patch16_224',
     'deit_tiny_gmm_gd_cls_patch16_224',
    'deit_tiny_gmm_softe_cls_patch16_224',
    'deit_tiny_gmm_softe_gd_patch16_224', 'deit_tiny_resk_patch16_224',
     'deit_tiny_gaussresk_patch16_224', 'deit_tiny_resk_beta01_patch16_224',
    'deit_tiny_resk_beta001_patch16_224', 'deit_tiny_resv_patch16_224', 'deit_tiny_resk_beta05_patch16_224', 'deit_tiny_resqk_patch16_224', 'deit_tiny_gmm_qk_pi_patch16_224', 'deit_tiny_denseqk_patch16_224', 'deit_tiny_denseqk_norm_patch16_224', 'deit_tiny_resa_patch16_224',
    'deit_tiny_hdp2q_patch16_224', 'deit_tiny_hdp2k_patch16_224', 'deit_tiny_hdp2qk_patch16_224', 'deit_small_hdp2qk_patch16_224', 'deit_small_hdp4qk_patch16_224', 'deit_tiny_hdp2gqk_patch16_224', 'deit_small_hdp_adaptive_qk_patch16_224',
    'xcit_small_12_p16', 'xcit_small_12_p16_hdp_4qk'
]


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2


@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


###########HDP
@register_model
def deit_tiny_hdp2q_patch16_224(pretrained=False, **kwargs):
    model = HDPVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_global_heads = 2, mode ='q', **kwargs) # <name>'s NOTE: in the original code, num_heads = 3 here
    model.default_cfg = _cfg()
    return model

@register_model
def deit_tiny_hdp2k_patch16_224(pretrained=False, **kwargs):
    model = HDPVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_global_heads = 2, mode ='k', **kwargs) # <name>'s NOTE: in the original code, num_heads = 3 here
    model.default_cfg = _cfg()
    return model

@register_model
def deit_tiny_hdp2qk_patch16_224(pretrained=False, **kwargs):
    model = HDPVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_global_heads = 2, mode ='qk', **kwargs) # <name>'s NOTE: in the original code, num_heads = 3 here
    model.default_cfg = _cfg()
    return model

@register_model
def deit_tiny_hdp2gqk_patch16_224(pretrained=False, **kwargs):
    model = HDPVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_global_heads = 2, mode ='gqk', **kwargs) # <name>'s NOTE: in the original code, num_heads = 3 here
    model.default_cfg = _cfg()
    return model

########### GP
@register_model
def deit_tiny_resk_patch16_224(pretrained=False, **kwargs):
    model = ResKVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), use_layernorm = True, **kwargs) # <name>'s NOTE: in the original code, num_heads = 3 here
    model.default_cfg = _cfg()
    return model

@register_model
def deit_tiny_denseqk_patch16_224(pretrained=False, **kwargs):
    model = DenseQKVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), layernorm = False, **kwargs) # <name>'s NOTE: in the original code, num_heads = 3 here
    model.default_cfg = _cfg()
    return model

@register_model
def deit_tiny_denseqk_norm_patch16_224(pretrained=False, **kwargs):
    model = DenseQKVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), layernorm = True, **kwargs) # <name>'s NOTE: in the original code, num_heads = 3 here
    model.default_cfg = _cfg()
    return model

@register_model
def deit_tiny_resqk_patch16_224(pretrained=False, **kwargs):
    model = ResQKVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs) # <name>'s NOTE: in the original code, num_heads = 3 here
    model.default_cfg = _cfg()
    return model

@register_model
def deit_tiny_resa_patch16_224(pretrained=False, **kwargs):
    model = ResAVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs) # <name>'s NOTE: in the original code, num_heads = 3 here
    model.default_cfg = _cfg()
    return model

@register_model
def deit_tiny_resv_patch16_224(pretrained=False, **kwargs):
    model = ResVVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs) # <name>'s NOTE: in the original code, num_heads = 3 here
    model.default_cfg = _cfg()
    return model


@register_model
def deit_tiny_gaussresk_patch16_224(pretrained=False, **kwargs):
    model = GaussResKVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs) # <name>'s NOTE: in the original code, num_heads = 3 here
    model.default_cfg = _cfg()
    return model

@register_model
def deit_tiny_resk_beta01_patch16_224(pretrained=False, **kwargs):
    model = ResKVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), beta = 0.1, **kwargs) # <name>'s NOTE: in the original code, num_heads = 3 here
    model.default_cfg = _cfg()
    return model

@register_model
def deit_tiny_resk_beta001_patch16_224(pretrained=False, **kwargs):
    model = ResKVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), beta = 1e-2, **kwargs) # <name>'s NOTE: in the original code, num_heads = 3 here
    model.default_cfg = _cfg()
    return model

@register_model
def deit_tiny_resk_beta05_patch16_224(pretrained=False, **kwargs):
    model = ResKVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), beta = 0.5, **kwargs) # <name>'s NOTE: in the original code, num_heads = 3 here
    model.default_cfg = _cfg()
    return model

# @register_model
# def deit_tiny_gaussresk_beta05_patch16_224(pretrained=False, **kwargs):
#     model = GaussResKVisionTransformer(
#         patch_size=16, embed_dim=192, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), beta = 0.5, **kwargs) # <name>'s NOTE: in the original code, num_heads = 3 here
#     model.default_cfg = _cfg()
#     return model

########### GMM
@register_model
def deit_tiny_gmm_patch16_224(pretrained=False, **kwargs):
    model = GMMVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), update_mode = None, beta = None, **kwargs) # <name>'s NOTE: in the original code, num_heads = 3 here
    model.default_cfg = _cfg()
    return model


@register_model
def deit_tiny_gmm_softe_patch16_224(pretrained=False, **kwargs):
    model = GMMVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), update_mode = 'soft_em', beta = 0.999, **kwargs) # <name>'s NOTE: in the original code, num_heads = 3 here
    model.default_cfg = _cfg()
    return model

@register_model
def deit_tiny_gmm_softe_09999_patch16_224(pretrained=False, **kwargs):
    model = GMMVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), update_mode = 'soft_em', beta = 0.9999, **kwargs) # <name>'s NOTE: in the original code, num_heads = 3 here
    model.default_cfg = _cfg()
    return model

@register_model
def deit_tiny_gmm_gd_patch16_224(pretrained=False, **kwargs):
    model = GMMVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), update_mode = 'gd', beta = None, **kwargs) # <name>'s NOTE: in the original code, num_heads = 3 here
    model.default_cfg = _cfg()
    return model

@register_model
def deit_tiny_gmm_qk_pi_patch16_224(pretrained=False, **kwargs):
    model = GMMVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), update_mode = 'qk_pi', beta = None, **kwargs) # <name>'s NOTE: in the original code, num_heads = 3 here
    model.default_cfg = _cfg()
    return model

@register_model
def deit_tiny_gmm_gd_cls_patch16_224(pretrained=False, **kwargs):
    model = GMMVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), update_mode = 'gd', beta = None, cls_dealer= True, **kwargs) # <name>'s NOTE: in the original code, num_heads = 3 here
    model.default_cfg = _cfg()
    return model

@register_model
def deit_tiny_gmm_softe_cls_patch16_224(pretrained=False, **kwargs):
    model = GMMVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), update_mode = 'soft_em', beta = 1e-3, cls_dealer= True, **kwargs) # <name>'s NOTE: in the original code, num_heads = 3 here
    model.default_cfg = _cfg()
    return model

@register_model
def deit_tiny_gmm_gd_sharekey_patch16_224(pretrained=False, **kwargs):
    model = GMMVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), update_mode = 'gd', share_key = True, **kwargs) # <name>'s NOTE: in the original code, num_heads = 3 here
    model.default_cfg = _cfg()
    return model

# @register_model
# def deit_tiny_gmm_softe_09999_cls_patch16_224(pretrained=False, **kwargs):
#     model = GMMVisionTransformer(
#         patch_size=16, embed_dim=192, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), update_mode = 'soft_em', beta = 0.9999, cls_dealer= True, **kwargs) # <name>'s NOTE: in the original code, num_heads = 3 here
#     model.default_cfg = _cfg()
#     return model

# @register_model
# def deit_tiny_gmm_softe_099999_cls_patch16_224(pretrained=False, **kwargs):
#     model = GMMVisionTransformer(
#         patch_size=16, embed_dim=192, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), update_mode = 'soft_em', beta = 0.9999, cls_dealer= True, **kwargs) # <name>'s NOTE: in the original code, num_heads = 3 here
#     model.default_cfg = _cfg()
#     return model

@register_model
def deit_tiny_gmm_softe_gd_patch16_224(pretrained=False, **kwargs):
    model = GMMVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), update_mode = 'softe_gd', beta = 1e-3, cls_dealer= True, **kwargs) # <name>'s NOTE: in the original code, num_heads = 3 here
    model.default_cfg = _cfg()
    return model






############
@register_model
def deit_tiny_mgk_2head_patch16_224(pretrained=False, **kwargs):
    model = MGK2keyVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=2, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), head_dim = 48, scale_k1 = 1., **kwargs) # <name>'s NOTE: in the original code, num_heads = 3 here
    model.default_cfg = _cfg()
    return model

@register_model
def deit_tiny_mgk_2head_scale3_patch16_224(pretrained=False, **kwargs):
    model = MGK2keyVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=2, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), head_dim = 48, scale_k1 = 3., **kwargs) # <name>'s NOTE: in the original code, num_heads = 3 here
    model.default_cfg = _cfg()
    return model

@register_model
def deit_tiny_mgk_2head_scale3_patch16_224_dim72(pretrained=False, **kwargs):
    model = MGK2keyVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=2, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), head_dim = 72, scale_k1 = 3., **kwargs) # <name>'s NOTE: in the original code, num_heads = 3 here
    model.default_cfg = _cfg()
    return model

##### OUR MODEL TO TRY
# class PerformerVisionTransformer(nn.Module):
#     def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=192,proj_dim = 16, d_head =48, depth=12,
#                  num_heads=4, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
#                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
#                  act_layer=None, weight_init='',update_mode = 'rbf2keys', scale_w = 1., two_proj_matrix = False):
# @register_model
# def deit_tiny_patch16_224_performer(pretrained=False, **kwargs):
#     model = PerformerVisionTransformer(
#         proj_dim = 8, d_head =48, update_mode = 'standard', scale_w = 1., two_proj_matrix = False,
#         patch_size=16, embed_dim=192, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs) 
#     model.default_cfg = _cfg()
#     # if pretrained:
#     #     checkpoint = torch.hub.load_state_dict_from_url(
#     #         url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
#     #         map_location="cpu", check_hash=True
#     #     )
#     #     model.load_state_dict(checkpoint["model"])
#     return model

# @register_model
# def deit_tiny_patch16_224_performer_2keys_2phi_scale(pretrained=False, **kwargs):
#     model = PerformerVisionTransformer(
#         proj_dim = 8, d_head =48, update_mode = 'rbf2keys', scale_w = .5, two_proj_matrix = True,
#         patch_size=16, embed_dim=192, depth=12, num_heads=2, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs) 
#     model.default_cfg = _cfg()
#     # if pretrained:
#     #     checkpoint = torch.hub.load_state_dict_from_url(
#     #         url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
#     #         map_location="cpu", check_hash=True
#     #     )
#     #     model.load_state_dict(checkpoint["model"])
#     return model


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def deit_small_hdp2qk_patch16_224(pretrained=False, **kwargs):
    model = HDPVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),num_global_heads = 3, mode ='qk', **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def deit_small_hdp4qk_patch16_224(pretrained=False, **kwargs):
    model = HDPVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),num_global_heads = 4, mode ='qk', **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def deit_small_hdp_adaptive_qk_patch16_224(pretrained=False, **kwargs):
    model = HDPVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),num_global_heads = None, adaptive_global_heads= True, mode ='qk', **kwargs)
    model.default_cfg = _cfg()
    return model

# @register_model
# def deit_tiny_hdp2qk_patch16_224(pretrained=False, **kwargs):
#     model = HDPVisionTransformer(
#         patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), num_global_heads = 2, mode ='qk', **kwargs) # <name>'s NOTE: in the original code, num_heads = 3 here
#     model.default_cfg = _cfg()
#     return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model



# Patch size 16x16 models
@register_model
def xcit_nano_12_p16(pretrained=False, **kwargs):
    model = XCiT(
        patch_size=16, embed_dim=128, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), eta=1.0, tokens_norm=False, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def xcit_tiny_12_p16(pretrained=False, **kwargs):
    model = XCiT(
        patch_size=16, embed_dim=192, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), eta=1.0, tokens_norm=True, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def xcit_small_12_p16(pretrained=False, **kwargs):
    model = XCiT(
        patch_size=16, embed_dim=384, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), eta=1.0, tokens_norm=True, **kwargs)
    model.default_cfg = _cfg()
    return model

#####################HDP
@register_model
def xcit_small_12_p16_hdp_4qk(pretrained=False, **kwargs):
    model = HDPXCiT(
        patch_size=16, embed_dim=384, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), eta=1.0, tokens_norm=True, hdp = 'qk', hdp_num_heads= 4, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def xcit_tiny_24_p16(pretrained=False, **kwargs):
    model = XCiT(
        patch_size=16, embed_dim=192, depth=24, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), eta=1e-5, tokens_norm=True, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def xcit_small_24_p16(pretrained=False, **kwargs):
    model = XCiT(
        patch_size=16, embed_dim=384, depth=24, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), eta=1e-5, tokens_norm=True, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def xcit_medium_24_p16(pretrained=False, **kwargs):
    model = XCiT(
        patch_size=16, embed_dim=512, depth=24, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), eta=1e-5, tokens_norm=True, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def xcit_large_24_p16(pretrained=False, **kwargs):
    model = XCiT(
        patch_size=16, embed_dim=768, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), eta=1e-5, tokens_norm=True, **kwargs)
    model.default_cfg = _cfg()
    return model


# Patch size 8x8 models
@register_model
def xcit_nano_12_p8(pretrained=False, **kwargs):
    model = XCiT(
        patch_size=8, embed_dim=128, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), eta=1.0, tokens_norm=False, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def xcit_tiny_12_p8(pretrained=False, **kwargs):
    model = XCiT(
        patch_size=8, embed_dim=192, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), eta=1.0, tokens_norm=True, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def xcit_small_12_p8(pretrained=False, **kwargs):
    model = XCiT(
        patch_size=8, embed_dim=384, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), eta=1.0, tokens_norm=True, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def xcit_tiny_24_p8(pretrained=False, **kwargs):
    model = XCiT(
        patch_size=8, embed_dim=192, depth=24, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), eta=1e-5, tokens_norm=True, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def xcit_small_24_p8(pretrained=False, **kwargs):
    model = XCiT(
        patch_size=8, embed_dim=384, depth=24, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), eta=1e-5, tokens_norm=True, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def xcit_medium_24_p8(pretrained=False, **kwargs):
    model = XCiT(
        patch_size=8, embed_dim=512, depth=24, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), eta=1e-5, tokens_norm=True, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def xcit_large_24_p8(pretrained=False, **kwargs):
    model = XCiT(
        patch_size=8, embed_dim=768, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), eta=1e-5, tokens_norm=True, **kwargs)
    model.default_cfg = _cfg()
    return model