# from _typeshed import BytesPath
import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.vision_transformer import _init_vit_weights, _load_weights
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from gpytorch.kernels.kernel import Distance
import pdb

class GMMAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., update_mode = None, beta = 1e-4, cls_dealer = False, share_key = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.share_key = share_key

        if self.share_key:
            self.q_net = nn.Linear(dim, dim, bias = qkv_bias)
            self.k_net = nn.Linear(dim, head_dim, bias = qkv_bias)
            self.v_net = nn.Linear(dim, dim, bias = qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.dist = Distance()
        self.cls_dealer = cls_dealer
        self.update_mode = update_mode
        if update_mode == 'soft_em':
            # self.beta = beta
            self.register_buffer('beta', torch.tensor(beta, requires_grad = False))
            if not self.cls_dealer:
                self.register_buffer('pi', (torch.ones(1, self.num_heads, 197, 197, requires_grad = False)/197))
            else:
                #for training before 21_10_2021_06:18:11_deit_tiny_gmm_softe_cls_patch16_224
                # self.register_buffer('pi', (torch.ones(1, self.num_heads, 197, 197, requires_grad = False)/197))
                self.register_buffer('pi', (torch.ones(1, self.num_heads,1, 196, requires_grad = False)/196))
                # self.pi_cls =  nn.Parameter(torch.ones(1, self.num_heads, 197, 1)/math.pow(197, 0.5), requires_grad = True)
        if update_mode == 'soft_em_cls_learn':
            self.register_buffer('pi', (torch.ones(1, self.num_heads,197, 196, requires_grad = False)/196))
            self.pi_cls_linear = nn.Linear(self.num_heads)
            self.pi_cls =  nn.Parameter(torch.ones(1, self.num_heads, 197, 1)/math.pow(197, 0.5), requires_grad = True)
    
        elif update_mode == 'gd':
            if not self.cls_dealer:
                # this is for e^pi
                # self.register_buffer('pi', (torch.ones(1, self.num_heads,197, 197, requires_grad = False)/196))
                # self.pi = nn.Parameter(torch.ones(1, self.num_heads, 197, 197)/197, requires_grad = True)
                # self.pi = nn.Parameter(torch.ones(1, self.num_heads, 197, 197)/math.pow(197, 0.25), requires_grad = True)
                # self.register_buffer('pi_mask', torch.empty(1, self.num_heads, 197, 197, requires_grad = False))
                self.pi = nn.Parameter(torch.ones(1, self.num_heads, 197, 197)/math.pow(197, 0.25), requires_grad = False)
            else: 
                self.pi = nn.Parameter(torch.ones(1, self.num_heads, 1, 196)/math.pow(196, 0.25), requires_grad = True)
        elif update_mode =='softe_gd':
            self.register_buffer('beta', torch.tensor(beta, requires_grad = False))
            self.register_buffer('pi_s', (torch.ones(1, self.num_heads,1, 197, requires_grad = False)/math.pow(196, 0.5))) 
            self.pi_g = nn.Parameter(torch.ones(1, self.num_heads, 1, 197)/math.pow(197, 0.25), requires_grad = True)
        elif update_mode == 'qk_pi':
            self.pi_q = nn.Parameter(torch.ones(1, self.num_heads, 197, 1)/math.pow(197, 0.25), requires_grad = True)
            self.pi_k = nn.Parameter(torch.ones(1, self.num_heads, 1, 197)/math.pow(197, 0.25), requires_grad = True)




    def forward(self, x):
        B, N, C = x.shape
        if self.share_key:
            q = self.q_net(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2,1,3)
            k = self.k_net(x).reshape(B, N, 1, C // self.num_heads).permute(0, 2,1,3)
            v = self.v_net(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2,1,3)
            # assert 1==2
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (-self.scale/2.0)*self.dist._sq_dist(q, k, postprocess = False)
        # attn = attn - attn.max(dim = -1, keepdim = True)[0]
        if self.update_mode == 'gd':
            if not self.cls_dealer:
                # attn = torch.clamp(self.pi[:,:,:,:N], min=0.0, max=1.0) * torch.exp(attn)
                # attn = torch.exp(self.pi)*torch.exp(attn)
                # attn = torch.softmax(self.pi, dim = -1)*torch.exp(attn)
                # attn = torch.clamp(torch.square(self.pi), max = 1.)*torch.exp(attn)
                # attn = torch.clamp(torch.abs(self.pi), max = 2.)*torch.exp(attn)

                ####### THIS IS FOR PRUNING
                attn = torch.clamp(torch.abs(self.pi), max = 1.)*torch.exp(attn)
                # attn = torch.clamp(torch.abs(self.pi*self.pi_mask), max = 1.)*torch.exp(attn)
                # attn = self.pi_mask*torch.exp(attn)


            else:
                assert 1==2
                #help the cls token bias
                # cls_pi = torch.clamp(self.pi, min = 0., max = 1.).mean(-1, keepdim = True)
                # pi_square = torch.clamp(torch.square(self.pi), max = 1.)
                pi = torch.clamp(torch.abs(self.pi), max  = 1.)
                cls_pi = pi.mean(-1, keepdim = True)
                # pi_abs = torch.clamp(torch.abs(self.pi), max = 1.)
                # cls_pi = pi_abs.mean(-1, keepdim = True) 
                total_pi = torch.cat([cls_pi,pi], dim = -1)

                # clamping before normalizing to get sum = 1, cuz pi_j < 0 is meaningless
                # total_pi = torch.clamp(total_pi, min=0.0, max=1.0)

                # prevent the total mass getting big for qerry cls and 0 for other queries
                # total_pi = total_pi/(total_pi.sum(-1, keepdim = True))
                attn = total_pi * torch.exp(attn)
            attn = attn / ((attn.sum(dim=-1))[:, :, :, None] + 1e-6)

        elif self.update_mode == 'soft_em':
            pi = self.pi.clone().detach()
            pi_old = self.pi.clone().detach()
            if not self.cls_dealer:

                attn = torch.clamp(self.pi[:,:,:N,:N], min=0.0, max=1.0) * torch.exp(attn)
                attn = attn / ((attn.sum(dim=-1))[:, :, :, None] + 1e-6)
                if self.training:
                    pi_old = self.pi.clone().detach()
                    pi_new = torch.sum(attn, dim=0, keepdim=True)/(B)
                    pi_new = (1 - self.beta)*pi_old + self.beta*pi_new
                    pi_new = pi_new.to(x)
                    self.pi.copy_(pi_new.detach())
            else:

                # attn = torch.exp(attn)
                # cls_pi = pi[:,:,:,:].sum(-1, keepdim = True)/196
                # # cls_pi = pi[:,:,:,1:].sum(-1, keepdim = True)/196
                # total_pi = torch.cat([cls_pi, pi], dim = -1)
                # total_pi = total_pi/(total_pi.sum(-1, keepdim = True))
                pi_cls = pi.mean(-1, keepdim = True)
                pi_ = torch.cat([torch.clamp(torch.abs(pi_cls), max = 1.), pi], dim = -1)
                pi_ = pi_/(pi_.sum(-1, keepdim = True))
                attn = pi_ * torch.exp(attn)
                # attn = pi * torch.exp(attn)
                attn = attn / ((attn.sum(dim=-1))[:, :, :, None] + 1e-6)


                if self.training:
                    ######## I CHANGE THE ROLE OF BETA HERE

                    # beta = torch.clamp(self.beta, min = 0.99, max  = 0.999995)
                    # pi_new = (torch.sum(attn, dim=0, keepdim=True)/(B))[:, :, :, 1:]

                    pi_new = (torch.sum(attn, dim=(0,2), keepdim=True)/(B*197))[:, :, :, 1:]
                    # pi_new = torch.cat([pi_new.mean(-1, keepdim = True), pi_new], dim = -1)
                    # pi_new = torch.cat([torch.max(pi_new, dim = -1, keepdim = True)[0], pi_new], dim = -1)
                    # pi_new = pi_new/(pi_new.sum(-1, keepdim = True))
                    pi_new = (1 - self.beta)*pi_old + self.beta*pi_new
                    pi_new = pi_new.to(x)
                    self.pi.copy_(pi_new.detach())

        elif self.update_mode == 'softe_gd':
            pi_s = self.pi_s.clone().detach()
            pi_old = self.pi_s.clone().detach()
            pi = pi_s*torch.softmax(self.pi_g, dim = -1)
            attn = pi * torch.exp(attn)
            attn = attn / ((attn.sum(dim=-1))[:, :, :, None] + 1e-6)
            if self.training:
                
                # pi_new = (torch.sum(attn, dim=0, keepdim=True)/(B))[:, :, :, 1:]
                # pi_new = (torch.sum(attn, dim=(0,2), keepdim=True)/(B*197))[:, :, :, 1:]
                # pi_new = torch.cat([pi_new.mean(-1, keepdim = True), pi_new], dim = -1)
                # pi_new = pi_new/(pi_new.sum(-1, keepdim = True))
                # pi_new = self.beta*pi_old + (1-self.beta)*pi_new
                # pi_new = pi_new.to(x)
                # self.pi_s.copy_(pi_new.detach())
                pi_new = (torch.sum(attn, dim=(0,2), keepdim=True)/(B*197))[:, :, :, 1:]
                pi_new = torch.cat([pi_new.mean(-1, keepdim = True), pi_new], dim = -1)
                pi_new = pi_new/(pi_new.sum(-1, keepdim = True))
                pi_new = (1 - self.beta)*pi_old + self.beta*pi_new
                pi_new = pi_new.to(x)
                self.pi_s.copy_(pi_new.detach())
        elif self.update_mode == 'qk_pi':
            # assert 1==2,'qk_pi'
            # pi = torch.clamp(abs(self.pi_q), max = 1.)* torch.clamp(abs(self.pi_k), max = 1.)
            # print(pi.shape)
            # print(pi)
            # assert 1==2
            attn = torch.exp(attn)*(torch.clamp(abs(self.pi_q), max = 1.)* torch.clamp(abs(self.pi_k), max = 1.))
            attn = attn / ((attn.sum(dim=-1))[:, :, :, None] + 1e-6)
        else:
            attn = torch.exp(attn)
            attn = attn / ((attn.sum(dim=-1))[:, :, :, None] + 1e-6)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class GMMBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, update_mode = None, beta = 0.999, cls_dealer = False, share_key = False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GMMAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                                    attn_drop=attn_drop, proj_drop=drop,
                                    update_mode = update_mode, beta = beta, cls_dealer = cls_dealer, share_key= share_key)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class GMMVisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', update_mode = None, beta = 0.999, cls_dealer = False, share_key = False):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            GMMBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                update_mode = update_mode, beta = beta, cls_dealer= cls_dealer, share_key=share_key)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            assert 1==2
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            assert 1==2, 'whats head dist'
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x