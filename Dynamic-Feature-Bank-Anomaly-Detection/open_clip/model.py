""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
from dataclasses import dataclass
import logging
import math
from typing import Optional, Tuple, Union
from collections import OrderedDict
import re
from sklearn.metrics import pairwise

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch import Tensor
import open_clip.utils.misc as misc
import argparse
from functools import partial
from open_clip.utils.env import checkpoint_pathmgr as pathmgr

from .hf_model import HFTextEncoder
from .modified_resnet import ModifiedResNet
from .timm_model import TimmModel
from .transformer import LayerNormFp32, LayerNorm, QuickGELU, Attention, VisionTransformer, TextTransformer, VisionTransformer_Mul
from .new_utils import to_2tuple
from .adaptive_anomaly import (
    AdaptiveResolutionScanner,
    CalibratedMultiLevelResidual,
    ContentAdaptiveSemanticBank,
    IlluminationSemanticAligner,
    MultiLevelAnomalyRanker,
    TagSemanticAligner,
    build_semantic_prompts,
    nearest_patch_distance,
    normalize_category,
)
from .gpt_semantic import GPTSemanticValidator

from .vp import (
    PadPrompter,
    RandomPatchPrompter,
    FixedPatchPrompter
)

from torch.autograd import Variable, grad

PROMPT_TYPES = {
    "padding": PadPrompter,
    "random_patch": RandomPatchPrompter,
    "fixed_patch": FixedPatchPrompter
}


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224

    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    input_patchnorm: bool = False  # whether to use dual patchnorm - would only apply the input layernorm on each patch, as post-layernorm already exist in original clip vit design
    global_average_pool: bool = False  # whether to global average pool the last embedding layer, instead of using CLS token (https://arxiv.org/abs/2205.01580)
    attentional_pool: bool = False  # whether to use attentional pooler in the last embedding layer
    n_queries: int = 256  # n_queries for attentional pooler
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    output_tokens: bool = True

    timm_model_name: str = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj_bias: bool = False  # enable bias final projection
    timm_drop: float = 0.  # head dropout
    timm_drop_path: Optional[float] = None  # backbone stochastic depth


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    ls_init_value: Optional[float] = None  # layer scale initial value
    hf_model_name: str = None
    hf_tokenizer_name: str = None
    hf_model_pretrained: bool = True
    proj: str = 'mlp'
    pooler_type: str = 'mean_pooler'
    embed_cls: bool = False
    pad_id: int = 0
    output_tokens: bool = False

def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype


def get_input_dtype(precision: str):
    input_dtype = None
    if precision in ('bf16', 'pure_bf16'):
        input_dtype = torch.bfloat16
    elif precision in ('fp16', 'pure_fp16'):
        input_dtype = torch.float16
    return input_dtype

# state_level = {
#                "normal":["{}", "flawless {}", "perfect {}", "unblemished {}",
#                          "{} without flaw", "{} without defect", "{} without damage"],
#                 "anomaly":["damaged {}", "{} with flaw", "{} with defect", "{} with damage"]
# }

state_level = {
    "normal": [
        "{} with no visible defect",
        "flawless {} grid-cloth surface",
        "perfect {} fabric with uniform grid lines",
        "clean and unblemished {}",
        "{} without flaw",
        "{} without defect",
        "{} without any damage",
        "{} with straight and well-aligned warp and weft threads",
        "a pristine {} grid fabric with consistent color and intact yarns"
    ],
    "anomaly": [
        "a {} grid-cloth surface with misaligned or distorted grid lines",
        "a {} fabric with missing yarns or broken warp threads",
        "a {} fabric with long-float or floating yarn defects",
        "a {} grid cloth with thick yarn or slub irregularities",
        "a {} fabric with holes or missing material",
        "a {} fabric contaminated by stains or dirt",
        "a {} grid cloth with color inconsistency or discoloration",
        "a {} fabric with abnormal yarn tension or grid deformation",
        "a {} surface showing clear structural or texture defects",
        "{} with visible defect",
        "{} with noticeable damage",
        "{} with obvious abnormal grid patterns"
    ]
} # WFDD grid-cloth

template_level = [
                  "a cropped photo of the {}.",
                  "a cropped photo of a {}.",
                  "a close-up photo of a {}.",
                  "a close-up photo of the {}.",
                  "a bright photo of a {}.",
                  "a bright photo of the {}.",
                  "a dark photo of a {}.",
                  "a dark photo of the {}.",
                  "a jpeg corrupted photo of a {}.",
                  "a jpeg corrupted photo of the {}.",
                  "a blurry photo of the {}.",
                  "a blurry photo of a {}.",
                  "a photo of the {}.",
                  "a photo of a {}.",
                  "a photo of a small {}.",
                  "a photo of the small {}.",
                  "a photo of a large {}.",
                  "a photo of the large {}.",
                  "a photo of a {} for visual inspection.",
                  "a photo of the {} for visual inspection.",
                  "a photo of a {} for anomaly detection.",
                  "a photo of the {} for anomaly detection."
]

def get_texts(obj_name):

    l = ["airplane", "automobile", "bird",
         "cat", "deer", "dog", "frog", "horse", "ship", "truck", "animal"]

    if obj_name in l:
        normal_texts = []
        anomaly_texts = []
        normal = "a photo of " + obj_name + " for anomaly detection."
        normal_texts.append(normal)
        anomaly = "a photo without " + obj_name + " for anomaly detection."
        anomaly_texts.append(anomaly)
    else:
        normal_states = [s.format(obj_name) for s in state_level["normal"]]
        anomaly_states = [s.format(obj_name) for s in state_level["anomaly"]]

        normal_texts = [t.format(state) for state in normal_states for t in template_level]
        anomaly_texts = [t.format(state) for state in anomaly_states for t in template_level]

    return normal_texts, anomaly_texts


def _build_vision_tower(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.timm_model_name:
        visual = TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            drop=vision_cfg.timm_drop,
            drop_path=vision_cfg.timm_drop_path,
            patch_drop=vision_cfg.patch_dropout if vision_cfg.patch_dropout > 0 else None,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size,
        )
    elif isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
        )
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            input_patchnorm=vision_cfg.input_patchnorm,
            global_average_pool=vision_cfg.global_average_pool,
            attentional_pool=vision_cfg.attentional_pool,
            n_queries=vision_cfg.n_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
    return visual

def _build_vision_tower_Mul(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.timm_model_name:
        visual = TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            drop=vision_cfg.timm_drop,
            drop_path=vision_cfg.timm_drop_path,
            patch_drop=vision_cfg.patch_dropout if vision_cfg.patch_dropout > 0 else None,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size,
        )
    elif isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
        )
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        visual = VisionTransformer_Mul(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            input_patchnorm=vision_cfg.input_patchnorm,
            global_average_pool=vision_cfg.global_average_pool,
            attentional_pool=vision_cfg.attentional_pool,
            n_queries=vision_cfg.n_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
    return visual

def _build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    if text_cfg.hf_model_name:
        text = HFTextEncoder(
            text_cfg.hf_model_name,
            output_dim=embed_dim,
            proj=text_cfg.proj,
            pooler_type=text_cfg.pooler_type,
            pretrained=text_cfg.hf_model_pretrained,
            output_tokens=text_cfg.output_tokens,
        )
    else:
        act_layer = QuickGELU if quick_gelu else nn.GELU
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm

        text = TextTransformer(
            context_length=text_cfg.context_length,
            vocab_size=text_cfg.vocab_size,
            width=text_cfg.width,
            heads=text_cfg.heads,
            layers=text_cfg.layers,
            ls_init_value=text_cfg.ls_init_value,
            output_dim=embed_dim,
            embed_cls=text_cfg.embed_cls,
            output_tokens=text_cfg.output_tokens,
            pad_id=text_cfg.pad_id,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
    return text

class BatchNormPoint(nn.Module):
    def __init__(self, feat_size):
        super().__init__()
        self.feat_size = feat_size
        self.bn = nn.BatchNorm1d(feat_size)

    def forward(self, x):
        assert len(x.shape) == 3
        s1, s2, s3 = x.shape[0], x.shape[1], x.shape[2]
        assert s3 == self.feat_size
        x = x.view(s1 * s2, self.feat_size)
        x = self.bn(x)
        return x.view(s1, s2, s3)

class CLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)

        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None
        if self.output_dict:
            return {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
        return image_features, text_features, self.logit_scale.exp()

class TransformerBasicHead(nn.Module):
    """
    Basic Transformer Head. No pool.
    """

    def __init__(
        self,
        dim_in,
        num_classes
    ):
        super(TransformerBasicHead, self).__init__()
        self.projection1 = nn.Linear(dim_in, 128, bias=True)
        self.projection2 = nn.Linear(128, 64, bias=True)
        self.projection3 = nn.Linear(64, num_classes, bias=True)
        self.bn1 = nn.BatchNorm1d(dim_in)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.projection1(x)
        x = F.relu(x, inplace=True)
        x = self.bn2(x)
        x = self.projection2(x)
        x = F.relu(x, inplace=True)
        x = self.bn3(x)
        x = self.projection3(x)
        return torch.sigmoid(x)

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class InCTRL(nn.Module):
    def __init__(
            self,
            args,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower_Mul(embed_dim, vision_cfg, quick_gelu, cast_dtype)

        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        self.adapter = Adapter(640, 4)
        self.diff_head = TransformerBasicHead(225, 1)
        self.diff_head_ref = TransformerBasicHead(640, 1)

        for p in self.visual.parameters():
            p.requires_grad = False

        for p in text.parameters():
            p.requires_grad = False

    def encode_image(self, image, out_layers: list = [7, 9, 11], normalize: bool = False):
        features = self.visual.forward(image, out_layers)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x

    def forward(self, tokenizer, image: Optional[torch.Tensor] = None, text: Optional[torch.Tensor] = None, normal_list = None):
        if normal_list == None:
            img = image[0].cuda(non_blocking=True)
            normal_image = image[1:]
            normal_image = torch.stack(normal_image)
            shot, b, _, _, _ = normal_image.shape
            normal_image = normal_image.reshape(-1, 3, 240, 240).cuda(non_blocking=True)
        else:
            img = image[0].cuda(non_blocking=True)
            normal_image = normal_list
            normal_image = torch.stack(normal_image)
            normal_image = normal_image.unsqueeze(1)
            b = len(img)
            normal_image = normal_image.repeat(1, b, 1, 1, 1)
            shot, _, _, _, _ = normal_image.shape
            normal_image = normal_image.reshape(-1, 3, 240, 240).cuda(non_blocking=True)

        token, Fp_list, Fp = self.encode_image(img, normalize=False)
        token_n, Fp_list_n, Fp_n = self.encode_image(normal_image, normalize=False)

        Fp_list = torch.stack(Fp_list)
        Fp_list_n = torch.stack(Fp_list_n)

        Fp_list = Fp_list[:, :, 1:, :]
        Fp_list_n = Fp_list_n[:, :, 1:, :]

        Fp_list = Fp_list.reshape(b, 3, 225, -1)
        Fp_list_n = Fp_list_n.reshape(b, 3, 225 * shot, -1)

        token_n = token_n.reshape(b, shot, -1)

        token_ad = self.adapter.forward(token)
        token_n = self.adapter.forward(token_n)
        token_n = torch.mean(token_n, dim=1)
        token_ref = token_n - token_ad

        text_score = []
        max_diff_score = []
        patch_ref_map = []
        for i in range(len(token)):
            Fp = Fp_list[i, :, :, :]
            Fp_n = Fp_list_n[i, :, :, :]

            Fp_map = list()
            for j in range(len(Fp)):
                tmp_x = Fp[j, :, :]
                tmp_n = Fp_n[j, :, :]
                am_fp = list()
                for k in range(len(tmp_x)):
                    tmp = tmp_x[k]
                    tmp = tmp.unsqueeze(0)
                    tmp_n = tmp_n / tmp_n.norm(dim=-1, keepdim=True)
                    tmp = tmp / tmp.norm(dim=-1, keepdim=True)
                    s = (0.5 * (1 - (tmp @ tmp_n.T))).min(dim=1).values
                    am_fp.append(s)
                am_fp = torch.stack(am_fp)
                Fp_map.append(am_fp)
            Fp_map = torch.stack(Fp_map)
            Fp_map = torch.mean(Fp_map.squeeze(2), dim=0)
            patch_ref_map.append(Fp_map)
            score = Fp_map.max(dim=0).values
            max_diff_score.append(score)

            # zero shot
            image_feature = token[i]
            image_feature = image_feature.unsqueeze(0)
            image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)

            obj_type = text[i]
            normal_texts, anomaly_texts = get_texts(obj_type.replace('_', " "))
            pos_features = tokenizer(normal_texts).cuda()
            neg_features = tokenizer(anomaly_texts).cuda()
            pos_features = self.encode_text(pos_features)
            neg_features = self.encode_text(neg_features)
            pos_features = pos_features / pos_features.norm(dim=-1, keepdim=True)
            neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)
            pos_features = torch.mean(pos_features, dim=0, keepdim=True)
            neg_features = torch.mean(neg_features, dim=0, keepdim=True)
            pos_features = pos_features / pos_features.norm(dim=-1, keepdim=True)
            neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)
            text_features = torch.cat([pos_features, neg_features], dim=0)
            score = (100 * image_feature @ text_features.T).softmax(dim=-1)
            tmp = score[0, 1]
            text_score.append(tmp)

        text_score = torch.stack(text_score).unsqueeze(1)
        img_ref_score = self.diff_head_ref.forward(token_ref)
        patch_ref_map = torch.stack(patch_ref_map)
        holistic_map = text_score + img_ref_score + patch_ref_map
        hl_score = self.diff_head.forward(holistic_map)

        hl_score = hl_score.squeeze(1)
        fg_score = torch.stack(max_diff_score)
        final_score = (hl_score + fg_score) / 2

        img_ref_score = img_ref_score.squeeze(1)

        return final_score, img_ref_score

class InCTRL_visualization(nn.Module):
    def __init__(
            self,
            args,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower_Mul(embed_dim, vision_cfg, quick_gelu, cast_dtype)

        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        self.adapter = Adapter(640, 4)
        self.diff_head = TransformerBasicHead(225, 1)
        self.diff_head_ref = TransformerBasicHead(640, 1)

        for p in self.visual.parameters():
            p.requires_grad = False

        for p in text.parameters():
            p.requires_grad = False

    def encode_image(self, image, out_layers: list = [7, 9, 11], normalize: bool = False):
        features = self.visual.forward(image, out_layers)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x

    def forward(self, tokenizer, image: Optional[torch.Tensor] = None, text: Optional[torch.Tensor] = None, normal_list = None):
        if normal_list == None:
            img = image[0].cuda(non_blocking=True)
            normal_image = image[1:]
            normal_image = torch.stack(normal_image)
            shot, b, _, _, _ = normal_image.shape
            normal_image = normal_image.reshape(-1, 3, 240, 240).cuda(non_blocking=True)
        else:
            img = image[0].cuda(non_blocking=True)
            normal_image = normal_list
            normal_image = torch.stack(normal_image)
            normal_image = normal_image.unsqueeze(1)
            b = len(img)
            normal_image = normal_image.repeat(1, b, 1, 1, 1)
            shot, _, _, _, _ = normal_image.shape
            normal_image = normal_image.reshape(-1, 3, 240, 240).cuda(non_blocking=True)

        token, Fp_list, Fp = self.encode_image(img, normalize=False)
        token_n, Fp_list_n, Fp_n = self.encode_image(normal_image, normalize=False)

        Fp_list = torch.stack(Fp_list)
        Fp_list_n = torch.stack(Fp_list_n)

        Fp_list = Fp_list[:, :, 1:, :]
        Fp_list_n = Fp_list_n[:, :, 1:, :]

        Fp_list = Fp_list.reshape(b, 3, 225, -1)
        Fp_list_n = Fp_list_n.reshape(b, 3, 225 * shot, -1)

        token_n = token_n.reshape(b, shot, -1)

        token_ad = self.adapter.forward(token)
        token_n = self.adapter.forward(token_n)
        token_n = torch.mean(token_n, dim=1)
        token_ref = token_n - token_ad

        text_score = []
        max_diff_score = []
        patch_ref_map = []
        for i in range(len(token)):
            Fp = Fp_list[i, :, :, :]
            Fp_n = Fp_list_n[i, :, :, :]

            Fp_map = list()
            for j in range(len(Fp)):
                tmp_x = Fp[j, :, :]
                tmp_n = Fp_n[j, :, :]
                am_fp = list()
                for k in range(len(tmp_x)):
                    tmp = tmp_x[k]
                    tmp = tmp.unsqueeze(0)
                    tmp_n = tmp_n / tmp_n.norm(dim=-1, keepdim=True)
                    tmp = tmp / tmp.norm(dim=-1, keepdim=True)
                    s = (0.5 * (1 - (tmp @ tmp_n.T))).min(dim=1).values
                    am_fp.append(s)
                am_fp = torch.stack(am_fp)
                Fp_map.append(am_fp)
            Fp_map = torch.stack(Fp_map)
            Fp_map = torch.mean(Fp_map.squeeze(2), dim=0)
            patch_ref_map.append(Fp_map)
            score = Fp_map.max(dim=0).values
            max_diff_score.append(score)

            # zero shot
            image_feature = token[i]
            image_feature = image_feature.unsqueeze(0)
            image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)

            obj_type = text[i]
            normal_texts, anomaly_texts = get_texts(obj_type.replace('_', " "))
            pos_features = tokenizer(normal_texts).cuda()
            neg_features = tokenizer(anomaly_texts).cuda()
            pos_features = self.encode_text(pos_features)
            neg_features = self.encode_text(neg_features)
            pos_features = pos_features / pos_features.norm(dim=-1, keepdim=True)
            neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)
            pos_features = torch.mean(pos_features, dim=0, keepdim=True)
            neg_features = torch.mean(neg_features, dim=0, keepdim=True)
            pos_features = pos_features / pos_features.norm(dim=-1, keepdim=True)
            neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)
            text_features = torch.cat([pos_features, neg_features], dim=0)
            score = (100 * image_feature @ text_features.T).softmax(dim=-1)
            tmp = score[0, 1]
            text_score.append(tmp)

        text_score = torch.stack(text_score).unsqueeze(1)
        img_ref_score = self.diff_head_ref.forward(token_ref)
        patch_ref_map = torch.stack(patch_ref_map)
        holistic_map = text_score + img_ref_score + patch_ref_map
        hl_score = self.diff_head.forward(holistic_map)

        hl_score = hl_score.squeeze(1)
        fg_score = torch.stack(max_diff_score)
        final_score = (hl_score + fg_score) / 2

        img_ref_score = img_ref_score.squeeze(1)

        return final_score, img_ref_score, patch_ref_map


class AdaptiveInCTRL(InCTRL):
    """InCTRL with dynamic resolution scanning and multi-level ranking.

    This class preserves all trainable parameter names from ``InCTRL`` so that
    existing checkpoints remain loadable. The added modules are parameter-free
    and provide deterministic Stage2-4 feature conversion and ranking.
    """

    def __init__(
            self,
            args,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__(
            args,
            embed_dim,
            vision_cfg,
            text_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
            output_dict=output_dict,
        )
        adaptive_cfg = getattr(args, "ADAPTIVE", None)
        scales = getattr(adaptive_cfg, "SCALES", [0.35, 0.50, 0.70, 1.0])
        view_temperature = getattr(adaptive_cfg, "VIEW_TEMPERATURE", 0.08)
        self.resolution_scanner = AdaptiveResolutionScanner(
            scales=scales,
            temperature=view_temperature,
        )
        self.patch_residual = CalibratedMultiLevelResidual(
            calibration=getattr(adaptive_cfg, "PATCH_CALIBRATION", 0.25),
        )
        self.semantic_bank = ContentAdaptiveSemanticBank(
            selection_temperature=getattr(adaptive_cfg, "PROMPT_TEMPERATURE", 0.10),
            score_temperature=getattr(adaptive_cfg, "SEMANTIC_TEMPERATURE", 0.07),
        )
        self.d_tag = (
            TagSemanticAligner(getattr(adaptive_cfg, "SEMANTIC_TEMPERATURE", 0.07))
            if getattr(adaptive_cfg, "D_TAG_ENABLE", True)
            else None
        )
        self.d_illum = (
            IlluminationSemanticAligner(getattr(adaptive_cfg, "SEMANTIC_TEMPERATURE", 0.07))
            if getattr(adaptive_cfg, "D_ILLUM_ENABLE", True)
            else None
        )
        self.d_tag_weight = float(getattr(adaptive_cfg, "D_TAG_WEIGHT", 0.15))
        self.d_illum_weight = float(getattr(adaptive_cfg, "D_ILLUM_WEIGHT", 0.10))
        self.gpt_validator = GPTSemanticValidator(
            enabled=getattr(adaptive_cfg, "GPT_VALIDATION_ENABLE", False),
            endpoint=getattr(adaptive_cfg, "GPT_VALIDATION_ENDPOINT", "https://api.openai.com/v1/chat/completions"),
            model=getattr(adaptive_cfg, "GPT_VALIDATION_MODEL", "gpt-5.5"),
            timeout=getattr(adaptive_cfg, "GPT_VALIDATION_TIMEOUT", 20.0),
        )
        self.anomaly_ranker = MultiLevelAnomalyRanker(
            score_weights=getattr(adaptive_cfg, "SCORE_WEIGHTS", [0.60, 0.25, 0.15]),
            difficulty_weights=getattr(adaptive_cfg, "DIFFICULTY_WEIGHTS", [0.40, 0.25, 0.35]),
        )
        self._semantic_cache = {}
        self._normal_cache_key = None
        self._normal_cache = None

    def clear_runtime_cache(self):
        self._semantic_cache.clear()
        self._normal_cache_key = None
        self._normal_cache = None

    def validate_semantic_text(self, text: str, category: str = "texture"):
        """Explicit opt-in hook for offline/online GPT semantic validation.

        The forward path never calls this method.  It is provided for a future
        preprocessing script or an experiment that deliberately enables the
        configured validator.
        """
        return self.gpt_validator.validate(text, category)

    @staticmethod
    def _category_list(categories, batch_size):
        if isinstance(categories, str):
            return [categories] * batch_size
        categories = list(categories)
        if len(categories) == 1 and batch_size > 1:
            categories = categories * batch_size
        if len(categories) != batch_size:
            raise ValueError("The number of categories must match the image batch size.")
        return categories

    def _prepare_images(self, image, normal_list):
        device = next(self.parameters()).device
        query = image[0].to(device=device, non_blocking=True)
        batch_size = query.shape[0]
        cache_key = None
        if normal_list is None:
            normals = torch.stack(
                [sample.to(device=device, non_blocking=True) for sample in image[1:]], dim=1
            )
        else:
            if isinstance(normal_list, Tensor):
                normal_tensor = normal_list
            else:
                normal_tensor = torch.stack(list(normal_list), dim=0)
            if normal_tensor.ndim == 5 and normal_tensor.shape[1] == 1:
                normal_tensor = normal_tensor.squeeze(1)
            if normal_tensor.ndim != 4:
                raise ValueError("normal_list must contain [C,H,W] tensors.")
            cache_key = (id(normal_list), str(device), str(normal_tensor.dtype))
            normals = normal_tensor.to(device=device, non_blocking=True).unsqueeze(0)
            normals = normals.expand(batch_size, -1, -1, -1, -1)
        return query, normals, cache_key

    def _encode_references(self, normals, cache_key):
        batch_size, shot, channels, height, width = normals.shape
        if (
            not self.training
            and cache_key is not None
            and cache_key == self._normal_cache_key
            and self._normal_cache is not None
        ):
            token_base, level_base = self._normal_cache
            tokens = token_base.unsqueeze(0).expand(batch_size, -1, -1)
            levels = [
                level.unsqueeze(0).expand(batch_size, -1, -1, -1)
                for level in level_base
            ]
            return tokens, levels

        flat = normals.reshape(batch_size * shot, channels, height, width)
        tokens, patch_levels, _ = self.encode_image(flat, normalize=False)
        tokens = tokens.reshape(batch_size, shot, -1)
        levels = [
            level[:, 1:, :].reshape(batch_size, shot, level.shape[1] - 1, level.shape[2])
            for level in patch_levels
        ]
        if not self.training and cache_key is not None and batch_size == 1:
            self._normal_cache_key = cache_key
            self._normal_cache = (
                tokens[0].detach(),
                [level[0].detach() for level in levels],
            )
        return tokens, levels

    def _semantic_features(self, tokenizer, categories, device):
        normal_banks = []
        anomaly_banks = []
        prompt_text = []
        for category in categories:
            key = (normalize_category(category), str(device))
            if key not in self._semantic_cache:
                normal_prompts, anomaly_prompts = build_semantic_prompts(category)
                tokens = tokenizer(normal_prompts + anomaly_prompts).to(device)
                with torch.no_grad():
                    features = self.encode_text(tokens, normalize=True).detach()
                split = len(normal_prompts)
                self._semantic_cache[key] = (
                    features[:split],
                    features[split:],
                    normal_prompts,
                    anomaly_prompts,
                )
            normal_bank, anomaly_bank, normal_prompts, anomaly_prompts = self._semantic_cache[key]
            normal_banks.append(normal_bank)
            anomaly_banks.append(anomaly_bank)
            prompt_text.append((normal_prompts, anomaly_prompts))
        return torch.stack(normal_banks), torch.stack(anomaly_banks), prompt_text

    def forward(
            self,
            tokenizer,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
            normal_list=None,
            labels: Optional[Tensor] = None,
            return_details: bool = False,
    ):
        query, normals, cache_key = self._prepare_images(image, normal_list)
        batch_size = query.shape[0]
        categories = self._category_list(text, batch_size)

        query_token, query_levels_all, _ = self.encode_image(query, normalize=False)
        query_levels = [level[:, 1:, :] for level in query_levels_all]
        normal_tokens, normal_levels_grouped = self._encode_references(normals, cache_key)
        normal_levels = [
            level.reshape(batch_size, -1, level.shape[-1])
            for level in normal_levels_grouped
        ]

        patch_map, residual_details = self.patch_residual(query_levels, normal_levels)
        patch_side = int(math.sqrt(patch_map.shape[1]))
        if patch_side * patch_side != patch_map.shape[1]:
            raise ValueError("The visual patch count must form a square grid.")
        patch_grid = patch_map.reshape(batch_size, patch_side, patch_side)

        views, scan_centre = self.resolution_scanner.make_views(query, patch_grid)
        normal_token_prototype = F.normalize(normal_tokens.mean(dim=1), dim=-1)
        view_tokens = []
        view_evidence = []
        for scale, view in zip(self.resolution_scanner.scales, views):
            if math.isclose(scale, 1.0):
                view_token = query_token
                view_levels = query_levels
            else:
                view_token, view_levels_all, _ = self.encode_image(view, normalize=False)
                view_levels = [level[:, 1:, :] for level in view_levels_all]
            global_novelty = 0.5 * (
                1.0 - F.cosine_similarity(F.normalize(view_token, dim=-1), normal_token_prototype)
            )
            local_distance = nearest_patch_distance(view_levels[-1], normal_levels[-1])
            topk = max(1, int(local_distance.shape[1] * 0.05))
            local_novelty = local_distance.topk(topk, dim=1).values.mean(dim=1)
            view_tokens.append(view_token)
            view_evidence.append(0.45 * global_novelty + 0.55 * local_novelty)

        view_tokens = torch.stack(view_tokens, dim=1)
        view_evidence = torch.stack(view_evidence, dim=1)
        dynamic_token, view_weights = self.resolution_scanner.fuse(view_tokens, view_evidence)

        normal_adapted = self.adapter(normal_tokens).mean(dim=1)
        query_adapted = self.adapter(query_token)
        dynamic_adapted = self.adapter(dynamic_token)
        base_image_score = self.diff_head_ref(normal_adapted - query_adapted).squeeze(1)
        scan_image_score = self.diff_head_ref(normal_adapted - dynamic_adapted).squeeze(1)
        image_score = (0.70 * base_image_score + 0.30 * scan_image_score).clamp(0.0, 1.0)
        visual_distance = 0.5 * (
            1.0 - F.cosine_similarity(F.normalize(query_token, dim=-1), normal_token_prototype)
        )

        normal_bank, anomaly_bank, prompt_text = self._semantic_features(
            tokenizer, categories, query.device
        )
        semantic_score, semantic_details = self.semantic_bank(
            dynamic_token, normal_bank, anomaly_bank
        )
        # D_tag and D_illum are lightweight auxiliary alignment branches.  The
        # base CLIP semantic score remains dominant, while both branches are
        # exposed in details for ablation and paper-to-code traceability.
        tag_score = dynamic_token.new_zeros(batch_size)
        illum_score = dynamic_token.new_zeros(batch_size)
        if self.d_tag is not None:
            tag_score, tag_logits = self.d_tag(dynamic_token, normal_bank, anomaly_bank)
            semantic_details["d_tag_logits"] = tag_logits
        if self.d_illum is not None:
            illum_score, illum_logits = self.d_illum(dynamic_token, normal_bank, anomaly_bank)
            semantic_details["d_illum_logits"] = illum_logits
        total_aux_weight = self.d_tag_weight + self.d_illum_weight
        if total_aux_weight > 1.0:
            raise ValueError("D_tag/D_illum weights must sum to at most one.")
        semantic_score = (
            (1.0 - total_aux_weight) * semantic_score
            + self.d_tag_weight * tag_score
            + self.d_illum_weight * illum_score
        ).clamp(0.0, 1.0)
        semantic_details["d_tag_score"] = tag_score
        semantic_details["d_illum_score"] = illum_score
        semantic_details["gpt_validation_enabled"] = self.gpt_validator.enabled
        final_score, rank_details = self.anomaly_ranker(
            patch_map,
            image_score,
            semantic_score,
            query_levels[-1],
            visual_distance,
        )

        details = {
            "patch_map": patch_map,
            "patch_grid": patch_grid,
            "view_weights": view_weights,
            "view_evidence": view_evidence,
            "scan_centre": scan_centre,
            "scan_scales": self.resolution_scanner.scales,
            "visual_distance": visual_distance,
            "prompt_text": prompt_text,
            **residual_details,
            **semantic_details,
            **rank_details,
        }
        if labels is not None:
            labels = labels.to(final_score.device).float().flatten()
            predicted_target = torch.where(labels > 0.5, semantic_score, 1.0 - semantic_score)
            semantic_error = 1.0 - predicted_target
            normal_separation = 1.0 - torch.exp(-visual_distance / 0.25)
            anomaly_separation = torch.exp(-visual_distance / 0.25)
            separation_error = torch.where(labels > 0.5, anomaly_separation, normal_separation)
            details["training_difficulty"] = (
                0.40 * rank_details["texture_complexity"]
                + 0.25 * semantic_error
                + 0.35 * separation_error
            ).clamp(0.0, 1.0)

        if return_details:
            return final_score, details
        return final_score, image_score

class CustomTextCLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.context_length = self.text.context_length
        self.vocab_size = self.text.vocab_size
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None
        if self.output_dict:
            return {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
        return image_features, text_features, self.logit_scale.exp()


def convert_weights_to_lp(model: nn.Module, dtype=torch.float16):
    """Convert applicable model parameters to low-precision (bf16 or fp16)"""

    def _convert_weights(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(dtype)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(dtype)

        if isinstance(l, (nn.MultiheadAttention, Attention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.to(dtype)

        if isinstance(l, (CLIP, TextTransformer)):
            # convert text nn.Parameter projections
            attr = getattr(l, "text_projection", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

        if isinstance(l, VisionTransformer):
            # convert vision nn.Parameter projections
            attr = getattr(l, "proj", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

    model.apply(_convert_weights)


convert_weights_to_fp16 = convert_weights_to_lp  # backwards compat


# used to maintain checkpoint compatibility
def convert_to_custom_text_state_dict(state_dict: dict):
    if 'text_projection' in state_dict:
        # old format state_dict, move text tower -> .text
        new_state_dict = {}
        for k, v in state_dict.items():
            if any(k.startswith(p) for p in (
                'text_projection',
                'positional_embedding',
                'token_embedding',
                'transformer',
                'ln_final',
            )):
                k = 'text.' + k
            new_state_dict[k] = v
        return new_state_dict
    return state_dict


def build_model_from_openai_state_dict(
        state_dict: dict,
        quick_gelu=True,
        cast_dtype=torch.float16,
):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_size = vision_patch_size * grid_size
    else:
        counts: list = [
            len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_size = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers,
    )
    model = CLIP(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=quick_gelu,  # OpenAI models were trained with QuickGELU
        cast_dtype=cast_dtype,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)

    convert_weights_to_fp16(model)  # OpenAI state dicts are partially converted to float16
    model.load_state_dict(state_dict)
    return model.eval()


def trace_model(model, batch_size=256, device=torch.device('cpu')):
    model.eval()
    image_size = model.visual.image_size
    example_images = torch.ones((batch_size, 3, image_size, image_size), device=device)
    example_text = torch.zeros((batch_size, model.context_length), dtype=torch.int, device=device)
    model = torch.jit.trace_module(
        model,
        inputs=dict(
            forward=(example_images, example_text),
            encode_text=(example_text,),
            encode_image=(example_images,)
        ))
    model.visual.image_size = image_size
    return model


def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic', antialias: bool = True):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get('visual.positional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['visual.positional_embedding'] = new_pos_embed
