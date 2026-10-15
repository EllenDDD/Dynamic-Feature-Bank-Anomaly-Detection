"""Windows-native modules for adaptive multi-resolution anomaly detection.

The implementation keeps the useful ideas behind DynRefer (nested views and
content-dependent view selection) but only relies on PyTorch operators already
available in the InCTRL environment.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


NORMAL_PROMPTS = (
    "a close-up photo of a normal {category} surface with regular texture",
    "an intact {category} surface with uniform structure",
    "a flawless {category} sample with consistent color and pattern",
    "a clean {category} surface without visible defects",
    "a normal {category} texture with well aligned local details",
    "an undamaged {category} sample for visual inspection",
)

ANOMALY_PROMPTS = (
    "a close-up photo of an abnormal {category} surface with a local defect",
    "a damaged {category} surface with broken or distorted texture",
    "a defective {category} sample with a stain hole or missing material",
    "an abnormal {category} texture with misaligned local structure",
    "a {category} surface with inconsistent color or pattern",
    "a visibly flawed {category} sample for anomaly detection",
)


def normalize_category(category: str) -> str:
    """Convert dataset identifiers to readable prompt text."""
    category = re.sub(r"[_-]+", " ", str(category)).strip().lower()
    return re.sub(r"\s+", " ", category) or "texture"


def build_semantic_prompts(category: str) -> Tuple[List[str], List[str]]:
    category = normalize_category(category)
    return (
        [prompt.format(category=category) for prompt in NORMAL_PROMPTS],
        [prompt.format(category=category) for prompt in ANOMALY_PROMPTS],
    )


def nearest_patch_distance(query: Tensor, reference: Tensor) -> Tensor:
    """Return nearest-neighbour cosine distance for every query patch.

    Args:
        query: ``[batch, patches, channels]``.
        reference: ``[batch, reference_patches, channels]``.
    """
    query = F.normalize(query, dim=-1)
    reference = F.normalize(reference, dim=-1)
    similarity = torch.bmm(query, reference.transpose(1, 2)).amax(dim=-1)
    return 0.5 * (1.0 - similarity)


class AdaptiveResolutionScanner(nn.Module):
    """Build and select nested views around an anomaly-directed soft centre."""

    def __init__(
        self,
        scales: Sequence[float] = (0.35, 0.50, 0.70, 1.0),
        temperature: float = 0.08,
        centre_topk_ratio: float = 0.05,
    ) -> None:
        super().__init__()
        if not scales or any(scale <= 0.0 or scale > 1.0 for scale in scales):
            raise ValueError("All scan scales must be in (0, 1].")
        self.scales = tuple(float(scale) for scale in scales)
        self.temperature = float(temperature)
        self.centre_topk_ratio = float(centre_topk_ratio)

    def locate(self, anomaly_map: Tensor) -> Tensor:
        """Estimate a stable centre from the strongest residual responses."""
        if anomaly_map.ndim != 3:
            raise ValueError("anomaly_map must have shape [batch, height, width].")
        batch, height, width = anomaly_map.shape
        smoothed = F.avg_pool2d(anomaly_map.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
        flat = smoothed.flatten(1)
        count = max(1, int(flat.shape[1] * self.centre_topk_ratio))
        values, indices = flat.topk(count, dim=1)
        weights = F.softmax(values / max(self.temperature, 1e-6), dim=1)
        y = torch.div(indices, width, rounding_mode="floor").to(anomaly_map.dtype)
        x = (indices % width).to(anomaly_map.dtype)
        x = (weights * x).sum(dim=1) / max(width - 1, 1) * 2.0 - 1.0
        y = (weights * y).sum(dim=1) / max(height - 1, 1) * 2.0 - 1.0
        return torch.stack((x, y), dim=1)

    @staticmethod
    def _crop(image: Tensor, centre: Tensor, scale: float) -> Tensor:
        batch = image.shape[0]
        if math.isclose(scale, 1.0):
            return image
        max_offset = 1.0 - scale
        offset = centre.clamp(min=-max_offset, max=max_offset)
        theta = image.new_zeros((batch, 2, 3))
        theta[:, 0, 0] = scale
        theta[:, 1, 1] = scale
        theta[:, :, 2] = offset
        grid = F.affine_grid(theta, image.shape, align_corners=False)
        return F.grid_sample(
            image,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

    def make_views(self, image: Tensor, anomaly_map: Tensor) -> Tuple[List[Tensor], Tensor]:
        centre = self.locate(anomaly_map).detach()
        views = [self._crop(image, centre, scale) for scale in self.scales]
        return views, centre

    def fuse(self, view_tokens: Tensor, evidence: Tensor) -> Tuple[Tensor, Tensor]:
        """Fuse ``[B, V, C]`` view tokens using anomaly-aware evidence."""
        if view_tokens.ndim != 3 or evidence.shape != view_tokens.shape[:2]:
            raise ValueError("view_tokens/evidence have incompatible shapes.")
        scale_prior = evidence.new_tensor(
            [1.0 - abs(scale - 0.65) for scale in self.scales]
        ).unsqueeze(0)
        logits = (evidence + 0.03 * scale_prior) / max(self.temperature, 1e-6)
        weights = F.softmax(logits, dim=1)
        fused = (weights.unsqueeze(-1) * view_tokens).sum(dim=1)
        return F.normalize(fused, dim=-1), weights


class CalibratedMultiLevelResidual(nn.Module):
    """Fuse CLIP patch residuals while accounting for reference variability."""

    def __init__(self, calibration: float = 0.25, temperature: float = 0.10) -> None:
        super().__init__()
        self.calibration = float(calibration)
        self.temperature = float(temperature)

    def forward(
        self,
        query_levels: Sequence[Tensor],
        reference_levels: Sequence[Tensor],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        if len(query_levels) != len(reference_levels) or not query_levels:
            raise ValueError("Query and reference levels must be non-empty and aligned.")

        maps = []
        dispersions = []
        strengths = []
        for query, reference in zip(query_levels, reference_levels):
            reference = F.normalize(reference, dim=-1)
            residual = nearest_patch_distance(query, reference)
            centre = F.normalize(reference.mean(dim=1, keepdim=True), dim=-1)
            dispersion = (0.5 * (1.0 - (reference * centre).sum(dim=-1))).mean(dim=1)
            offset = self.calibration * dispersion.unsqueeze(1)
            residual = ((residual - offset).clamp_min(0.0) / (1.0 - offset).clamp_min(1e-6)).clamp_max(1.0)
            topk = max(1, int(residual.shape[1] * 0.05))
            strength = residual.topk(topk, dim=1).values.mean(dim=1)
            maps.append(residual)
            dispersions.append(dispersion)
            strengths.append(strength)

        maps_tensor = torch.stack(maps, dim=1)
        dispersion_tensor = torch.stack(dispersions, dim=1)
        strength_tensor = torch.stack(strengths, dim=1)
        reliability = -torch.log(dispersion_tensor.clamp_min(1e-4))
        logits = strength_tensor / max(self.temperature, 1e-6) + 0.15 * reliability
        layer_weights = F.softmax(logits, dim=1)
        fused = (layer_weights.unsqueeze(-1) * maps_tensor).sum(dim=1)
        return fused, {
            "level_maps": maps_tensor,
            "level_weights": layer_weights,
            "reference_dispersion": dispersion_tensor,
        }


class ContentAdaptiveSemanticBank(nn.Module):
    """Select prompt mixtures according to the current dynamic visual feature."""

    def __init__(self, selection_temperature: float = 0.10, score_temperature: float = 0.07) -> None:
        super().__init__()
        self.selection_temperature = float(selection_temperature)
        self.score_temperature = float(score_temperature)

    def forward(
        self,
        image_feature: Tensor,
        normal_bank: Tensor,
        anomaly_bank: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        image_feature = F.normalize(image_feature, dim=-1)
        normal_bank = F.normalize(normal_bank, dim=-1)
        anomaly_bank = F.normalize(anomaly_bank, dim=-1)
        normal_similarity = torch.einsum("bd,bnd->bn", image_feature, normal_bank)
        anomaly_similarity = torch.einsum("bd,bnd->bn", image_feature, anomaly_bank)
        normal_weights = F.softmax(normal_similarity / self.selection_temperature, dim=1)
        anomaly_weights = F.softmax(anomaly_similarity / self.selection_temperature, dim=1)
        normal_prototype = F.normalize(
            (normal_weights.unsqueeze(-1) * normal_bank).sum(dim=1), dim=-1
        )
        anomaly_prototype = F.normalize(
            (anomaly_weights.unsqueeze(-1) * anomaly_bank).sum(dim=1), dim=-1
        )
        logits = torch.stack(
            (
                (image_feature * normal_prototype).sum(dim=-1),
                (image_feature * anomaly_prototype).sum(dim=-1),
            ),
            dim=1,
        ) / self.score_temperature
        anomaly_score = F.softmax(logits, dim=1)[:, 1]
        return anomaly_score, {
            "normal_prompt_weights": normal_weights,
            "anomaly_prompt_weights": anomaly_weights,
            "normal_prototype": normal_prototype,
            "anomaly_prototype": anomaly_prototype,
        }


class TagSemanticAligner(nn.Module):
    """D_tag: category/tag-oriented visual-language alignment branch.

    The branch deliberately operates on the same CLIP embedding space as the
    normal/anomaly prompt banks.  It is therefore a lightweight counterpart of
    the tag-alignment block shown in the paper, rather than a second backbone.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, image_feature: Tensor, normal_bank: Tensor, anomaly_bank: Tensor):
        image_feature = F.normalize(image_feature, dim=-1)
        normal_bank = F.normalize(normal_bank, dim=-1)
        anomaly_bank = F.normalize(anomaly_bank, dim=-1)
        normal_similarity = torch.einsum("bd,bnd->bn", image_feature, normal_bank).amax(dim=1)
        anomaly_similarity = torch.einsum("bd,bnd->bn", image_feature, anomaly_bank).amax(dim=1)
        logits = torch.stack((normal_similarity, anomaly_similarity), dim=1) / max(self.temperature, 1e-6)
        return F.softmax(logits, dim=1)[:, 1], logits


class IlluminationSemanticAligner(nn.Module):
    """D_illum: illumination-robust alignment branch.

    CLIP does not expose a dedicated illumination stream.  This branch uses a
    channel-normalized visual token, which suppresses global feature gain and
    offset before comparing it with the same text prototypes.  It is kept
    separate so a future learned illumination encoder can replace it without
    changing the surrounding model API.
    """

    def __init__(self, temperature: float = 0.07, eps: float = 1e-6) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.eps = float(eps)

    def forward(self, image_feature: Tensor, normal_bank: Tensor, anomaly_bank: Tensor):
        image_feature = image_feature - image_feature.mean(dim=-1, keepdim=True)
        image_feature = image_feature / image_feature.std(dim=-1, keepdim=True).clamp_min(self.eps)
        image_feature = F.normalize(image_feature, dim=-1)
        normal_bank = F.normalize(normal_bank, dim=-1)
        anomaly_bank = F.normalize(anomaly_bank, dim=-1)
        normal_similarity = torch.einsum("bd,bnd->bn", image_feature, normal_bank).amax(dim=1)
        anomaly_similarity = torch.einsum("bd,bnd->bn", image_feature, anomaly_bank).amax(dim=1)
        logits = torch.stack((normal_similarity, anomaly_similarity), dim=1) / max(self.temperature, 1e-6)
        return F.softmax(logits, dim=1)[:, 1], logits


class MultiLevelAnomalyRanker(nn.Module):
    """Stage3/4 ranking and difficulty estimation defined in the paper."""

    def __init__(
        self,
        score_weights: Sequence[float] = (0.60, 0.25, 0.15),
        difficulty_weights: Sequence[float] = (0.40, 0.25, 0.35),
    ) -> None:
        super().__init__()
        if len(score_weights) != 3 or len(difficulty_weights) != 3:
            raise ValueError("Both weight groups must contain three values.")
        if not math.isclose(sum(score_weights), 1.0, abs_tol=1e-6):
            raise ValueError("Anomaly score weights must sum to one.")
        if not math.isclose(sum(difficulty_weights), 1.0, abs_tol=1e-6):
            raise ValueError("Difficulty weights must sum to one.")
        self.score_weights = tuple(float(value) for value in score_weights)
        self.difficulty_weights = tuple(float(value) for value in difficulty_weights)

    @staticmethod
    def texture_complexity(patch_features: Tensor) -> Tensor:
        patch_features = F.normalize(patch_features, dim=-1)
        centre = F.normalize(patch_features.mean(dim=1, keepdim=True), dim=-1)
        complexity = (1.0 - (patch_features * centre).sum(dim=-1)).mean(dim=1)
        return (2.0 * complexity).clamp(0.0, 1.0)

    def forward(
        self,
        patch_map: Tensor,
        image_score: Tensor,
        semantic_score: Tensor,
        patch_features: Tensor,
        visual_distance: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        patch_score = patch_map.amax(dim=1)
        image_score = image_score.flatten().clamp(0.0, 1.0)
        semantic_score = semantic_score.flatten().clamp(0.0, 1.0)
        alpha, beta, gamma = self.score_weights
        final_score = alpha * patch_score + beta * image_score + gamma * semantic_score

        texture = self.texture_complexity(patch_features)
        semantic_ambiguity = 1.0 - (2.0 * semantic_score - 1.0).abs()
        separation = torch.exp(-visual_distance.flatten().clamp_min(0.0) / 0.25)
        wt, ws, wa = self.difficulty_weights
        difficulty = (wt * texture + ws * semantic_ambiguity + wa * separation).clamp(0.0, 1.0)
        return final_score.clamp(0.0, 1.0), {
            "patch_score": patch_score,
            "image_score": image_score,
            "semantic_score": semantic_score,
            "texture_complexity": texture,
            "semantic_ambiguity": semantic_ambiguity,
            "separation_difficulty": separation,
            "difficulty": difficulty,
        }

    @staticmethod
    def curriculum_weight(difficulty: Tensor, threshold: float, sharpness: float = 12.0) -> Tensor:
        return torch.sigmoid(sharpness * (float(threshold) - difficulty))
