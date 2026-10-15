import unittest

import torch

from open_clip.adaptive_anomaly import (
    AdaptiveResolutionScanner,
    CalibratedMultiLevelResidual,
    ContentAdaptiveSemanticBank,
    IlluminationSemanticAligner,
    MultiLevelAnomalyRanker,
    TagSemanticAligner,
)
from open_clip.gpt_semantic import GPTSemanticValidator


class AdaptiveAnomalyTests(unittest.TestCase):
    def test_scanner_locates_and_crops_peak(self):
        scanner = AdaptiveResolutionScanner(scales=(0.5, 1.0))
        image = torch.randn(1, 3, 32, 32)
        anomaly_map = torch.zeros(1, 8, 8)
        anomaly_map[0, 6, 1] = 10.0
        views, centre = scanner.make_views(image, anomaly_map)
        self.assertEqual(len(views), 2)
        self.assertEqual(views[0].shape, image.shape)
        self.assertLess(float(centre[0, 0]), 0.0)
        self.assertGreater(float(centre[0, 1]), 0.0)

    def test_calibrated_residual_prefers_changed_query(self):
        module = CalibratedMultiLevelResidual()
        reference = torch.eye(4).unsqueeze(0)
        identical = reference.clone()
        changed = -reference.clone()
        identical_map, _ = module([identical], [reference])
        changed_map, details = module([changed], [reference])
        self.assertEqual(changed_map.shape, (1, 4))
        self.assertEqual(details["level_weights"].shape, (1, 1))
        self.assertGreater(float(changed_map.mean()), float(identical_map.mean()))

    def test_semantic_bank_selects_anomaly_prototype(self):
        module = ContentAdaptiveSemanticBank()
        image = torch.tensor([[0.0, 1.0]])
        normal = torch.tensor([[[1.0, 0.0], [0.9, 0.1]]])
        anomaly = torch.tensor([[[0.0, 1.0], [0.1, 0.9]]])
        score, details = module(image, normal, anomaly)
        self.assertGreater(float(score[0]), 0.9)
        self.assertEqual(details["anomaly_prompt_weights"].shape, (1, 2))

    def test_ranker_uses_declared_score_weights(self):
        ranker = MultiLevelAnomalyRanker()
        patch_map = torch.tensor([[0.2, 0.8]])
        image_score = torch.tensor([0.4])
        semantic_score = torch.tensor([0.6])
        patch_features = torch.randn(1, 2, 4)
        visual_distance = torch.tensor([0.2])
        score, details = ranker(
            patch_map, image_score, semantic_score, patch_features, visual_distance
        )
        expected = 0.60 * 0.8 + 0.25 * 0.4 + 0.15 * 0.6
        self.assertAlmostEqual(float(score[0]), expected, places=6)
        self.assertTrue(0.0 <= float(details["difficulty"][0]) <= 1.0)

    def test_tag_and_illum_alignment_branches(self):
        image = torch.randn(2, 8)
        normal = torch.randn(2, 3, 8)
        anomaly = torch.randn(2, 3, 8)
        for branch in (TagSemanticAligner(), IlluminationSemanticAligner()):
            score, logits = branch(image, normal, anomaly)
            self.assertEqual(score.shape, (2,))
            self.assertEqual(logits.shape, (2, 2))
            self.assertTrue(torch.isfinite(score).all())

    def test_gpt_validator_is_offline_by_default(self):
        result = GPTSemanticValidator().validate("a normal texture")
        self.assertFalse(result.enabled)
        self.assertEqual(result.text, "a normal texture")


if __name__ == "__main__":
    unittest.main()
