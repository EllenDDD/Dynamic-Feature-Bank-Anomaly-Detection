import json
import os
import tempfile
import unittest

from PIL import Image

from datasets.split_validation import (
    validate_no_data_leakage,
    validate_reference_pool,
)


class SplitValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def image(self, name, colour, image_format="PNG"):
        path = os.path.join(self.root, name)
        Image.new("RGB", (8, 8), colour).save(path, format=image_format)
        return path

    def manifest(self, name, paths, target):
        path = os.path.join(self.root, name)
        records = [
            {"image_path": image_path, "target": target, "type": "fabric"}
            for image_path in paths
        ]
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(records, handle)
        return path

    def test_clean_splits_pass(self):
        train_normal = self.manifest("train_normal.json", [self.image("tn.png", "white")], 0)
        train_outlier = self.manifest("train_outlier.json", [self.image("ta.png", "red")], 1)
        test_normal = self.manifest("test_normal.json", [self.image("vn.png", "gray")], 0)
        test_outlier = self.manifest("test_outlier.json", [self.image("va.png", "blue")], 1)

        result = validate_no_data_leakage(
            train_normal, train_outlier, test_normal, test_outlier
        )
        self.assertEqual(result["content_overlap"], 0)

    def test_copy_with_different_path_is_rejected(self):
        train_image = self.image("train.png", (12, 34, 56))
        copied_test_image = self.image("test.bmp", (12, 34, 56), image_format="BMP")
        train_normal = self.manifest("train_normal.json", [train_image], 0)
        train_outlier = self.manifest("train_outlier.json", [self.image("ta.png", "red")], 1)
        test_normal = self.manifest("test_normal.json", [copied_test_image], 0)
        test_outlier = self.manifest("test_outlier.json", [self.image("va.png", "blue")], 1)

        with self.assertRaisesRegex(ValueError, "decoded-content"):
            validate_no_data_leakage(
                train_normal, train_outlier, test_normal, test_outlier
            )

    def test_test_image_cannot_be_a_reference_prompt(self):
        test_image = self.image("test.png", (20, 40, 60))
        copied_reference = self.image("reference.bmp", (20, 40, 60), image_format="BMP")
        references = self.manifest("references.json", [copied_reference], 0)
        test_normal = self.manifest("test_normal.json", [test_image], 0)
        test_outlier = self.manifest("test_outlier.json", [self.image("va.png", "red")], 1)

        with self.assertRaisesRegex(ValueError, "reference pool overlaps"):
            validate_reference_pool(references, test_normal, test_outlier)


if __name__ == "__main__":
    unittest.main()
