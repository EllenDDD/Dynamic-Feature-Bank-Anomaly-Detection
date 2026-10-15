import json
import os
import tempfile
import unittest

from PIL import Image
from torchvision import transforms

from datasets.IC_dataset_new import IC_dataset


class DatasetReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = self.temp_dir.name
        self.to_tensor = transforms.ToTensor()

    def tearDown(self):
        self.temp_dir.cleanup()

    def image(self, name, colour):
        path = os.path.join(self.root, name)
        Image.new("RGB", (8, 8), colour).save(path)
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

    def test_fixed_seed_reuses_one_reference_set(self):
        query_normal = self.manifest(
            "query_normal.json", [self.image("query.png", "white")], 0
        )
        query_outlier = self.manifest(
            "query_outlier.json", [self.image("outlier.png", "red")], 1
        )
        reference_paths = [
            self.image("reference_{}.png".format(index), (index * 20, 10, 30))
            for index in range(6)
        ]
        references = self.manifest("references.json", reference_paths, 0)

        first = IC_dataset(
            self.root, query_normal, query_outlier, self.to_tensor, 2,
            reference_normal_json_path=references, fixed_reference_seed=17,
        )
        second = IC_dataset(
            self.root, query_normal, query_outlier, self.to_tensor, 2,
            reference_normal_json_path=references, fixed_reference_seed=17,
        )
        first_paths = [item["image_path"] for item in first.fixed_reference_samples["fabric"]]
        second_paths = [item["image_path"] for item in second.fixed_reference_samples["fabric"]]
        self.assertEqual(first_paths, second_paths)
        self.assertEqual(len(first[0][0]), 3)
        self.assertEqual(len(first[1][0]), 3)

    def test_training_query_cannot_reference_itself(self):
        only_image = self.image("only.png", "white")
        normal = self.manifest("normal.json", [only_image], 0)
        outlier = self.manifest("outlier.json", [], 1)
        dataset = IC_dataset(
            self.root, normal, outlier, self.to_tensor, 1,
            reference_normal_json_path=normal,
        )

        with self.assertRaisesRegex(ValueError, "independent clean reference"):
            dataset[0]


if __name__ == "__main__":
    unittest.main()
