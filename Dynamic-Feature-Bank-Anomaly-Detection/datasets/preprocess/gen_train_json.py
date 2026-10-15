"""Generate leakage-free training JSON files for an MVTec-style dataset."""

import argparse
import json
import os
from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".npy", ".png", ".tif", ".tiff"}


def get_args_parser():
    parser = argparse.ArgumentParser("Leakage-free AD training JSON generation")
    parser.add_argument(
        "--dataset_dir",
        "--dataset-dir",
        dest="dataset_dir",
        type=Path,
        default=Path("./data/mvtec_ad_texture"),
        help="Dataset root containing <category>/train/good.",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs" / "AD_json",
        help="Directory for <category>_normal.json and <category>_outlier.json.",
    )
    parser.add_argument(
        "--dataset_name",
        "--dataset-name",
        dest="dataset_name",
        default="yellow_cloth",
        help="Category directory name.",
    )
    parser.add_argument(
        "--synthetic_outlier_dir",
        "--synthetic-outlier-dir",
        dest="synthetic_outlier_dir",
        type=Path,
        default=None,
        help="Optional directory containing Stage1 synthetic anomaly images.",
    )
    return parser


def image_files(directory, recursive=False):
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(
        path.resolve()
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def is_within(path, parent):
    try:
        return os.path.commonpath([str(path.resolve()), str(parent.resolve())]) == str(parent.resolve())
    except ValueError:
        return False


def records(paths, target, dataset_name):
    return [
        {"image_path": str(path), "target": target, "type": dataset_name}
        for path in paths
    ]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def main():
    args = get_args_parser().parse_args()
    image_dir = (args.dataset_dir / args.dataset_name).resolve()
    normal_dir = image_dir / "train" / "good"
    test_dir = image_dir / "test"

    if not normal_dir.is_dir():
        raise FileNotFoundError("Training normal directory does not exist: {}".format(normal_dir))

    normal_paths = image_files(normal_dir)
    if not normal_paths:
        raise ValueError("No training normal images found in {}".format(normal_dir))

    synthetic_paths = []
    if args.synthetic_outlier_dir is not None:
        synthetic_dir = args.synthetic_outlier_dir.resolve()
        if not synthetic_dir.is_dir():
            raise FileNotFoundError("Synthetic anomaly directory does not exist: {}".format(synthetic_dir))
        if test_dir.is_dir() and is_within(synthetic_dir, test_dir):
            raise ValueError(
                "Synthetic anomalies must not come from the test split: {}".format(synthetic_dir)
            )
        synthetic_paths = image_files(synthetic_dir, recursive=True)
        if not synthetic_paths:
            raise ValueError("No synthetic anomaly images found in {}".format(synthetic_dir))

    normal_output = records(normal_paths, 0, args.dataset_name)
    outlier_output = records(synthetic_paths, 1, args.dataset_name)
    normal_path = args.output_dir / (args.dataset_name + "_normal.json")
    outlier_path = args.output_dir / (args.dataset_name + "_outlier.json")
    write_json(normal_path, normal_output)
    write_json(outlier_path, outlier_output)

    print("Wrote {} training normal records to {}".format(len(normal_output), normal_path))
    print("Wrote {} synthetic anomaly records to {}".format(len(outlier_output), outlier_path))
    if not outlier_output:
        print(
            "WARNING: the outlier JSON is empty. Supply --synthetic-outlier-dir "
            "after Stage1 generation before training. Real test anomalies are never used."
        )


if __name__ == "__main__":
    main()
