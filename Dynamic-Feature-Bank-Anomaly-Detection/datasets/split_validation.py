"""Dataset split checks used before InCTRL training and evaluation."""

import hashlib
import json
import os
from functools import lru_cache

import numpy as np
from PIL import Image


def _as_list(value):
    if isinstance(value, (str, os.PathLike)):
        return [value]
    return list(value)


def _canonical(path):
    return os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(path))))


def load_records(json_paths, expected_target=None):
    records = []
    for json_path in _as_list(json_paths):
        with open(json_path, "r", encoding="utf-8") as handle:
            current = json.load(handle)
        if not isinstance(current, list):
            raise ValueError("Dataset JSON must contain a list: {}".format(json_path))
        for index, record in enumerate(current):
            if not isinstance(record, dict) or "image_path" not in record:
                raise ValueError("Invalid record {} in {}".format(index, json_path))
            if expected_target is not None and int(record.get("target", -1)) != expected_target:
                raise ValueError(
                    "Unexpected target in {} record {}: expected {}".format(
                        json_path, index, expected_target
                    )
                )
        records.extend(current)
    return records


def record_paths(records):
    return {_canonical(record["image_path"]): record["image_path"] for record in records}


@lru_cache(maxsize=None)
def _content_digest_cached(canonical_path, size, modified_ns):
    del size, modified_ns
    digest = hashlib.sha256()
    suffix = os.path.splitext(canonical_path)[1].lower()
    if suffix == ".npy":
        array = np.load(canonical_path, allow_pickle=False)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
    else:
        with Image.open(canonical_path) as image:
            image = image.convert("RGB")
            digest.update(repr(image.size).encode("ascii"))
            digest.update(image.tobytes())
    return digest.hexdigest()


def content_digest(path):
    """Hash decoded sample content so copied files cannot bypass split checks."""
    canonical_path = _canonical(path)
    if not os.path.isfile(canonical_path):
        raise FileNotFoundError("Dataset image does not exist: {}".format(path))
    stat = os.stat(canonical_path)
    return _content_digest_cached(canonical_path, stat.st_size, stat.st_mtime_ns)


def record_contents(records):
    contents = {}
    for record in records:
        contents.setdefault(content_digest(record["image_path"]), record["image_path"])
    return contents


def _find_overlap(left_records, right_records):
    left_paths = record_paths(left_records)
    right_paths = record_paths(right_records)
    path_overlap = set(left_paths).intersection(right_paths)

    left_contents = record_contents(left_records)
    right_contents = record_contents(right_records)
    content_overlap = set(left_contents).intersection(right_contents)
    examples = []
    for key in sorted(path_overlap):
        examples.append(right_paths[key])
    for key in sorted(content_overlap):
        if right_contents[key] not in examples:
            examples.append(right_contents[key])
    return path_overlap, content_overlap, examples[:5]


def validate_reference_pool(reference_normal_json, test_normal_json, test_outlier_json):
    """Ensure few-shot prompts come from clean data outside the test split."""
    references = load_records(reference_normal_json, expected_target=0)
    test_normal = load_records(test_normal_json, expected_target=0)
    test_outlier = load_records(test_outlier_json, expected_target=1)
    if not references:
        raise ValueError("The normal reference pool is empty.")

    path_overlap, content_overlap, examples = _find_overlap(
        references, test_normal + test_outlier
    )
    if path_overlap or content_overlap:
        raise ValueError(
            "Data leakage detected: the normal reference pool overlaps the test split "
            "({} path match(es), {} decoded-content match(es)). Examples: {}".format(
                len(path_overlap), len(content_overlap), examples
            )
        )
    return {
        "reference_normal": len(references),
        "test_normal": len(test_normal),
        "test_outlier": len(test_outlier),
        "path_overlap": 0,
        "content_overlap": 0,
    }


def validate_no_data_leakage(
    train_normal_json,
    train_outlier_json,
    test_normal_json,
    test_outlier_json,
    require_train_outliers=True,
):
    train_normal = load_records(train_normal_json, expected_target=0)
    train_outlier = load_records(train_outlier_json, expected_target=1)
    test_normal = load_records(test_normal_json, expected_target=0)
    test_outlier = load_records(test_outlier_json, expected_target=1)

    if not train_normal:
        raise ValueError("The training normal JSON is empty.")
    contradictory_paths, contradictory_contents, examples = _find_overlap(
        train_normal, train_outlier
    )
    if contradictory_paths or contradictory_contents:
        raise ValueError(
            "Images have both normal and anomaly labels "
            "({} path match(es), {} decoded-content match(es)): {}".format(
                len(contradictory_paths), len(contradictory_contents), examples
            )
        )

    path_overlap, content_overlap, examples = _find_overlap(
        train_normal + train_outlier, test_normal + test_outlier
    )
    if path_overlap or content_overlap:
        raise ValueError(
            "Data leakage detected: the training and test splits overlap "
            "({} path match(es), {} decoded-content match(es)). Examples: {}".format(
                len(path_overlap), len(content_overlap), examples
            )
        )

    if require_train_outliers and not train_outlier:
        raise ValueError(
            "The training outlier JSON is empty. Generate Stage1 synthetic anomalies and "
            "re-run gen_train_json.py with --synthetic-outlier-dir."
        )

    return {
        "train_normal": len(train_normal),
        "train_outlier": len(train_outlier),
        "test_normal": len(test_normal),
        "test_outlier": len(test_outlier),
        "path_overlap": 0,
        "content_overlap": 0,
    }
