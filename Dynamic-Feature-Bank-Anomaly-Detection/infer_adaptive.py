"""Run AdaptiveInCTRL inference and export ranking diagnostics."""

import argparse
import csv
import json
import os
import time

import cv2
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision import transforms

import open_clip
import open_clip.utils.checkpoint as checkpoint_utils
from datasets import loader
from datasets.split_validation import validate_reference_pool
from open_clip.config.defaults import assert_and_infer_cfg, get_cfg
from open_clip.model import get_cast_dtype


def _convert_to_rgb(image):
    return image.convert("RGB")


def parse_args():
    parser = argparse.ArgumentParser(description="AdaptiveInCTRL Stage2-4 inference")
    parser.add_argument("--normal-json", default="./configs/AD_json/grid_cloth_val_normal.json")
    parser.add_argument("--outlier-json", default="./configs/AD_json/grid_cloth_val_outlier.json")
    parser.add_argument(
        "--reference-normal-json",
        default="./configs/AD_json/grid_cloth_normal.json",
        help="Leakage-free train/good JSON used only as the in-context reference pool.",
    )
    parser.add_argument("--checkpoint", default="./outputs/checkpoints/checkpoint_10.pyth")
    parser.add_argument(
        "--few-shot-dir", default=None,
        help="Deprecated and ignored; references come from --reference-normal-json.",
    )
    parser.add_argument("--category", default="pink_flower")
    parser.add_argument("--shot", type=int, default=8)
    parser.add_argument("--reference-seed", type=int, default=0)
    parser.add_argument("--output-dir", default="./outputs/adaptive_inctrl/grid_cloth_8shot")
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates the complete test set")
    parser.add_argument("--visual-count", type=int, default=3)
    return parser.parse_args()


def build_model(cfg, device):
    config_path = "./open_clip/model_configs/ViT-B-16-plus-240.json"
    with open(config_path, "r", encoding="utf-8") as handle:
        model_cfg = json.load(handle)
    model = open_clip.model.AdaptiveInCTRL(
        cfg,
        model_cfg["embed_dim"],
        model_cfg["vision_cfg"],
        model_cfg["text_cfg"],
        quick_gelu=False,
        cast_dtype=get_cast_dtype("fp32"),
    ).to(device)
    checkpoint_utils.load_test_checkpoint(cfg, model)
    model.eval()
    return model


def save_overlay(record, output_dir, suffix):
    image = cv2.imread(record["image_path"])
    if image is None:
        return
    image = cv2.resize(image, (240, 240), interpolation=cv2.INTER_CUBIC)
    residual = record.pop("_patch_map")
    residual = cv2.resize(residual, (240, 240), interpolation=cv2.INTER_CUBIC)
    residual = (residual - residual.min()) / (np.ptp(residual) + 1e-8)
    heatmap = cv2.applyColorMap((255.0 * residual).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.58, heatmap, 0.42, 0.0)
    stem = os.path.splitext(os.path.basename(record["image_path"]))[0]
    defect_type = os.path.basename(os.path.dirname(record["image_path"]))
    filename = "{}_{}_{}_score_{:.4f}.png".format(suffix, defect_type, stem, record["score"])
    cv2.imwrite(os.path.join(output_dir, filename), overlay)


def main():
    args = parse_args()
    split_stats = validate_reference_pool(
        args.reference_normal_json,
        args.normal_json,
        args.outlier_json,
    )
    print("Leakage check passed: {}".format(split_stats))
    if not torch.cuda.is_available():
        raise RuntimeError("AdaptiveInCTRL inference requires a CUDA GPU in this repository.")
    device = torch.device("cuda:0")
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = get_cfg()
    cfg.TRAIN.ENABLE = False
    cfg.TEST.ENABLE = True
    cfg.TEST.BATCH_SIZE = 1
    cfg.TEST.CHECKPOINT_FILE_PATH = args.checkpoint
    cfg.normal_json_path = args.reference_normal_json
    cfg.outlier_json_path = args.outlier_json
    cfg.val_normal_json_path = args.normal_json
    cfg.val_outlier_json_path = args.outlier_json
    cfg.category = args.category
    cfg.shot = args.shot
    cfg.reference_seed = args.reference_seed
    cfg.NUM_GPUS = 1
    cfg = assert_and_infer_cfg(cfg)

    transform = transforms.Compose([
        transforms.Resize(size=240, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=(240, 240)),
        _convert_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])
    test_loader = loader.construct_loader(cfg, "test", transform)
    tokenizer = open_clip.get_tokenizer("ViT-B-16-plus-240")
    model = build_model(cfg, device)

    records = []
    labels = []
    scores = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for index, (inputs, categories, batch_labels) in enumerate(test_loader):
            if args.limit and index >= args.limit:
                break
            prediction, details = model(
                tokenizer,
                inputs,
                categories,
                None,
                return_details=True,
            )
            score = float(prediction[0].cpu())
            label = int(batch_labels[0])
            normal_prompt_index = int(details["normal_prompt_weights"][0].argmax().cpu())
            anomaly_prompt_index = int(details["anomaly_prompt_weights"][0].argmax().cpu())
            normal_prompts, anomaly_prompts = details["prompt_text"][0]
            patch_grid = details["patch_grid"][0].detach().float().cpu().numpy()
            record = {
                "image_path": test_loader.dataset.image[index]["image_path"],
                "label": label,
                "score": score,
                "patch_score": float(details["patch_score"][0].cpu()),
                "image_score": float(details["image_score"][0].cpu()),
                "semantic_score": float(details["semantic_score"][0].cpu()),
                "difficulty": float(details["difficulty"][0].cpu()),
                "texture_complexity": float(details["texture_complexity"][0].cpu()),
                "semantic_ambiguity": float(details["semantic_ambiguity"][0].cpu()),
                "separation_difficulty": float(details["separation_difficulty"][0].cpu()),
                "scan_centre_xy": details["scan_centre"][0].float().cpu().tolist(),
                "scan_scales": list(details["scan_scales"]),
                "view_weights": details["view_weights"][0].float().cpu().tolist(),
                "level_weights": details["level_weights"][0].float().cpu().tolist(),
                "selected_normal_prompt": normal_prompts[normal_prompt_index],
                "selected_anomaly_prompt": anomaly_prompts[anomaly_prompt_index],
                "_patch_map": patch_grid,
            }
            records.append(record)
            labels.append(label)
            scores.append(score)
            print(
                "[{}/{}] label={} score={:.4f} patch={:.4f} image={:.4f} semantic={:.4f}".format(
                    index + 1,
                    min(len(test_loader), args.limit) if args.limit else len(test_loader),
                    label,
                    score,
                    record["patch_score"],
                    record["image_score"],
                    record["semantic_score"],
                )
            )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    if len(set(labels)) < 2:
        auroc = None
        auprc = None
    else:
        auroc = float(roc_auc_score(labels, scores))
        auprc = float(average_precision_score(labels, scores))

    for rank, record in enumerate(sorted(records, key=lambda item: item["score"], reverse=True), 1):
        record["anomaly_rank"] = rank
    for rank, record in enumerate(sorted(records, key=lambda item: item["difficulty"]), 1):
        record["difficulty_rank_easy_to_hard"] = rank

    visual_count = min(args.visual_count, len(records) // 2)
    visual_records = sorted(records, key=lambda item: item["score"], reverse=True)[:visual_count]
    visual_records += sorted(records, key=lambda item: item["score"])[:visual_count]
    for index, record in enumerate(visual_records):
        save_overlay(record, args.output_dir, "high" if index < visual_count else "low")
    for record in records:
        record.pop("_patch_map", None)

    summary = {
        "model": "AdaptiveInCTRL",
        "category": args.category,
        "shot": args.shot,
        "checkpoint": os.path.abspath(args.checkpoint),
        "samples": len(records),
        "auroc": auroc,
        "auprc": auprc,
        "elapsed_seconds": elapsed,
        "seconds_per_image": elapsed / max(len(records), 1),
        "stage4_score_weights": list(cfg.ADAPTIVE.SCORE_WEIGHTS),
        "stage3_difficulty_weights": list(cfg.ADAPTIVE.DIFFICULTY_WEIGHTS),
        "records": records,
    }
    json_path = os.path.join(args.output_dir, "inference_results.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    csv_path = os.path.join(args.output_dir, "inference_ranking.csv")
    csv_fields = [
        "anomaly_rank",
        "difficulty_rank_easy_to_hard",
        "label",
        "score",
        "patch_score",
        "image_score",
        "semantic_score",
        "difficulty",
        "image_path",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda item: item["anomaly_rank"]))

    print("AUROC: {}".format("n/a" if auroc is None else "{:.6f}".format(auroc)))
    print("AUPRC: {}".format("n/a" if auprc is None else "{:.6f}".format(auprc)))
    print("Results: {}".format(os.path.abspath(json_path)))


if __name__ == "__main__":
    main()
