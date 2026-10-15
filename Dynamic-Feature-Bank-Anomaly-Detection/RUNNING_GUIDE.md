# Running Guide

## 1. Environment

Use Python 3.10 or newer and install a compatible PyTorch/torchvision pair first. Install the remaining imports with:

```powershell
python -m pip install -r requirements.txt
```

The release process does not install packages or download weights.

## 2. Data and manifests

Keep datasets outside the repository. Generate MVTec-style train and validation manifests with the scripts under `datasets/preprocess/`, for example:

```powershell
python datasets/preprocess/gen_train_json.py --dataset-dir ./data/mvtec_ad_texture --dataset-name <category> --output-dir ./configs/AD_json
python datasets/preprocess/gen_val_json.py --dataset_dir ./data/mvtec_ad_texture --dataset_name <category> --output_dir ./configs/AD_json
```

Training outliers must be synthetic anomalies from the training side. Do not use test anomalies as training outliers.

## 3. Training

From the repository root, provide four manifest paths:

```powershell
python main.py --normal_json_path <train-normal-json> --outlier_json_path <train-outlier-json> --val_normal_json_path <val-normal-json> --val_outlier_json_path <val-outlier-json> --epochs 10 --shot 2
```

The command is the argument interface implemented by `main.py`; hardware, checkpoint, and distributed settings remain configuration-dependent.

## 4. Few-shot evaluation

```powershell
python test.py --normal_json_path <reference-normal-json> --val_normal_json_path <test-normal-json> --val_outlier_json_path <test-outlier-json> --category <category> --shot 8 --reference_seed 0
```

The reference pool is checked against the test split by path and decoded image content.

## 5. Adaptive inference

```powershell
python infer_adaptive.py --normal-json <test-normal-json> --outlier-json <test-outlier-json> --reference-normal-json <reference-normal-json> --checkpoint ./weights/<checkpoint-file> --category <category> --shot 8 --output-dir ./outputs/adaptive_inctrl
```

This writes inference JSON/CSV and optional visual overlays only when executed. Those outputs are intentionally ignored and not included.

## 6. Static tests

The included tests can be run without training or inference:

```powershell
python -m unittest discover -s tests -v
```

Full experiment reproduction has not been run as part of release preparation.
