# Dynamic Feature Bank-Guided Progressive Unsupervised Anomaly Detection for Fabric Surfaces

## Overview

This repository contains the research code for unsupervised anomaly detection on fabric and other texture surfaces. It provides training, few-shot reference-based inference, adaptive multi-resolution feature processing, semantic prompt banks, curriculum difficulty weighting, and multi-level residual scoring. Datasets, pretrained weights, checkpoints, and experiment outputs are intentionally excluded.

## Method Overview

The model is built on the local `open_clip` implementation. `AdaptiveResolutionScanner` selects nested views around high-residual regions. `ContentAdaptiveSemanticBank` builds normal and anomaly prompt banks, while `MultiLevelAnomalyRanker` combines patch residuals, image residuals, and semantic anomaly probability and supplies the easy-to-hard curriculum signal. `CalibratedMultiLevelResidual` compares query patches with few-shot normal references at multiple visual layers. `datasets/IC_dataset_new.py` and `datasets/loader.py` enforce normal-reference selection and split-leakage checks.

## Repository Structure

```text
.
├── main.py                         # training and optional evaluation entry point
├── test.py                         # few-shot evaluation entry point
├── infer_adaptive.py               # adaptive inference and optional visual outputs
├── engine_IC.py                    # training/evaluation loop
├── engine_test.py                  # evaluation loop and heatmap generation
├── binary_focal_loss.py            # focal loss
├── datasets/                       # loaders, transforms, validation, preprocessing
├── open_clip/                      # local CLIP and adaptive anomaly implementation
├── tests/                          # static/unit tests
├── README.md
├── RUNNING_GUIDE.md
├── RELEASE_REPORT.md
├── requirements.txt
└── .gitignore
```

## Requirements

Python 3.10 or newer is indicated by the source README. The code imports PyTorch, torchvision, NumPy, SciPy, scikit-image, scikit-learn, Pillow, OpenCV, matplotlib, seaborn, tqdm, fvcore, iopath, simplejson, ftfy, regex, psutil, and imgaug. See `requirements.txt`; exact versions are not asserted because the original environment file was not present.

## Installation

Create an isolated environment, install a compatible PyTorch/torchvision pair for the target CUDA or CPU platform, then install the remaining packages from `requirements.txt`. No dependency installation or model download is performed by this release process.

## Dataset Preparation

Prepare MVTec AD, Texture-AD, or WFDD outside this repository. The loader expects MVTec-style records (`<category>/train/good` and `<category>/test/...`). The preprocessing scripts include converters for AITEX, BrainMRI, ELPV, HeadCT, SDD, and VisA, plus JSON-manifest generators. Dataset files are not redistributed here. Generate manifests into a user-managed location such as `./configs/AD_json/`, or pass their paths explicitly to the entry points.

## Pretrained Weights

Training and evaluation require a compatible CLIP/InCTRL checkpoint. The original project references a ViT-B/16-plus-240 pretrained file and trained checkpoints, but no weight files or download links are included here. Place user-obtained files outside version control, for example under `./weights/`, and pass the paths through the configuration or command line.

## Configuration

`open_clip/config/defaults.py` defines batch sizes, image size, checkpoint paths, output directories, adaptive stages, score weights, curriculum thresholds, and JSON manifest defaults. Replace placeholder paths (`./data`, `./configs/AD_json`, `./weights`, and `./outputs`) for a local setup. The source code retains the original model and training logic.

## Training

The training entry point is `main.py`. It accepts `--normal_json_path`, `--outlier_json_path`, `--val_normal_json_path`, `--val_outlier_json_path`, `--epochs`, `--steps_per_epoch`, `--shot`, and configuration overrides. Training requires synthetic training outlier records; test anomalies must not be used as training outliers.

## Inference

`test.py` runs few-shot evaluation using a leakage-free normal reference pool. `infer_adaptive.py` runs the adaptive inference path and accepts normal/outlier manifest paths, a checkpoint path, category, shot count, and an output directory. Inference outputs are generated only when the user runs the scripts and are not part of this repository.

## Evaluation

The implemented evaluation reports AUROC and average precision (AUPRC/AP). The evaluation code can optionally save residual heatmaps and overlays under a user-selected output directory.

## Reproducibility Notes

Reproduction requires compatible Python/PyTorch dependencies, the relevant dataset files, generated JSON manifests, and compatible pretrained/trained weights. Static syntax checks passed for the included Python files. Full training, inference, metric reproduction, GPU compatibility, and dependency-version compatibility were not run during preparation. TODO: verify the exact public dataset split and checkpoint provenance before release.

## Citation

**Dynamic Feature Bank-Guided Progressive Unsupervised Anomaly Detection for Fabric Surfaces**

Formal journal citation information will be added after publication. Do not infer a DOI, volume, issue, or year from this repository.

## Contact

Author: Junxian Di  
Email: 1046993859@qq.com
