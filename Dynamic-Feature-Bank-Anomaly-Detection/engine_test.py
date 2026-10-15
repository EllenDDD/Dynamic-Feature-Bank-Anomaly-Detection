# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

"""Train/Evaluation workflow."""
import os
import random
import json
import open_clip
from open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss
import open_clip.utils.checkpoint as cu
import open_clip.utils.distributed as du
import open_clip.utils.logging as logging
import open_clip.utils.misc as misc
import numpy as np
import torch
from datasets import loader
from torchvision import transforms
from open_clip.utils.meters import EpochTimer, TrainMeter, ValMeter
from sklearn.metrics import average_precision_score, roc_auc_score
from binary_focal_loss import BinaryFocalLoss
import torch.distributed as dist
import matplotlib.pyplot as plt
from open_clip.model import get_cast_dtype
from open_clip.utils.env import checkpoint_pathmgr as pathmgr
from PIL import Image

logger = logging.get_logger(__name__)

def _convert_to_rgb(image):
    return image.convert('RGB')

import os
import torch
import numpy as n
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from PIL import Image

def visualize_heatmap(M, orig_img_path, save_root, prefix):
    # Extract defect category automatically (e.g. "good", "crack", etc.)
    defect_type = os.path.basename(os.path.dirname(orig_img_path))
    save_dir = os.path.join(save_root, defect_type)
    os.makedirs(save_dir, exist_ok=True)

    # Normalize residual map
    M = (M - M.min()) / (M.max() - M.min() + 1e-8)
    M = cv2.resize(M, (240, 240))  # match model input size (15×15→240×240)

    # Save standalone heatmap
    plt.figure(figsize=(4, 4))
    sns.heatmap(M, cmap="YlGnBu", cbar=False, vmin=0, vmax=1)
    plt.axis('off')
    heatmap_path = os.path.join(save_dir, f"{prefix}_heatmap.png")
    plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Overlay on original image
    orig = np.array(Image.open(orig_img_path).convert("RGB"))
    orig = cv2.resize(orig, (240, 240))  # same size for overlay
    heatmap_color = cv2.applyColorMap((M * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig, 0.6, heatmap_color, 0.4, 0)
    overlay_path = os.path.join(save_dir, f"{prefix}_overlay.png")
    cv2.imwrite(overlay_path, overlay)

    print(f"✅ Saved to {defect_type}/{prefix}_heatmap.png and _overlay.png")


@torch.no_grad()
def eval_epoch(val_loader, model, cfg, tokenizer, mode=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            open_clip/config/defaults.py
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()

    total_label = torch.Tensor([]).cuda()
    total_pred = torch.Tensor([]).cuda()

    for cur_iter, (inputs, types, labels) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            labels = labels.cuda()

        preds, details = model(
            tokenizer,
            inputs,
            types,
            None,
            return_details=True,
        )
        img_ref_score = details["image_score"]
        patch_ref_map = details["patch_map"]

        # get heatmap for this image
        M = patch_ref_map[0].detach().cpu().numpy()
        M = M.reshape(15, 15)  # InCTRL patch map size

        # get original image path
        img_path = val_loader.dataset.image[cur_iter]["image_path"]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # save results
        visualize_heatmap(M, img_path, save_root="results/heatmaps", prefix=img_name)  #热力图储存路径

        total_pred = torch.cat((total_pred, preds), 0)
        total_label = torch.cat((total_label, labels), 0)

    total_pred = total_pred.cpu().numpy()  #.squeeze()
    total_label = total_label.cpu().numpy()

    print("Predict " + mode + " set: ")
    total_roc, total_pr = aucPerformance(total_pred, total_label)

    return total_roc

def aucPerformance(mse, labels, prt=True):
    roc_auc = roc_auc_score(labels, mse)
    ap = average_precision_score(labels, mse)
    if prt:
        print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    return roc_auc, ap;


def drawing(cfg, data, xlabel, ylabel, dir):
    plt.switch_backend('Agg')
    plt.figure()
    plt.plot(data, 'b', label='loss')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, dir))


def test(cfg, load=None, mode = None):
    """
    Perform testing on the pretrained model.
    Args:
        cfg (CfgNode): configs. Details can be found in open_clip/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)
    device = torch.cuda.current_device()

    transform = transforms.Compose([
        transforms.Resize(size=240, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=(240, 240)),
        _convert_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    cf = './open_clip/model_configs/ViT-B-16-plus-240.json'
    with open(cf, 'r') as f:
        model_cfg = json.load(f)
    embed_dim = model_cfg["embed_dim"]
    vision_cfg = model_cfg["vision_cfg"]
    text_cfg = model_cfg["text_cfg"]
    cast_dtype = get_cast_dtype('fp32')
    quick_gelu = False

    model = open_clip.model.AdaptiveInCTRL(
        cfg, embed_dim, vision_cfg, text_cfg, quick_gelu, cast_dtype=cast_dtype
    )

    model = model.cuda(device=device)

    cu.load_test_checkpoint(cfg, model)

    tokenizer = open_clip.get_tokenizer('ViT-B-16-plus-240')

    if load == None:
        load = loader.construct_loader(cfg, "test", transform)
        mode = "test"

    # Create meters.
    total_roc = eval_epoch(load, model, cfg, tokenizer, mode)

    return total_roc
