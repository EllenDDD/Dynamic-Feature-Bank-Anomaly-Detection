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
import torch.nn.functional as F
from datasets import loader
from torchvision import transforms
from open_clip.utils.meters import EpochTimer, TrainMeter, ValMeter
from sklearn.metrics import average_precision_score, roc_auc_score
from binary_focal_loss import BinaryFocalLoss
import torch.distributed as dist
import matplotlib.pyplot as plt
from open_clip.model import get_cast_dtype
from open_clip.utils.env import checkpoint_pathmgr as pathmgr

logger = logging.get_logger(__name__)

def _convert_to_rgb(image):
    return image.convert('RGB')

def train_epoch(
    train_loader,
    model,
    optimizer,
    tokenizer,
    cfg,
    cur_epoch,
):
    """
    Perform the training for one epoch.
    Args:
        train_loader (loader): training loader.
        model (model): the model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        scaler (GradScaler): the `GradScaler` to help perform the steps of gradient scaling.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            open_clip/config/defaults.py
    """
    # Enable train mode.
    model.train()

    all_loss = 0.0
    all_difficulty = 0.0
    thresholds = list(cfg.ADAPTIVE.CURRICULUM_THRESHOLDS)
    stage_length = max(1, int(np.ceil(cfg.TRAIN.EPOCHS / len(thresholds))))
    stage_index = min(cur_epoch // stage_length, len(thresholds) - 1)
    threshold = thresholds[stage_index]
    for cur_iter, (inputs, types, labels) in enumerate(train_loader):
        preds, details = model(
            tokenizer,
            inputs,
            types,
            None,
            labels=labels,
            return_details=True,
        )
        labels = labels.to(preds.device).float().flatten()
        difficulty = details["training_difficulty"]
        weights = model.anomaly_ranker.curriculum_weight(
            difficulty,
            threshold,
            cfg.ADAPTIVE.CURRICULUM_SHARPNESS,
        )
        weights = weights / weights.mean().clamp_min(1e-6)
        detection_loss = F.binary_cross_entropy(
            preds.clamp(1e-6, 1.0 - 1e-6), labels, reduction="none"
        )
        detection_loss = (weights * detection_loss).mean()

        normal_scores = preds[labels < 0.5]
        anomaly_scores = preds[labels >= 0.5]
        if len(normal_scores) and len(anomaly_scores):
            pairwise_margin = (
                cfg.ADAPTIVE.RANK_MARGIN
                - anomaly_scores.unsqueeze(1)
                + normal_scores.unsqueeze(0)
            )
            ranking_loss = F.relu(pairwise_margin).mean()
        else:
            ranking_loss = preds.new_zeros(())
        loss = detection_loss + cfg.ADAPTIVE.RANK_LOSS_WEIGHT * ranking_loss

        # check Nan Loss.
        misc.check_nan_losses(loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        loss.backward()

        # Update the parameters.
        optimizer.step()

        # dist.all_reduce(loss)
        loss_value = loss.item()
        all_loss = all_loss + loss_value
        all_difficulty += difficulty.mean().item()

    all_loss = all_loss / (cur_iter + 1)
    all_difficulty = all_difficulty / (cur_iter + 1)
    print(
        "train_loss: {:.6f}, stage: {}, threshold: {:.2f}, mean_difficulty: {:.4f}".format(
            all_loss, stage_index + 1, threshold, all_difficulty
        )
    )
    return all_loss


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

        preds, _ = model(tokenizer, inputs, types, None)

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

def train(cfg):
    """
    Train a model on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in open_clip/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    if cfg.NUM_GPUS:
        device = torch.cuda.current_device()

    # Build the model and print model statistics.
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

    if torch.cuda.is_available():
        assert (
            cfg.NUM_GPUS <= torch.cuda.device_count()
        ), "Cannot use more GPU devices than available"
    else:
        assert (
            cfg.NUM_GPUS == 0
        ), "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."

    if cfg.NUM_GPUS:
        # Transfer the model to the current GPU device
        model = model.cuda(device=device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[device], output_device=device
        )

    transform = transforms.Compose([
        transforms.Resize(size=240, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=(240, 240)),
        _convert_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    # Load a checkpoint to resume training if applicable.
    with pathmgr.open(cfg.TRAIN.CHECKPOINT_FILE_PATH, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")
    start_epoch = 0
    # model = model.module
    model.load_state_dict(checkpoint, strict=False)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
        betas=[0.9, 0.999],
    )

    # Create the train and val loaders.
    train_loader = loader.construct_loader(cfg, "train", transform)

    tokenizer = open_clip.get_tokenizer('ViT-B-16-plus-240')

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))
    epoch_losses = []

    epoch_timer = EpochTimer()
    for cur_epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        print("Epoch: ", cur_epoch)
        # Train for one epoch.
        epoch_timer.epoch_tic()
        epoch_loss = train_epoch(
            train_loader,
            model,
            optimizer,
            tokenizer,
            cfg,
            cur_epoch,
        )
        epoch_losses.append(epoch_loss)
        epoch_timer.epoch_toc()
        logger.info(
            f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
            f"from {start_epoch} to {cur_epoch} take "
            f"{epoch_timer.avg_epoch_time():.2f}s in average and "
            f"{epoch_timer.median_epoch_time():.2f}s in median."
        )
        logger.info(
            f"For epoch {cur_epoch}, each iteraction takes "
            f"{epoch_timer.last_epoch_time()/len(train_loader):.2f}s in average. "
            f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
            f"{epoch_timer.avg_epoch_time()/len(train_loader):.2f}s in average."
        )

        path = os.path.join(cfg.OUTPUT_DIR, "checkpoints", "checkpoint_" + str(cur_epoch + 1) + ".pyth")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)

        # Test metrics are intentionally not computed during training. Evaluate a
        # fixed checkpoint once with test.py after all training decisions are final.


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
