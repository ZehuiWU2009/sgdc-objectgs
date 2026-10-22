#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
import time
import numpy as np

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from plyfile import PlyData

def load_label_mapping(model_path):
    """从 PLY 文件重建 one-hot 索引 -> 原始 label ID 的映射"""
    pc_dir = os.path.join(model_path, "point_cloud")
    if not os.path.exists(pc_dir):
        return None
    
    iterations = [d for d in os.listdir(pc_dir) if d.startswith("iteration_")]
    if not iterations:
        return None
    
    latest = sorted(iterations, key=lambda x: int(x.split("_")[1]))[-1]
    ply_path = os.path.join(pc_dir, latest, "point_cloud.ply")
    
    if not os.path.exists(ply_path):
        return None
    
    plydata = PlyData.read(ply_path)
    label_ids = np.asarray(plydata.elements[0]["label"]).astype(np.int64)
    
    unique_labels = np.unique(label_ids)
    idx_to_label = {0: 0}
    foreground = sorted(unique_labels[unique_labels != 0].tolist())
    for i, label in enumerate(foreground, start=1):
        idx_to_label[i] = label
    
    return idx_to_label

def readImages(renders_dir, gt_dir, render_object_masks_dir, gt_object_masks_dir):
    renders = []
    gts = []
    render_object_masks = []
    gt_object_masks = []
    image_names = []
    
    fnames = sorted(os.listdir(renders_dir))
    for fname in fnames:
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        render_object_mask = Image.open(render_object_masks_dir / fname).convert('L')
        gt_object_mask = Image.open(gt_object_masks_dir / fname).convert('L')
        
        render_image = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
        render_mask = tf.to_tensor(render).unsqueeze(0)[:, 3:4, :, :].cuda()
        render_image = render_image * render_mask
        gt_image = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()
        gt_mask = tf.to_tensor(gt).unsqueeze(0)[:, 3:4, :, :].cuda()
        gt_image = gt_image * gt_mask
        
        render_object_mask = torch.from_numpy(np.array(render_object_mask, dtype=np.uint8)).unsqueeze(0).cuda()
        gt_object_mask = torch.from_numpy(np.array(gt_object_mask, dtype=np.uint8)).unsqueeze(0).cuda()
        
        renders.append(render_image)
        gts.append(gt_image)
        image_names.append(fname)
        render_object_masks.append(render_object_mask)
        gt_object_masks.append(gt_object_mask)

    return renders, gts, image_names, render_object_masks, gt_object_masks

def calculate_segmentation_metrics(pred, gt, ignore_index=0):
    pred_flat = pred.view(-1)
    gt_flat = gt.view(-1)
    
    pixel_accuracy = (pred_flat == gt_flat).float().mean()
    
    class_ids = gt_flat.unique()
    class_ids = class_ids[class_ids != ignore_index]
    
    ious = []
    dices = []
    
    for class_id in class_ids:
        pred_class = (pred_flat == class_id).float()
        gt_class = (gt_flat == class_id).float()
        
        if gt_class.sum() < 100:
            continue
            
        intersection = (pred_class * gt_class).sum()
        union = pred_class.sum() + gt_class.sum() - intersection
        iou = intersection / (union + 1e-6)
        ious.append(iou)
        
        dice = 2.0 * intersection / (pred_class.sum() + gt_class.sum() + 1e-6)
        dices.append(dice)
    
    mean_iou = torch.tensor(ious).mean() if len(ious) > 0 else torch.tensor(0.0)
    mean_dice = torch.tensor(dices).mean() if len(dices) > 0 else torch.tensor(0.0)
    
    return mean_iou, mean_dice, pixel_accuracy

def evaluate(model_paths, eval_name, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):

    # ===== ADDED: Evaluation Timing & Memory =====
    torch.cuda.reset_peak_memory_stats()
    eval_start_time = time.time()
    # =============================================
    
    full_dict = {}
    per_view_dict = {}
    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}

    test_dir = Path(scene_dir) / eval_name

    idx_to_label = load_label_mapping(model_paths)
    if idx_to_label is not None and logger:
        logger.info(f"[Mapping] Loaded {len(idx_to_label)} classes from PLY")

    for method in os.listdir(test_dir):
        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}

        base_method_dir = test_dir / method
        method_dir = base_method_dir 
        if os.path.exists(method_dir):
            gt_dir = method_dir / "gt"
            renders_dir = method_dir / "renders"
            render_object_masks_dir = method_dir / "semantic"
            gt_object_masks_dir = method_dir / "semantic_gt"
            
            renders, gts, image_names, render_object_masks, gt_object_masks = readImages(
                renders_dir, gt_dir, render_object_masks_dir, gt_object_masks_dir
            )

            ssims = []
            psnrs = []
            lpipss = []
            all_ious = []
            all_dices = []
            all_pixel_accuracies = []

            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                ssims.append(ssim(renders[idx], gts[idx]))
                psnrs.append(psnr(renders[idx], gts[idx]))
                lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())

                pred_mask = render_object_masks[idx]
                
                iou, dice, pixel_acc = calculate_segmentation_metrics(
                    pred_mask, gt_object_masks[idx]
                )
                all_ious.append(iou)
                all_dices.append(dice)
                all_pixel_accuracies.append(pixel_acc)

            mean_ious = torch.tensor(all_ious)
            mean_dices = torch.tensor(all_dices)
            mean_pixel_accuracies = torch.tensor(all_pixel_accuracies)

            if logger:
                logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
                logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
                logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
                logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
                logger.info("  mIoU : \033[1;35m{:>12.7f}\033[0m".format(mean_ious.mean().item(), ".5"))
                logger.info("  mDice: \033[1;35m{:>12.7f}\033[0m".format(mean_dices.mean().item(), ".5"))
                logger.info("  Pixel Accuracy: \033[1;35m{:>12.7f}\033[0m".format(mean_pixel_accuracies.mean().item(), ".5"))
                if visible_count is not None:
                    logger.info("  GS_NUMS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(visible_count).float().mean(), ".5"))
            print("")

            full_dict[scene_dir][method].update({
                "PSNR": torch.tensor(psnrs).mean().item(),
                "SSIM": torch.tensor(ssims).mean().item(),
                "LPIPS": torch.tensor(lpipss).mean().item(),
                "IoU": mean_ious.mean().item(),
                "Dice": mean_dices.mean().item(),
                "Pixel Accuracy": mean_pixel_accuracies.mean().item(),
            })

            per_view_dict[scene_dir][method].update({
                "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                "IoU": {name: iou for iou, name in zip(mean_ious.tolist(), image_names)},
                "Dice": {name: dice for dice, name in zip(mean_dices.tolist(), image_names)},
                "Pixel Accuracy": {name: pa for pa, name in zip(mean_pixel_accuracies.tolist(), image_names)},
            })

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)

    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)
    
    # ===== ADDED: Evaluation Stats Output =====
    eval_total_time = time.time() - eval_start_time
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    if logger:
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluation Stats:")
        logger.info(f"  Total Time: {eval_total_time:.2f} s")
        logger.info(f"  Peak GPU Memory: {peak_mem_mb:.1f} MB")
        logger.info(f"{'='*50}\n")
    
    eval_stats = {
        "eval_time_sec": eval_total_time,
        "peak_memory_mb": float(peak_mem_mb),
    }
    with open(scene_dir + "/eval_stats.json", 'w') as f:
        json.dump(eval_stats, f, indent=2)
    # ==========================================


if __name__ == "__main__":
    import logging
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s"))
    logger.addHandler(handler)
    
    lpips_fn = lpips.LPIPS(net='vgg').cuda()
    
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, type=str, default="")
    args = parser.parse_args()
    
    evaluate(args.model_paths, "test", logger=logger)