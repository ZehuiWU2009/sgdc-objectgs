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
import os
import sys
import imageio
import yaml
from os import makedirs
import torch
import numpy as np

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

from scene import Scene
import json
import time
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state, parse_cfg, visualize_depth, visualize_normal
from utils.image_utils import save_rgba, save_mask
from argparse import ArgumentParser

def render_set(model_path, name, iteration, views, gaussians, pipe, background):
    vis_normal=False
    vis_depth=False
    vis_semantic=True
    if gaussians.gs_attr == "2D":
        vis_normal=True
        vis_depth=True
        

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    if vis_normal:
        normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normal")
        makedirs(normal_path, exist_ok=True)
    if vis_depth:
        depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
        makedirs(depth_path, exist_ok=True)
    if vis_semantic:
        semantic_gt_path = os.path.join(model_path, name, "ours_{}".format(iteration), "semantic_gt")
        semantic_path = os.path.join(model_path, name, "ours_{}".format(iteration), "semantic")
        makedirs(semantic_gt_path, exist_ok=True)   
        makedirs(semantic_path, exist_ok=True)

    modules = __import__('gaussian_renderer')
    
    # ===== ADDED: FPS & Memory Stats =====
    torch.cuda.reset_peak_memory_stats()
    t_list = []
    t_list_seg = []
    # =====================================
    
    visible_count_list = []
    per_view_dict = {}
    for idx, view in enumerate(tqdm(views, desc="rendering progress")):
        
        # ---- RGB Only ----
        torch.cuda.synchronize()
        t_start = time.time()
        render_pkg = getattr(modules, 'render')(view, gaussians, pipe, background)
        torch.cuda.synchronize()
        t_rgb_end = time.time()

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = render_pkg["visibility_filter"].sum()

        # ---- RGB + Seg ----
        if vis_semantic:
            semantic_map = render_pkg["render_semantics"]
            object_id = gaussians.id_encoder.inverse_transform(semantic_map)
            if object_id.dim() == 3:
                object_id = object_id.squeeze(0)
        torch.cuda.synchronize()
        t_seg_end = time.time()
        
        t_list.append(t_rgb_end - t_start)
        t_list_seg.append(t_seg_end - t_start)

        # gts
        gt = view.original_image.cuda()
        alpha_mask = view.alpha_mask.cuda()
        rendering = torch.cat([rendering, alpha_mask], dim=0)
        gt = torch.cat([gt, alpha_mask], dim=0)
        
        # error maps
        if gt.device != rendering.device:
            rendering = rendering.to(gt.device)
        errormap = (rendering - gt).abs()
        
        # object masks
        object_mask = view.object_mask.cuda()

        if vis_normal == True:
            normal_map = render_pkg['render_normals'][0] 
            vis_normal_map = visualize_normal(normal_map, view)
            vis_alpha_mask = ((alpha_mask * 255).byte()).permute(1, 2, 0).cpu().numpy()
            vis_normal_map = np.concatenate((vis_normal_map,vis_alpha_mask),axis=2)
            imageio.imwrite(os.path.join(normal_path, '{0:05d}'.format(idx) + ".png"), vis_normal_map)
            
        if vis_depth == True:
            depth_map = render_pkg["render_depth"]
            vis_depth_map = visualize_depth(depth_map) 
            vis_depth_map = torch.concat([vis_depth_map,alpha_mask],dim=0)
            torchvision.utils.save_image(vis_depth_map, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))

        if vis_semantic == True:
            semantic_map = render_pkg["render_semantics"]
            object_id = gaussians.id_encoder.inverse_transform(semantic_map)
            imageio.imwrite(os.path.join(semantic_path, '{0:05d}'.format(idx) + ".png"), object_id.squeeze().cpu().numpy())

        save_rgba(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        save_mask(object_mask, os.path.join(semantic_gt_path, '{0:05d}'.format(idx) + ".png"))
        save_rgba(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        visible_count_list.append(visible_count)
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
        json.dump(per_view_dict, fp, indent=True)

    # ===== ADDED: FPS & Memory Stats Output =====
    if len(t_list) > 0:
        avg_rgb_ms = np.mean(t_list) * 1000
        avg_seg_ms = np.mean(t_list_seg) * 1000
        fps_rgb = 1.0 / np.mean(t_list)
        fps_seg = 1.0 / np.mean(t_list_seg)
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        print(f"\n{'='*50}")
        print(f"Rendering Stats ({name}, {len(views)} frames):")
        print(f"  RGB Only:   {fps_rgb:.2f} FPS ({avg_rgb_ms:.2f} ms/frame)")
        print(f"  RGB + Seg:  {fps_seg:.2f} FPS ({avg_seg_ms:.2f} ms/frame)")
        print(f"  Peak Memory: {peak_mem_mb:.1f} MB")
        print(f"{'='*50}\n")
        
        render_stats = {
            "dataset": name,
            "num_frames": len(views),
            "fps_rgb": float(fps_rgb),
            "fps_rgb_seg": float(fps_seg),
            "avg_ms_rgb": float(avg_rgb_ms),
            "avg_ms_rgb_seg": float(avg_seg_ms),
            "peak_memory_mb": float(peak_mem_mb),
        }
        stats_path = os.path.join(model_path, name, f"ours_{iteration}", "render_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(render_stats, f, indent=2)
    # ============================================

    return visible_count_list

    
def render_sets(dataset, opt, pipe, iteration, skip_train, skip_test, explicit=None, logger=None):
    with torch.no_grad():
        if pipe.no_prefilter_step > 0:
            pipe.add_prefilter = False
        else:
            pipe.add_prefilter = True
        modules = __import__('scene')
        model_config = dataset.model_config
        gaussians = getattr(modules, model_config['name'])(**model_config['kwargs'])
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, explicit=explicit, logger=logger)
        gaussians.eval()

        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)
        
        if not skip_train:
            visible_count = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipe, scene.background)

        if not skip_test:
            visible_count = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipe, scene.background)

    return visible_count

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument('-m', '--model_path', type=str, required=True)
    parser.add_argument("--scene_name", default=None)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--ape", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--explicit", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    with open(os.path.join(args.model_path, "config.yaml")) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    args.scene_name = args.model_path.split('/')[-2]

    if args.scene_name is not None:
        try:
            cfg["model_params"]["exp_name"] = os.path.join(cfg["model_params"]["exp_name"], args.scene_name)
            cfg["model_params"]["source_path"] = os.path.join(cfg["model_params"]["source_path"], args.scene_name)
        except:
            print("OverrideError: Cannot override 'exp_name' and 'source_path' in 'model_params'. Exiting.")
            sys.exit(1)

    lp, op, pp = parse_cfg(cfg)
    lp.model_path = args.model_path
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(lp, op, pp, args.iteration, args.skip_train, args.skip_test, explicit=args.explicit)