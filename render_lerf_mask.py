import os
# ⚠️ 必须在所有 torch import 之前设置，且仅使用 max_split_size_mb
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import sys
import yaml
import torch
import gc
import time
import json
import numpy as np
from os import makedirs
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser

from scene import Scene
from utils.general_utils import safe_state, parse_cfg
from utils.image_utils import save_rgba
from grounded_sam import grouned_sam_output, load_model_hf, select_obj_ioa
from segment_anything import sam_model_registry, SamPredictor


def render_lerf_set(model_path, name, iteration, views, gaussians, pipe, background,
                    groundingdino_model, sam_predictor, positives, threshold=0.2):
    """LERF-Mask rendering with strict memory isolation for 16GB GPUs"""
    
    pred_obj_path = os.path.join(model_path, name, f"ours_{iteration}_text", "test_mask")
    render_path = os.path.join(model_path, name, f"ours_{iteration}_text", "renders")
    makedirs(pred_obj_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)

    modules = __import__('gaussian_renderer')
    render_fn = getattr(modules, 'render')

    # === Phase 1: First Frame Anchoring (GS on CPU, Text Models on GPU) ===
    print("🔍 Phase 1: First frame anchoring...")
    
    # 先将 GS 移到 CPU 并强制释放 GPU 缓存
    print("  💾 Moving GS to CPU...")
    if hasattr(gaussians, 'to'):
        gaussians.to('cpu')
    else:
        for attr in dir(gaussians):
            param = getattr(gaussians, attr, None)
            if isinstance(param, torch.nn.Parameter) and param.is_cuda:
                setattr(gaussians, attr, torch.nn.Parameter(param.cpu(), requires_grad=param.requires_grad))
    
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # 渲染首帧（GS 需要临时在 GPU 上渲染）
    print("  🎨 Rendering first frame on GPU...")
    if hasattr(gaussians, 'to'):
        gaussians.to('cuda')
    else:
        for attr in dir(gaussians):
            param = getattr(gaussians, attr, None)
            if isinstance(param, torch.nn.Parameter) and not param.is_cuda:
                setattr(gaussians, attr, torch.nn.Parameter(param.cuda(), requires_grad=param.requires_grad))
    
    torch.cuda.empty_cache()
    first_view = views[0]
    with torch.no_grad():
        first_pkg = render_fn(first_view, gaussians, pipe, background)
    
    first_render = torch.clamp(first_pkg["render"], 0.0, 1.0)
    semantic_map = first_pkg["render_semantics"]
    object_id_map = gaussians.id_encoder.inverse_transform(semantic_map)
    if object_id_map.dim() == 3:
        object_id_map = object_id_map.squeeze(0)
    
    image_np = (first_render.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    object_id_map_cpu = object_id_map.cpu()
    
    # 渲染完首帧后立即将 GS 移回 CPU 并彻底清理
    del first_pkg, first_render, semantic_map, object_id_map
    if hasattr(gaussians, 'to'):
        gaussians.to('cpu')
    else:
        for attr in dir(gaussians):
            param = getattr(gaussians, attr, None)
            if isinstance(param, torch.nn.Parameter) and param.is_cuda:
                setattr(gaussians, attr, torch.nn.Parameter(param.cpu(), requires_grad=param.requires_grad))
    
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"  ✅ First frame rendered. GPU free: {torch.cuda.mem_get_info()[0]/1024**3:.2f} GiB")

    # ===== MODIFIED: Anchor each text prompt with IoA scoring & mutual exclusion =====
    all_candidates = {}  # prompt -> [(id, ioa), ...]
    vis_dir = os.path.join(pred_obj_path, "_grounded_sam_vis")
    makedirs(vis_dir, exist_ok=True)
    
    for TEXT_PROMPT in positives:
        text_mask, annotated_frame = grouned_sam_output(
            groundingdino_model, sam_predictor, TEXT_PROMPT, image_np
        )
        selected_ids = select_obj_ioa(object_id_map_cpu, text_mask)
        
        # 计算每个选中ID的IoA分数
        scores = []
        if isinstance(text_mask, np.ndarray):
            text_mask_t = torch.from_numpy(text_mask).to(object_id_map_cpu.device)
        else:
            text_mask_t = text_mask.to(object_id_map_cpu.device)
        
        for sid in selected_ids:
            inst_mask = (object_id_map_cpu == sid.to(object_id_map_cpu.device))
            ioa = (inst_mask & text_mask_t).sum().item() / (text_mask_t.sum().item() + 1e-6)
            scores.append((sid.item(), ioa))
        
        all_candidates[TEXT_PROMPT] = scores
        print(f"  📌 [{TEXT_PROMPT}] → Object IDs: {selected_ids}, IoAs: {[f'{ioa:.3f}' for _, ioa in scores]}")
        Image.fromarray(annotated_frame).save(os.path.join(vis_dir, f"{TEXT_PROMPT}.png"))

    # 互斥分配：每个ID只给IoA最高的prompt
    id_best_prompt = {}
    for prompt, candidates in all_candidates.items():
        for sid, ioa in candidates:
            if sid not in id_best_prompt or ioa > id_best_prompt[sid][1]:
                id_best_prompt[sid] = (prompt, ioa)

    all_selected_ids = {prompt: [] for prompt in positives}
    for sid, (prompt, ioa) in id_best_prompt.items():
        all_selected_ids[prompt].append(torch.tensor(sid, dtype=torch.uint8, device='cuda'))

    for prompt in positives:
        print(f"  ✅ [{prompt}] Final IDs: {all_selected_ids[prompt]}")
    # =================================================================================

    # === Phase 2: Full Rendering (Text Models Unloaded, GS on GPU) ===
    print("\n🎨 Phase 2: Full view rendering...")
    
    # 彻底卸载文本模型
    del groundingdino_model, sam_predictor, image_np, object_id_map_cpu
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"  ✅ Text models unloaded. GPU free: {torch.cuda.mem_get_info()[0]/1024**3:.2f} GiB")

    # 将 GS 移回 GPU
    if hasattr(gaussians, 'to'):
        gaussians.to('cuda')
    else:
        for attr in dir(gaussians):
            param = getattr(gaussians, attr, None)
            if isinstance(param, torch.nn.Parameter) and not param.is_cuda:
                setattr(gaussians, attr, torch.nn.Parameter(param.cuda(), requires_grad=param.requires_grad))
    
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # ===== ADDED: FPS & Memory Stats =====
    torch.cuda.reset_peak_memory_stats()
    t_list = []
    t_list_seg = []
    # =====================================

    # Render all views
    for idx, view in enumerate(tqdm(views, desc="LERF-Mask Rendering")):
        if idx > 0:
            torch.cuda.empty_cache()
        
        with torch.no_grad():
            # ---- RGB Only ----
            torch.cuda.synchronize()
            t_start = time.time()
            render_pkg = render_fn(view, gaussians, pipe, background)
            torch.cuda.synchronize()
            t_rgb_end = time.time()
            
            rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
            
            # ---- RGB + Seg ----
            semantic_map = render_pkg["render_semantics"]
            object_id_map = gaussians.id_encoder.inverse_transform(semantic_map)
            if object_id_map.dim() == 3:
                object_id_map = object_id_map.squeeze(0)
            torch.cuda.synchronize()
            t_seg_end = time.time()

        t_list.append(t_rgb_end - t_start)
        t_list_seg.append(t_seg_end - t_start)

        save_rgba(rendering, os.path.join(render_path, f'{idx:05d}.png'))

        view_pred_dir = os.path.join(pred_obj_path, str(idx))
        makedirs(view_pred_dir, exist_ok=True)

        for TEXT_PROMPT in positives:
            selected_ids = all_selected_ids[TEXT_PROMPT]
            if len(selected_ids) > 0:
                id_tensor = object_id_map
                mask = torch.zeros_like(id_tensor, dtype=torch.bool)
                for sid in selected_ids:
                    mask = mask | (id_tensor == sid.to(id_tensor.device))
                pred_mask = (mask.cpu().numpy() * 255).astype(np.uint8)
            else:
                H, W = rendering.shape[1], rendering.shape[2]
                pred_mask = np.zeros((H, W), dtype=np.uint8)
            Image.fromarray(pred_mask).save(os.path.join(view_pred_dir, f"{TEXT_PROMPT}.png"))
        
        del render_pkg, rendering, semantic_map, object_id_map
        torch.cuda.empty_cache()
    
    # ===== ADDED: FPS & Memory Stats Output =====
    if len(t_list) > 0:
        avg_rgb_ms = np.mean(t_list) * 1000
        avg_seg_ms = np.mean(t_list_seg) * 1000
        fps_rgb = 1.0 / np.mean(t_list)
        fps_seg = 1.0 / np.mean(t_list_seg)
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        print(f"\n{'='*50}")
        print(f"LERF-Mask Rendering Stats ({name}, {len(views)} frames):")
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
        stats_path = os.path.join(model_path, name, f"ours_{iteration}_text", "render_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(render_stats, f, indent=2)
    # ============================================


def render_sets(dataset, opt, pipe, iteration, skip_train, skip_test, explicit=None, logger=None):
    with torch.no_grad():
        # ========== 新增：自动修复 source_path ==========
        scene_name = os.path.basename(os.path.dirname(dataset.model_path))
        if not os.path.exists(os.path.join(dataset.source_path, "sparse")):
            dataset.source_path = os.path.join(dataset.source_path, scene_name)
            print(f"[Info] Auto-fixed source_path: {dataset.source_path}")
        # ================================================
        
        if pipe.no_prefilter_step > 0:
            pipe.add_prefilter = False
        else:
            pipe.add_prefilter = True
            
        modules = __import__('scene')
        model_config = dataset.model_config
        gaussians = getattr(modules, model_config['name'])(**model_config['kwargs'])
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, explicit=explicit, logger=logger)
        gaussians.eval()

        # Load GroundingDINO + SAM
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filename = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
        groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename)

        sam_checkpoint = 'Tracking-Anything-with-DEVA/saves/sam_vit_h_4b8939.pth'
        if not os.path.exists(sam_checkpoint):
            raise FileNotFoundError(f"SAM checkpoint not found: {sam_checkpoint}")
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        sam.to(device='cuda')
        sam_predictor = SamPredictor(sam)

        # LERF-Mask Text Prompts
        scene_name = dataset.model_path.split('/')[-2].lower()
        if 'figurines' in scene_name:
            positives = "green apple;green toy chair;old camera;porcelain hand;red apple;red toy chair;rubber duck with red hat".split(";")
        elif 'ramen' in scene_name:
            positives = "chopsticks;egg;glass of water;pork belly;wavy noodles in bowl;yellow bowl".split(";")
        elif 'teatime' in scene_name:
            positives = "apple;bag of cookies;coffee mug;cookies on a plate;paper napkin;plate;sheep;spoon handle;stuffed bear;tea in a glass".split(";")
        else:
            raise NotImplementedError(f"Unknown LERF-Mask scene: {scene_name}")
        
        print(f"🎯 LERF-Mask prompts ({len(positives)}): {positives}")

        if not skip_train:
            render_lerf_set(dataset.model_path, "train", scene.loaded_iter,
                           scene.getTrainCameras(), gaussians, pipe, scene.background,
                           groundingdino_model, sam_predictor, positives)
        if not skip_test:
            render_lerf_set(dataset.model_path, "test", scene.loaded_iter,
                           scene.getTestCameras(), gaussians, pipe, scene.background,
                           groundingdino_model, sam_predictor, positives)


if __name__ == "__main__":
    parser = ArgumentParser(description="LERF-Mask OVS Evaluation for ObjectGS")
    parser.add_argument('-m', '--model_path', type=str, required=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--explicit", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    with open(os.path.join(args.model_path, "config.yaml")) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    scene_name = args.model_path.rstrip('/').split('/')[-1]
    try:
        cfg["model_params"]["exp_name"] = os.path.join(cfg["model_params"]["exp_name"], scene_name)
    except:
        pass

    lp, op, pp = parse_cfg(cfg)
    lp.model_path = args.model_path
    print("Rendering " + args.model_path)

    safe_state(args.quiet)
    render_sets(lp, op, pp, args.iteration, args.skip_train, args.skip_test, explicit=args.explicit)