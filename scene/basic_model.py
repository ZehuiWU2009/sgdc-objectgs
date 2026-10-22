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
import torch
import numpy as np
from torch import nn
from functools import reduce
from einops import repeat
from plyfile import PlyData, PlyElement
from utils.general_utils import inverse_sigmoid
from utils.general_utils import parse_cfg
import yaml    

class BasicModel:

    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
    
    def eval(self):
        return

    def train(self):
        return

    def set_appearance(self, num_cameras):
        self.embedding_appearance = None
    
    def oneupSHdegree(self):
        if self.active_sh_degree != None and self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def smooth_complement(self, visible_mask): 
        return torch.ones((visible_mask.sum(), 1), dtype=torch.float, device="cuda")

    def set_anchor_mask(self, *args):
        self._anchor_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device="cuda")
    
    def set_gs_mask(self, *args):
        self._gs_mask = torch.ones(self._xyz.shape[0], dtype=torch.bool, device="cuda")

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def get_camera_info(train_cameras):
        return

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'embedding' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def training_statis(self, opt, render_pkg, width, height):
        offset_selection_mask = render_pkg["selection_mask"]
        anchor_visible_mask = render_pkg["visible_mask"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        update_filter = render_pkg["visibility_filter"]
        opacity = render_pkg["opacity"]
        radii = render_pkg["radii"]
        
        temp_opacity = torch.zeros(offset_selection_mask.shape[0], dtype=torch.float32, device="cuda")
        temp_opacity[offset_selection_mask] = opacity.clone().view(-1).detach()
        
        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        if opt.pruning_type=="mean":
            temp_mask = offset_selection_mask.view([-1, self.n_offsets])
            sum_of_elements = temp_opacity.sum(dim=1, keepdim=True).cuda()
            count_of_elements = temp_mask.sum(dim=1, keepdim=True).float().cuda()
            average = sum_of_elements / torch.clamp(count_of_elements, min=1.0)
            average[count_of_elements == 0] = 0
            self.anchor_opacity_accum[anchor_visible_mask] += average
        elif opt.pruning_type=="max":
            self.anchor_opacity_accum[anchor_visible_mask] = torch.max(self.anchor_opacity_accum[anchor_visible_mask], torch.abs(temp_opacity.sum(dim=1, keepdim=True)))
        else:
            raise ValueError(f"Unknown pruning_type: {opt.pruning_type}")
        
        self.anchor_demon[anchor_visible_mask] += 1

        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter
        
        grad = viewspace_point_tensor.grad.squeeze(0)
        grad[:, 0] *= width * 0.5
        grad[:, 1] *= height * 0.5
        grad_norm = torch.norm(grad[update_filter,:2], dim=-1, keepdim=True)
        if opt.growing_type=="mean":
            self.offset_gradient_accum[combined_mask] += grad_norm
        elif opt.growing_type=="max":
            self.offset_gradient_accum[combined_mask] = torch.max(self.offset_gradient_accum[combined_mask], torch.abs(grad_norm))
            self.max_radii2D[combined_mask] = torch.max(self.max_radii2D[combined_mask], radii[update_filter])
            self.offset_opacity_accum[combined_mask] += opacity.clone().detach()[update_filter]
        else:
            raise ValueError(f"Unknown growing_type: {opt.growing_type}")
        
        self.offset_denom[combined_mask] += 1
        
    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'embedding' in group['name']:
                continue
            
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            
        return optimizable_tensors

    def get_remove_duplicates(self, grid_coords, selected_grid_coords_unique, use_chunk = True):
        if use_chunk:
            chunk_size = 4096
            max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
            remove_duplicates_list = []
            for i in range(max_iters):
                cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                remove_duplicates_list.append(cur_remove_duplicates)
            remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
        else:
            remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)
        return remove_duplicates
    
    def map_to_int_level(self, pred_level, cur_level):
        if self.dist2level=='floor':
            int_level = torch.floor(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level=='round':
            int_level = torch.round(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level=='ceil':   
            int_level = torch.ceil(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level=='progressive':
            pred_level = torch.clamp(pred_level+1.0, min=0.9999, max=cur_level + 0.9999)
            int_level = torch.floor(pred_level).int()
            self._prog_ratio = torch.frac(pred_level).unsqueeze(dim=1)
            self.transition_mask = (self._level.squeeze(dim=1) == int_level)
        else:
            raise ValueError(f"Unknown dist2level: {self.dist2level}")
        
        return int_level

    # ========== MODIFIED: run_densify with SADC full control ==========
    def run_densify(self, opt, iteration, ext_split_mask=None, ext_prune_mask=None):
        # ===== ADDED: 15000轮后强制清空 SADC mask，彻底走原生 =====
        anneal_end = getattr(opt, 'anneal_end_iter', 15000)
        if iteration > anneal_end:
            ext_split_mask = None
            ext_prune_mask = None
        # =========================================================
        
        if opt.growing_type=="mean":
            grads = self.offset_gradient_accum / self.offset_denom
            grads[grads.isnan()] = 0.0
            grads_norm = torch.norm(grads, dim=-1)
            offset_mask = (self.offset_denom > opt.update_interval * opt.success_threshold * 0.5).squeeze(dim=1)
        elif opt.growing_type=="max":
            grads = self.offset_gradient_accum
            grads[grads.isnan()] = 0.0
            
            opacities = self.offset_opacity_accum/self.offset_denom
            opacities[opacities.isnan()] = 0.0
            opacities = opacities.flatten()

            grads_norm = torch.norm(grads, dim=-1) * self.max_radii2D * torch.pow(opacities, 1/5.0)
            offset_mask = (self.offset_denom > opt.update_interval * opt.success_threshold * 0.5).squeeze(dim=1)
            offset_mask = torch.logical_and(offset_mask, opacities > 0.15)
        else:
            raise ValueError(f"Unknown growing_type: {opt.growing_type}")

        # ========== FIX 问题1: SADC split 直接信任 SADC，移除 offset_denom 过滤 ==========
        if ext_split_mask is not None:
            ext_split_mask_prim = ext_split_mask.unsqueeze(1).repeat(1, self.n_offsets).view(-1)
            # SADC 模式下：完全信任 SADC 的 split 决策
            # SADC 在 semantic_density_control.py 中已通过梯度阈值过滤低质量点
            offset_mask = ext_split_mask_prim
            skip_grad = True
            # FIX 1: SADC 模式下，重置所有 SADC 选中的 primitive 的统计量
            active_mask = ext_split_mask_prim.clone()
        else:
            skip_grad = False
            # 原生模式：保持原逻辑
            active_mask = offset_mask.clone()
        # ================================================================================

        self.anchor_growing(grads_norm, opt, offset_mask, iteration, skip_grad_check=skip_grad)
        
        # ===== FIX 1: 用 active_mask 全量重置统计量 =====
        self.offset_denom[active_mask] = 0
        # =================================================
        # ===== FIX 2: padding dtype 改为目标 tensor 的 dtype =====
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=self.offset_denom.dtype, 
                                           device=self.offset_denom.device)
        # =====================================================
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        # ===== FIX 1: 用 active_mask 全量重置 =====
        self.offset_gradient_accum[active_mask] = 0
        # =========================================
        # ===== FIX 2: padding dtype 改为目标 tensor 的 dtype =====
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=self.offset_gradient_accum.dtype, 
                                           device=self.offset_gradient_accum.device)
        # =====================================================
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)
        
        # ===== FIX 1: 用 active_mask 全量重置 =====
        self.offset_opacity_accum[active_mask] = 0
        # =========================================
        # ===== FIX 2: padding dtype 改为目标 tensor 的 dtype =====
        padding_offset_opacity_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_opacity_accum.shape[0], 1],
                                           dtype=self.offset_opacity_accum.dtype, 
                                           device=self.offset_opacity_accum.device)
        # =====================================================
        self.offset_opacity_accum = torch.cat([self.offset_opacity_accum, padding_offset_opacity_accum], dim=0)
        
        # align ext_prune_mask after growing
        if ext_prune_mask is not None:
            current_N = self.get_anchor.shape[0]
            if ext_prune_mask.shape[0] < current_N:
                padded = torch.zeros(current_N, dtype=torch.bool, device=ext_prune_mask.device)
                padded[:ext_prune_mask.shape[0]] = ext_prune_mask
                ext_prune_mask = padded
            elif ext_prune_mask.shape[0] > current_N:
                ext_prune_mask = ext_prune_mask[:current_N]

        # ========== FIX 问题2: SADC prune 直接信任 SADC，移除 anchor_demon 过滤 ==========
        if ext_prune_mask is not None:
            # SADC 模式下：完全信任 SADC 的 prune 决策
            # 原生 opacity 兜底仍然保留（低 opacity 点始终可被剪）
            opacity_prune = (self.anchor_opacity_accum < opt.min_opacity * self.anchor_demon).squeeze(dim=1)
            prune_mask = torch.logical_or(ext_prune_mask, opacity_prune)
            # 不再用 anchors_mask 限制 SADC 的剪枝决策
            anchors_mask = torch.ones(self.get_anchor.shape[0], dtype=torch.bool, device='cuda')
        else:
            # 原生模式：恢复原有逻辑
            anchors_mask = (self.anchor_demon > opt.update_interval * opt.success_threshold).squeeze(dim=1)
            if opt.pruning_type=="mean":
                prune_mask = (self.anchor_opacity_accum < opt.min_opacity*self.anchor_demon).squeeze(dim=1)
            elif opt.pruning_type=="max":
                prune_mask = (self.anchor_opacity_accum < opt.min_opacity).squeeze(dim=1)
            else:
                raise ValueError(f"Unknown pruning_type: {opt.pruning_type}")
            prune_mask = torch.logical_and(prune_mask, anchors_mask)
        # ===================================================================================

        prune_mask = self.prune_anchor(prune_mask)
                
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum
        
        offset_opacity_accum = self.offset_opacity_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_opacity_accum = offset_opacity_accum.view([-1, 1])
        del self.offset_opacity_accum
        self.offset_opacity_accum = offset_opacity_accum
        
        if anchors_mask.sum()>0:
            self.anchor_opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
        
        temp_opacity_accum = self.anchor_opacity_accum[~prune_mask]
        del self.anchor_opacity_accum
        self.anchor_opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon
        
        self.max_radii2D = torch.zeros(self.get_anchor.shape[0]*self.n_offsets, dtype=torch.float, device="cuda")
    # =================================================================

    def generate_neural_gaussians(self, viewpoint_camera, visible_mask=None, training=True):
        if visible_mask is None:
            visible_mask = torch.ones(self.get_anchor.shape[0], dtype=torch.bool, device = self.get_anchor.device)

        anchor = self.get_anchor[visible_mask]
        feat = self.get_anchor_feat[visible_mask]
        grid_offsets = self.get_offset[visible_mask]
        grid_scaling = self.get_scaling[visible_mask]
        semantics = self.get_semantic[visible_mask]

        ob_view = anchor - viewpoint_camera.camera_center
        ob_dist = ob_view.norm(dim=1, keepdim=True)
        ob_view = ob_view / ob_dist
        
        if self.view_dim > 0:
            cat_local_view = torch.cat([feat, ob_view], dim=1)
        else:
            cat_local_view = feat

        if self.appearance_dim > 0:
            if training:
                camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
                appearance = self.get_appearance(camera_indicies)
            else:
                camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * 0
                appearance = self.get_appearance(camera_indicies)
                
        neural_opacity = self.get_opacity_mlp(cat_local_view) * self.smooth_complement(visible_mask)
        neural_opacity = neural_opacity.reshape([-1, 1])
        mask = (neural_opacity>0.0)
        mask = mask.view(-1)

        opacity = neural_opacity[mask]

        if self.appearance_dim > 0:
            color = self.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = self.get_color_mlp(cat_local_view)

        color = color.reshape([anchor.shape[0]*self.n_offsets, self.color_dim])

        object_num = semantics.shape[-1]

        scale_rot = self.get_cov_mlp(cat_local_view)
        scale_rot = scale_rot.reshape([anchor.shape[0]*self.n_offsets, 7])
        
        offsets = grid_offsets.view([-1, 3])
        
        concatenated = torch.cat([grid_scaling, anchor, semantics], dim=-1)
        concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=self.n_offsets)
        concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
        _mask_idx = torch.nonzero(mask, as_tuple=True)[0]
        
        # ===== ADDED: 防御性边界检查，防止高斯数膨胀时的索引越界 =====
        if _mask_idx.numel() > 0:
            valid_mask = _mask_idx < concatenated_all.shape[0]
            if not valid_mask.all():
                num_invalid = (~valid_mask).sum().item()
                print(f"[WARN] generate_neural_gaussians: {num_invalid}/{_mask_idx.numel()} indices out of bounds "
                      f"(max_idx={_mask_idx.max().item()}, tensor_len={concatenated_all.shape[0]}), clamping")
                _mask_idx = _mask_idx[valid_mask]
        # =============================================================
        
        masked = concatenated_all[_mask_idx]
        scaling_repeat, repeat_anchor, semantics, color, scale_rot, offsets = masked.split([6, 3, object_num, self.color_dim, 7, 3], dim=-1)
        
        scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3])
        rot = self.rotation_activation(scale_rot[:,3:7])
        
        offsets = offsets * scaling_repeat[:,:3]
        xyz = repeat_anchor + offsets 

        if self.color_attr != "RGB": 
            color = color.reshape([color.shape[0], self.color_dim // 3, 3])

        return xyz, offsets, color, opacity, scaling, rot, self.active_sh_degree, mask, semantics

    def generate_explicit_gaussians(self, visible_mask=None):
        if visible_mask is None:
            visible_mask = torch.ones(self._xyz.shape[0], dtype=torch.bool, device = self._xyz.device)

        xyz = self._xyz[visible_mask]
        color = torch.cat((self._features_dc, self._features_rest), dim=1)[visible_mask]
        opacity = self._opacity[visible_mask]
        scaling = self._scaling[visible_mask]
        rot = self._rotation[visible_mask]
        mask = torch.ones(self._xyz.shape[0], dtype=torch.bool, device="cuda")
        return xyz, color, opacity, scaling, rot, self.active_sh_degree, mask

    def load_config(self, model_path):
        config_path = os.path.join(os.path.dirname(os.path.dirname(model_path)), "config.yaml")
        with open(config_path) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            lp, op, pp = parse_cfg(cfg)
        self.pretrained_config = lp.model_config

    def save_mlp_checkpoints(self, path):
        return

    def load_mlp_checkpoints(self, path):
        return

    def clean(self):
        del self.offset_opacity_accum
        del self.anchor_opacity_accum
        del self.anchor_demon
        del self.offset_gradient_accum
        del self.offset_denom
        del self.max_radii2D
        torch.cuda.empty_cache()