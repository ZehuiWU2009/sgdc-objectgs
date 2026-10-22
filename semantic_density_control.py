# semantic_density_control.py
# ObjectGS Adapter Version
import torch
import numpy as np
from typing import Tuple, Optional
import warnings

class SemanticDensityController:
    """
    Semantic Density Controller adapted for ObjectGS (Anchor-based).
    - Semantic labels from anchor label_ids (no classifier needed)
    - Primitive gradient aggregation to anchor level
    - Instance-wise quantile split + annealing
    - Gradient-based pruning
    """
    
    def __init__(self, 
                 classifier: Optional[torch.nn.Module] = None,
                 fixed_split_threshold: float = 0.00013,
                 split_quantile: float = 0.5,
                 prune_quantile: float = 0.06,
                 prune_abs_threshold: float = 0.00006,
                 min_points_per_instance: int = 2,
                 filter_low_confidence: int = 1,
                 confidence_threshold: float = 0.7,
                 anneal_enable: int = 1,
                 anneal_start_iter: int = 5000,
                 anneal_end_iter: int = 15000,
                 anneal_target_factor: float = 0.5,
                 device: str = 'cuda',
                 ablation_semantic: int = 0,
                 ablation_instance_split: int = 1,
                 ablation_gradient_prune: int = 1,
                 bg_split_quantile: float = 0.95):
        
        self.classifier = classifier
        self.device = device
        
        self.fixed_split_threshold = fixed_split_threshold
        self.split_quantile = split_quantile
        self.prune_quantile = prune_quantile
        self.prune_abs_threshold = prune_abs_threshold
        self.min_points_per_instance = min_points_per_instance
        self.filter_low_confidence = bool(filter_low_confidence)
        self.confidence_threshold = confidence_threshold
        
        self.anneal_enable = bool(anneal_enable)
        self.anneal_start_iter = anneal_start_iter
        self.anneal_end_iter = anneal_end_iter
        self.anneal_target_factor = anneal_target_factor
        
        self.ablation_semantic = bool(ablation_semantic)
        self.ablation_instance_split = bool(ablation_instance_split)
        self.ablation_gradient_prune = bool(ablation_gradient_prune)
        self.bg_split_quantile = bg_split_quantile
        
        self.statistics_history = []
        
        print(f"[SADC] ObjectGS Adapter Initialized: fixed_split={fixed_split_threshold}, "
              f"split_q={split_quantile}, prune_q={prune_quantile}, prune_abs={prune_abs_threshold}")
        print(f"[SADC] Ablation flags: semantic={self.ablation_semantic}, "
              f"instance_split={self.ablation_instance_split}, gradient_prune={self.ablation_gradient_prune}")
        if self.anneal_enable:
            print(f"[SADC] Annealing: start={anneal_start_iter}, end={anneal_end_iter}, "
                  f"target_factor={anneal_target_factor}")
        print(f"[SADC] Background split quantile: {self.bg_split_quantile}")

    def _get_current_split_threshold(self, iteration):
        """Dynamic split threshold with annealing"""
        if not self.anneal_enable or iteration < self.anneal_start_iter:
            return self.fixed_split_threshold
        if iteration >= self.anneal_end_iter:
            return self.fixed_split_threshold * self.anneal_target_factor
        t = (iteration - self.anneal_start_iter) / (self.anneal_end_iter - self.anneal_start_iter)
        current = self.fixed_split_threshold * (1 - (1 - self.anneal_target_factor) * t)
        return current

    def compute_semantic_labels(self, gaussians, iteration: int) -> torch.Tensor:
        """
        For ObjectGS: directly use anchor label_ids instead of classifier.
        ObjectGS background label is 0, internally mapped to -1.
        """
        if not self.ablation_semantic:
            N = gaussians.get_anchor.shape[0]
            return torch.full((N,), -1, dtype=torch.long, device=self.device)
        
        labels = gaussians.label_ids.squeeze().long().to(self.device)
        labels[labels == 0] = -1
        return labels

    # ===== MODIFIED: 改用 anchor 累积梯度，不再聚合子点梯度 =====
    def _get_anchor_grad_from_primitives(self, gaussians):
        """
        Use anchor's own accumulated gradient (_anchor_grad_accum / _anchor_denom)
        instead of primitive gradient aggregation.
        """
        # 检查 buffer 是否存在且非空
        if not hasattr(gaussians, '_anchor_grad_accum') or gaussians._anchor_grad_accum is None:
            # 如果 buffer 不存在，fallback 到子点 mean（警告）
            warnings.warn("anchor_grad_accum not found, falling back to primitive mean")
            primitive_grads = gaussians.offset_gradient_accum / (gaussians.offset_denom + 1e-7)
            primitive_grads[primitive_grads.isnan()] = 0.0
            N = gaussians.get_anchor.shape[0]
            k = gaussians.n_offsets
            anchor_grads = primitive_grads.view(N, k).mean(dim=1)
            return anchor_grads.squeeze()

        # 计算平均梯度
        anchor_grads = gaussians._anchor_grad_accum / (gaussians._anchor_denom + 1e-7)
        anchor_grads = anchor_grads.nan_to_num(0.0)
        # 取 L2 范数
        anchor_grads = torch.norm(anchor_grads, dim=-1)
        return anchor_grads.squeeze()

    def compute_split_prune_masks(self,
                                 gaussians,
                                 instance_labels: torch.Tensor,
                                 iteration: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate split/prune masks for ObjectGS anchors.
        """
        N = gaussians.get_anchor.shape[0]
        device = self.device
        
        smooth_grad = self._get_anchor_grad_from_primitives(gaussians)
        current_fixed_split = self._get_current_split_threshold(iteration)
        
        split_mask = torch.zeros(N, dtype=torch.bool, device=device)
        prune_mask = torch.zeros(N, dtype=torch.bool, device=device)
        
        # ---- Split Strategy ----
        if self.ablation_instance_split:
            unique_labels = torch.unique(instance_labels)
            
            valid_mask = instance_labels != -1
            if valid_mask.sum() > 0:
                global_split_threshold = torch.quantile(smooth_grad[valid_mask], self.split_quantile).item()
            else:
                global_split_threshold = current_fixed_split
            
            bg_mask = instance_labels == -1
            bg_grads = smooth_grad[bg_mask]
            if bg_grads.numel() > 0:
                bg_split_thresh = torch.quantile(bg_grads, self.bg_split_quantile).item()
                effective_bg_split_thresh = max(bg_split_thresh, current_fixed_split)
                bg_split = bg_grads > effective_bg_split_thresh
                bg_indices = torch.where(bg_mask)[0]
                split_mask[bg_indices[bg_split]] = True
            
            for label in unique_labels:
                if label == -1:
                    continue
                mask = instance_labels == label
                num_points = mask.sum().item()
                if num_points == 0:
                    continue
                inst_grads = smooth_grad[mask]
                if num_points >= self.min_points_per_instance:
                    split_thresh = torch.quantile(inst_grads, self.split_quantile).item()
                else:
                    split_thresh = global_split_threshold
                effective_split_thresh = max(split_thresh, current_fixed_split)
                inst_split = inst_grads > effective_split_thresh
                indices = torch.where(mask)[0]
                split_mask[indices[inst_split]] = True
        else:
            split_mask = smooth_grad > current_fixed_split
        
        # ---- Prune Strategy ----
        if self.ablation_gradient_prune:
            if self.ablation_instance_split:
                unique_labels = torch.unique(instance_labels)
                valid_mask = instance_labels != -1
                if valid_mask.sum() > 0:
                    global_prune_threshold = torch.quantile(smooth_grad[valid_mask], self.prune_quantile).item()
                else:
                    global_prune_threshold = 0.0
                
                bg_mask = instance_labels == -1
                bg_grads = smooth_grad[bg_mask]
                if bg_grads.numel() > 0:
                    bg_prune_thresh = torch.quantile(bg_grads, self.prune_quantile).item()
                    bg_prune = (bg_grads < bg_prune_thresh) & (bg_grads < self.prune_abs_threshold)
                    bg_indices = torch.where(bg_mask)[0]
                    prune_mask[bg_indices[bg_prune]] = True
                
                for label in unique_labels:
                    if label == -1:
                        continue
                    mask = instance_labels == label
                    num_points = mask.sum().item()
                    if num_points == 0:
                        continue
                    inst_grads = smooth_grad[mask]
                    if num_points >= self.min_points_per_instance:
                        prune_thresh = torch.quantile(inst_grads, self.prune_quantile).item()
                    else:
                        prune_thresh = global_prune_threshold
                    inst_prune = (inst_grads < prune_thresh) & (inst_grads < self.prune_abs_threshold)
                    indices = torch.where(mask)[0]
                    prune_mask[indices[inst_prune]] = True
            else:
                prune_thresh = torch.quantile(smooth_grad, self.prune_quantile).item()
                prune_mask = (smooth_grad < prune_thresh) & (smooth_grad < self.prune_abs_threshold)
        else:
            prune_mask = torch.zeros(N, dtype=torch.bool, device=device)
        
        if iteration % 500 == 0:
            split_count = split_mask.sum().item()
            prune_count = prune_mask.sum().item()
            num_instances = len(torch.unique(instance_labels)) - (1 if (instance_labels == -1).any() else 0)
            print(f"[SADC] Iter {iteration}: split={split_count}, prune={prune_count}, "
                  f"instances={num_instances}, current_split_th={current_fixed_split:.6f}")
            self.statistics_history.append({
                'iter': iteration,
                'split': split_count,
                'prune': prune_count,
                'instances': num_instances
            })
        
        return split_mask, prune_mask
    
    def reset(self):
        self.statistics_history.clear()
        print("[SADC] Reset")