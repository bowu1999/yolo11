import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .iou import ciou_loss
from .assigner import tal_assign
from ..models.utils import make_grid, decode_dfl, bbox_iou, bbox_area


def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction='mean'):
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = prob * targets + (1 - prob) * (1 - targets)
    mod = (1.0 - p_t) ** gamma
    alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
    loss = alpha_t * mod * ce
    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss.mean()


def softmax_focal_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'mean'
):
    # logits: (P, C), labels: (P,)
    logp = F.log_softmax(logits, dim=-1)  # (P, C)
    p = logp.exp()
    idx = labels.long().unsqueeze(1)
    pt = p.gather(1, idx).squeeze(1)      # (P,)
    log_pt = logp.gather(1, idx).squeeze(1)  # (P,)
    mod = (1.0 - pt) ** gamma
    loss = - alpha * mod * log_pt
    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss.mean()


class YOLOv8Loss(nn.Module):
    def __init__(
        self,
        strides: List[int] = [8, 16, 32],
        num_classes: int = 80,
        dfl_bins: int = 16,
        loss_cls_weight: float = 1.,
        loss_iou_weight: float = 7.5,
        loss_dfl_weight: float = 1.5,
        tal_topk: int = 10,
        use_focal: bool = True,          # focal for BCE path
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        debug: bool = False,
        # hard-neg 采样和正负样本权重
        neg_pos_ratio: int = 3,
        neg_thresh: float = 0.10,   # 从 0.05 调到 0.10 更保守
        pos_weight: float = 1.0,    # 先设为 1.0，避免过放大
        neg_weight: float = 1.0,    # 先设为 1.0
        pos_loss_type: str = 'bce',   # 'bce' (default) or 'ce' (softmax CE focal)
        use_neg_loss: bool = True,
        neg_selection: str = 'per_image',  # 'per_image' or 'global'
        label_smoothing: float = 0.1,
        loss_obj_weight: float = 3.0,
        max_pos_per_class: int = 128
    ):
        super().__init__()
        assert pos_loss_type in ('bce', 'ce')
        assert neg_selection in ('per_image', 'global')
        self.strides = strides
        self.num_classes = num_classes
        self.dfl_bins = dfl_bins
        self.loss_cls_weight = loss_cls_weight
        self.loss_iou_weight = loss_iou_weight
        self.loss_dfl_weight = loss_dfl_weight
        self.tal_topk = tal_topk
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.debug = debug
        self.neg_pos_ratio = neg_pos_ratio
        self.neg_thresh = neg_thresh
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.pos_loss_type = pos_loss_type
        self.use_neg_loss = use_neg_loss
        self.neg_selection = neg_selection
        self.label_smoothing = label_smoothing
        self.loss_obj_weight = loss_obj_weight
        self.max_pos_per_class = max_pos_per_class

    def _make_dfl_targets(
            self,
            gt_boxes: torch.Tensor,
            matched_gt_inds: torch.Tensor,
            grid: torch.Tensor,
            stride_map: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        B, N = matched_gt_inds.shape
        device = matched_gt_inds.device
        bins = self.dfl_bins
        target_dist = torch.zeros((B, N, 4, bins), device=device)
        if stride_map is None:
            stride_map = torch.ones((N,), device=device, dtype=grid.dtype)
        else:
            assert stride_map.shape[0] == N
        for b in range(B):
            matched = matched_gt_inds[b]
            pos_idx = (matched >= 0).nonzero(as_tuple=False).squeeze(1)
            if pos_idx.numel() == 0:
                continue
            assigned = matched[pos_idx]
            boxes = gt_boxes[b][assigned]
            # grid indexing safety
            cx = grid[:, 0][pos_idx]
            cy = grid[:, 1][pos_idx]
            l = (cx - boxes[:, 0]).clamp(min=0)
            t = (cy - boxes[:, 1]).clamp(min=0)
            r = (boxes[:, 2] - cx).clamp(min=0)
            b_ = (boxes[:, 3] - cy).clamp(min=0)
            dists = torch.stack([l, t, r, b_], dim=1)
            s = stride_map[pos_idx].unsqueeze(1)
            t_bins = dists / s
            t_bins = t_bins.clamp(0, bins - 1 - 1e-6)
            lower = t_bins.floor().long()
            upper = lower + 1
            upper = upper.clamp(max=bins - 1)
            alpha = (t_bins - lower.float())
            P_ = pos_idx.numel()
            for i_coord in range(4):
                li = lower[:, i_coord]
                ui = upper[:, i_coord]
                a = alpha[:, i_coord]
                temp = torch.zeros((P_, bins), device=device, dtype=target_dist.dtype)
                temp.scatter_add_(1, li.unsqueeze(1), (1 - a).unsqueeze(1))
                temp.scatter_add_(1, ui.unsqueeze(1), a.unsqueeze(1))
                target_dist[b, pos_idx, i_coord] += temp
        return target_dist

    def forward(self, preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch: dict) -> dict:
        device = preds[0].device
        B = preds[0].shape[0]

        per_scale_scores = []
        per_scale_dfl = []
        per_scale_bboxes = []
        Ns = []
        grids = []
        # decode and reshape per-scale outputs
        for i, p in enumerate(preds):
            stride = self.strides[i]
            Bp, Cc, H, W = p.shape
            N = H * W
            Ns.append(N)
            dfl, cls_logits = torch.split(p, [4 * self.dfl_bins, self.num_classes], dim=1)
            per_bboxes, decode_ok = decode_dfl(dfl, stride=stride)
            per_scale_bboxes.append(per_bboxes)               # (B, N, 4)
            cls_logits = cls_logits.permute(0, 2, 3, 1).view(Bp, N, self.num_classes)
            per_scale_scores.append(cls_logits)               # logits
            dfl = dfl.permute(0, 2, 3, 1).view(Bp, N, 4, self.dfl_bins)
            per_scale_dfl.append(dfl)
            grid = make_grid((H, W), stride, device)         # (H*W, 2) pixel-centered grid
            grid = grid.view(N, 2)
            grids.append(grid)

        all_scores = torch.cat(per_scale_scores, dim=1)     # (B, N_total, C) logits
        all_dfl = torch.cat(per_scale_dfl, dim=1)          # (B, N_total, 4, bins)
        pred_bboxes = torch.cat(per_scale_bboxes, dim=1)   # (B, N_total, 4)
        all_grids = torch.cat(grids, dim=0)                # (N_total,2)
        # stride map
        stride_list = []
        for i, N in enumerate(Ns):
            stride_list.append(torch.full((N,), fill_value=self.strides[i], device=device, dtype=all_grids.dtype))
        stride_map = torch.cat(stride_list, dim=0)  # (N_total,)

        gt_bboxes = batch['gt_bboxes']
        gt_labels = batch['gt_labels']

        # pass logits to tal_assign (tal_assign will sigmoided internally)
        target_scores, target_bboxes, fg_mask, matched_gt_inds = tal_assign(
            all_scores, pred_bboxes, gt_bboxes, gt_labels, topk=self.tal_topk)

        # ------------------ sanitize matched_gt_inds & gt_labels ------------------
        # Purpose: 防止 matched 指向越界的 gt 索引或指向被标为 -1 的 padding label
        for b in range(B):
            # get number of gt for this image
            num_gt_b = len(batch['gt_bboxes'][b])
            if num_gt_b == 0:
                # no gt in this image: mark all matched as -1
                matched_gt_inds[b].fill_(-1)
                continue

            # clamp any matched indexes that are out-of-range
            m = matched_gt_inds[b]
            # ensure long dtype for indexing
            if m.dtype != torch.long:
                try:
                    m = m.long()
                except Exception:
                    m = m.clone().to(torch.long)
                matched_gt_inds[b] = m

            # any index >= num_gt_b or < -1 -> mark as -1
            invalid_mask = (m >= num_gt_b) | (m < -1)
            if invalid_mask.any():
                m[invalid_mask] = -1

            # Now if gt_labels contains padding -1 entries, remove assignments to those gts
            gt_labels_b = batch['gt_labels'][b].long().to(device)
            if (gt_labels_b == -1).any():
                pos_idx = (m >= 0).nonzero(as_tuple=False).squeeze(1)
                if pos_idx.numel() > 0:
                    assigned = m[pos_idx].long()  # indices into gt_labels_b
                    valid_assigned_mask = (assigned >= 0) & (assigned < num_gt_b)
                    if valid_assigned_mask.any():
                        good_pos_idx = pos_idx[valid_assigned_mask]
                        good_assigned = assigned[valid_assigned_mask]
                        bad_label_mask = (gt_labels_b[good_assigned] == -1)
                        if bad_label_mask.any():
                            bad_positions = good_pos_idx[bad_label_mask]
                            m[bad_positions] = -1

            # write back
            matched_gt_inds[b] = m

        # recompute pos_mask from sanitized matched_gt_inds
        pos_mask = (matched_gt_inds >= 0)
        num_pos = int(pos_mask.sum().item())
        # -------------------------------------------------------------------------

        B, N_total, _ = all_scores.shape
        # ------------------------- CLASS LOSS -------------------------
        probs = torch.sigmoid(all_scores)  # (B,N,C)
        if self.use_focal:
            loss_all_elem = sigmoid_focal_loss(all_scores, target_scores, alpha=self.focal_alpha, gamma=self.focal_gamma, reduction='none')  # (B,N,C)
        else:
            loss_all_elem = F.binary_cross_entropy_with_logits(all_scores, target_scores, reduction='none')

        # per-anchor scalar loss for hardness scoring
        loss_per_anchor = loss_all_elem.sum(dim=-1)  # (B,N)

        # ---------------------------- NEGATIVE SELECTION (保留/复用你原有逻辑) ----------------------------
        selected_neg_mask = torch.zeros_like(pos_mask, dtype=torch.bool, device=device)  # (B,N)
        def select_topk_from_candidates(candidate_idx: torch.Tensor, hardness_scores: torch.Tensor, k: int):
            if k <= 0 or candidate_idx.numel() == 0:
                return torch.empty((0,), dtype=torch.long, device=device)
            if candidate_idx.numel() <= k:
                return candidate_idx
            topk_vals, topk_idx = torch.topk(hardness_scores, k=min(k, hardness_scores.numel()))
            return candidate_idx[topk_idx]

        if self.neg_selection == 'per_image':
            for b in range(B):
                pos_idx_b = pos_mask[b].nonzero(as_tuple=False).squeeze(1)
                num_pos_b = int(pos_idx_b.numel())
                if num_pos_b > 0:
                    num_neg_to_keep = min(int(num_pos_b * self.neg_pos_ratio), int((~pos_mask[b]).sum().item()))
                else:
                    num_neg_to_keep = min(64, int((~pos_mask[b]).sum().item()))
                if num_neg_to_keep <= 0:
                    continue
                neg_candidates = (~pos_mask[b]).nonzero(as_tuple=False).squeeze(1)
                if neg_candidates.numel() == 0:
                    continue
                pred_max_per_anchor = probs[b].max(dim=-1).values  # (N,)
                cand_scores = pred_max_per_anchor[neg_candidates]   # (M,)
                cand_loss = loss_per_anchor[b, neg_candidates]      # (M,)
                cand_loss_mean = cand_loss.mean().clamp_min(1e-6)
                cand_loss_norm = cand_loss / cand_loss_mean
                mask_high = cand_scores >= self.neg_thresh
                if mask_high.sum().item() > 0:
                    candidates_high = neg_candidates[mask_high]
                    hardness = (cand_loss_norm[mask_high]) * cand_scores[mask_high]
                    k = min(num_neg_to_keep, candidates_high.numel())
                    selected = select_topk_from_candidates(candidates_high, hardness, k)
                else:
                    hardness = cand_loss_norm * cand_scores
                    selected = select_topk_from_candidates(neg_candidates, hardness, num_neg_to_keep)
                selected_neg_mask[b, selected] = True
        else:
            all_candidates = []
            all_hardness = []
            for b in range(B):
                neg_candidates = (~pos_mask[b]).nonzero(as_tuple=False).squeeze(1)
                if neg_candidates.numel() == 0:
                    continue
                pred_max_per_anchor = probs[b].max(dim=-1).values
                cand_scores = pred_max_per_anchor[neg_candidates]
                cand_loss = loss_per_anchor[b, neg_candidates]
                hardness = (cand_scores * cand_loss).detach()
                all_candidates.append(torch.stack([torch.full_like(neg_candidates, b, dtype=torch.long, device=device), neg_candidates], dim=1))
                all_hardness.append(hardness)
            if len(all_candidates) > 0:
                all_cand = torch.cat(all_candidates, dim=0)  # (M,2) [b, idx]
                all_hardness_t = torch.cat(all_hardness, dim=0)  # (M,)
                tot_pos = pos_mask.sum().item()
                tot_neg_to_keep = min(int(tot_pos * self.neg_pos_ratio), all_cand.shape[0]) if tot_pos > 0 else min(256, all_cand.shape[0])
                if tot_neg_to_keep > 0:
                    topk_vals, topk_idx = torch.topk(all_hardness_t, k=tot_neg_to_keep)
                    chosen = all_cand[topk_idx]  # (K,2)
                    for row in chosen:
                        b_idx = int(row[0].item()); n_idx = int(row[1].item())
                        selected_neg_mask[b_idx, n_idx] = True

        # ---------------------------- OBJECTNESS SUPERVISION（改为加权 BCE） ----------------------------
        # positives => use assigned class logit as pos-objectness logit (target 1)
        pos_logits_list = []
        pos_obj_targets = []
        for b in range(B):
            matched = matched_gt_inds[b]  # (N,)
            pos_idx = (matched >= 0).nonzero(as_tuple=False).squeeze(1)
            if pos_idx.numel() == 0:
                continue
            assigned = matched[pos_idx].long()
            gt_labels_b = batch['gt_labels'][b].long().to(device)
            logits_b = all_scores[b, pos_idx]  # (P_b, C)
            assigned_labels = gt_labels_b[assigned]  # (P_b,)
            pos_class_logit = logits_b.gather(1, assigned_labels.unsqueeze(1)).squeeze(1)  # (P_b,)
            pos_logits_list.append(pos_class_logit)
            pos_obj_targets.append(torch.ones_like(pos_class_logit, device=device))

        if len(pos_logits_list) > 0:
            obj_pos_logits = torch.cat(pos_logits_list, dim=0)
            obj_pos_targets = torch.cat(pos_obj_targets, dim=0)
        else:
            obj_pos_logits = torch.empty((0,), device=device)
            obj_pos_targets = torch.empty((0,), device=device)

        # negatives => use each anchor's max-class-logit as neg-objectness logit (target 0)
        neg_logits_list = []
        neg_obj_targets = []
        if self.use_neg_loss:
            for b in range(B):
                neg_idx = (selected_neg_mask[b]).nonzero(as_tuple=False).squeeze(1)
                if neg_idx.numel() == 0:
                    continue
                logits_b = all_scores[b, neg_idx]  # (K, C)
                max_logits, _ = logits_b.max(dim=-1)  # (K,)
                neg_logits_list.append(max_logits)
                neg_obj_targets.append(torch.zeros_like(max_logits, device=device))
            if len(neg_logits_list) > 0:
                obj_neg_logits = torch.cat(neg_logits_list, dim=0)
                obj_neg_targets = torch.cat(neg_obj_targets, dim=0)
            else:
                obj_neg_logits = torch.empty((0,), device=device)
                obj_neg_targets = torch.empty((0,), device=device)
        else:
            obj_neg_logits = torch.empty((0,), device=device)
            obj_neg_targets = torch.empty((0,), device=device)

        # combine objectness samples and compute weighted BCE (separate pos/neg mean)
        if obj_pos_logits.numel() + obj_neg_logits.numel() > 0:
            if obj_pos_logits.numel() > 0:
                pos_bce_elem = F.binary_cross_entropy_with_logits(obj_pos_logits, torch.ones_like(obj_pos_logits, device=device), reduction='none')
                pos_count = pos_bce_elem.numel()
            else:
                pos_bce_elem = torch.tensor([], device=device)
                pos_count = 0
            if obj_neg_logits.numel() > 0:
                neg_bce_elem = F.binary_cross_entropy_with_logits(obj_neg_logits, torch.zeros_like(obj_neg_logits, device=device), reduction='none')
                neg_count = neg_bce_elem.numel()
            else:
                neg_bce_elem = torch.tensor([], device=device)
                neg_count = 0

            # weighted average: give positives a multiplier self.pos_weight
            if pos_count > 0 and neg_count > 0:
                loss_obj = (pos_bce_elem.sum() * self.pos_weight + neg_bce_elem.sum() * 1.0) / (pos_count * self.pos_weight + neg_count + 1e-12)
            elif pos_count > 0:
                loss_obj = pos_bce_elem.mean() * self.pos_weight
            else:
                loss_obj = neg_bce_elem.mean()
        else:
            loss_obj = torch.tensor(0.0, device=device)

        # ---------------------------- CLASS-LEVEL LOSS（正/负） ----------------------------
        cls_mask = pos_mask.clone()
        if self.use_neg_loss:
            cls_mask = cls_mask | selected_neg_mask

        # ---------- positive class loss implementation (with per-class downsampling) ----------
        loss_pos = torch.tensor(0.0, device=device)
        pos_logits_list = []
        labels_list = []
        # Gather per-positive logits and assigned labels correctly
        for b in range(B):
            pos_idx = (pos_mask[b]).nonzero(as_tuple=False).squeeze(1)
            if pos_idx.numel() == 0:
                continue
            logits_b = all_scores[b, pos_idx]  # (P_b, C)
            matched_b = matched_gt_inds[b][pos_idx].long()  # (P_b,) index into gt of this image
            gt_labels_b = batch['gt_labels'][b].long().to(device)
            # matched_b are indices into gt_labels_b
            assigned_labels = gt_labels_b[matched_b]  # (P_b,)
            pos_logits_list.append(logits_b)
            labels_list.append(assigned_labels)

        if len(pos_logits_list) > 0:
            pos_logits_cat = torch.cat(pos_logits_list, dim=0)  # (P, C)
            labels_cat = torch.cat(labels_list, dim=0)  # (P,)

            # ------------- per-class downsampling -------------
            max_keep = getattr(self, 'max_pos_per_class', 128)
            P = labels_cat.shape[0]
            if max_keep is not None and P > 0:
                keep_mask = torch.zeros((P,), dtype=torch.bool, device=device)
                unique_classes = labels_cat.unique()
                for cls in unique_classes:
                    cls = int(cls.item())
                    idxs = (labels_cat == cls).nonzero(as_tuple=False).squeeze(1)
                    cnt = idxs.numel()
                    if cnt <= max_keep:
                        keep_mask[idxs] = True
                    else:
                        perm = torch.randperm(cnt, device=device)[:max_keep]
                        chosen = idxs[perm]
                        keep_mask[chosen] = True
                pos_logits_sel = pos_logits_cat[keep_mask]
                labels_sel = labels_cat[keep_mask]
            else:
                pos_logits_sel = pos_logits_cat
                labels_sel = labels_cat

            # compute pos loss using selected positives
            if labels_sel.numel() == 0:
                loss_pos = torch.tensor(0.0, device=device)
            else:
                if self.pos_loss_type == 'ce':
                    if getattr(self, 'focal_gamma', 0.0) == 0.0:
                        loss_pos = F.cross_entropy(pos_logits_sel, labels_sel, reduction='mean')
                    else:
                        loss_pos = softmax_focal_ce(pos_logits_sel, labels_sel, alpha=self.focal_alpha, gamma=self.focal_gamma, reduction='mean')
                else:
                    targets_pos = torch.zeros_like(pos_logits_sel, device=device)
                    targets_pos.scatter_(1, labels_sel.unsqueeze(1), 1.0 - self.label_smoothing)
                    if self.label_smoothing > 0 and self.num_classes > 1:
                        targets_pos += self.label_smoothing / (self.num_classes - 1)
                    if self.use_focal:
                        loss_pos = sigmoid_focal_loss(pos_logits_sel, targets_pos, alpha=self.focal_alpha, gamma=self.focal_gamma, reduction='mean')
                    else:
                        loss_pos = F.binary_cross_entropy_with_logits(pos_logits_sel, targets_pos, reduction='mean')
                loss_pos = loss_pos * self.pos_weight
        else:
            loss_pos = torch.tensor(0.0, device=device)

        # negative class loss （若使用）
        loss_neg = torch.tensor(0.0, device=device)
        if self.use_neg_loss:
            neg_mask = selected_neg_mask
            if neg_mask.sum().item() > 0:
                neg_expand = neg_mask.unsqueeze(-1).expand_as(all_scores)  # (B,N,C)
                neg_logits = all_scores[neg_expand].view(-1, self.num_classes)  # (#neg, C)
                neg_targets = torch.zeros_like(neg_logits, device=device)
                if self.use_focal:
                    neg_loss_elem = sigmoid_focal_loss(neg_logits, neg_targets, alpha=self.focal_alpha, gamma=self.focal_gamma, reduction='none')  # (K,C)
                    loss_neg = neg_loss_elem.mean() * self.neg_weight
                else:
                    neg_loss_elem = F.binary_cross_entropy_with_logits(neg_logits, neg_targets, reduction='none')
                    loss_neg = neg_loss_elem.mean() * self.neg_weight

        # combine pos and neg class losses (anchor-平均)
        n_pos = max(1, int(pos_mask.sum().item()))
        n_neg = max(0, int(selected_neg_mask.sum().item())) if self.use_neg_loss else 0

        if self.use_neg_loss and n_neg > 0:
            loss_cls_component = (loss_pos * n_pos + loss_neg * n_neg) / (n_pos + n_neg)
        else:
            loss_cls_component = loss_pos

        # final weighted class loss (keeps original naming)
        loss_cls = loss_cls_component  # weight applied later

        # -------------------------   IOU LOSS (only positives) -------------------------  
        if num_pos > 0:
            pred_pos = pred_bboxes[pos_mask]    # (P,4)
            tgt_pos = target_bboxes[pos_mask]   # (P,4)
            loss_iou = ciou_loss(pred_pos, tgt_pos).mean()
        else:
            loss_iou = torch.tensor(0.0, device=device)
        # ------------------------- DFL LOSS (only positives) -------------------------
        target_dfl = self._make_dfl_targets(gt_bboxes, matched_gt_inds, all_grids, stride_map)
        if num_pos > 0:
            pred_dfl = all_dfl[pos_mask]  # (P,4,bins)
            targ = target_dfl[pos_mask]   # (P,4,bins)
            lsm = F.log_softmax(pred_dfl, dim=-1)
            loss_dfl = -(targ * lsm).sum() / max(1, pred_dfl.shape[0])
        else:
            loss_dfl = torch.tensor(0.0, device=device)

        # ------------------------- debug printing (optional) -------------------------
        if self.debug:
            # ========== DIAGNOSTICS: run this in debug mode ==========
            with torch.no_grad():
                # 1) 检查 matched_gt_inds 是否合理：是否所有 matched 对应 gt index 在该图像范围内
                bad_match = False
                for b in range(B):
                    matched = matched_gt_inds[b]
                    if matched.numel() == 0:
                        continue
                    max_assigned = (matched[matched>=0].max().item()) if (matched>=0).any() else -1
                    if max_assigned >= 0:
                        num_gt = len(batch['gt_bboxes'][b])
                        if max_assigned >= num_gt:
                            print(f"BUG: matched index >= num_gt for image {b}: {max_assigned} >= {num_gt}")
                            bad_match = True

                # 2) 统计 positives 的真实类别分布（查看是否被单类主导）
                pos_labels_all = []
                for b in range(B):
                    pos_idx = (pos_mask[b]).nonzero(as_tuple=False).squeeze(1)
                    if pos_idx.numel() == 0:
                        continue
                    matched = matched_gt_inds[b][pos_idx].long()
                    if matched.numel() == 0:
                        continue
                    labels_b = batch['gt_labels'][b].long().to(device)
                    assigned_labels = labels_b[matched]
                    pos_labels_all.append(assigned_labels.cpu())
                if len(pos_labels_all) > 0:
                    pos_labels_cat = torch.cat(pos_labels_all, dim=0)
                    vals, cnts = torch.unique(pos_labels_cat, return_counts=True)
                    freq = list(zip(vals.tolist(), cnts.tolist()))
                    print("POS label freq (label, count):", freq[:50])
                else:
                    print("No positives in this batch")

                # 3) 查看 class-head 在 batch 上的 top predicted class counts
                all_logits_flat = all_scores.view(-1, all_scores.shape[-1])
                pred_classes = torch.argmax(all_logits_flat, dim=-1)
                vals2, cnts2 = torch.unique(pred_classes, return_counts=True)
                print("Pred class freq (global, might include many background anchors):", list(zip(vals2.tolist(), cnts2.tolist()))[:50])

                # 4) 检查 gt_labels 全局分布（batch内）
                gt_all = []
                for b in range(B):
                    gt_all.append(batch['gt_labels'][b].long().cpu())
                if len(gt_all) > 0:
                    gt_all_cat = torch.cat(gt_all, dim=0)
                    gv, gc = torch.unique(gt_all_cat, return_counts=True)
                    print("GT freq (batch):", list(zip(gv.tolist(), gc.tolist()))[:50])

                # 5) 检查 classifier weight/bias 如果你可以访问 model (示例)
                try:
                    if hasattr(self, 'model_for_debug'):
                        m = self.model_for_debug
                        for n, p in m.named_parameters():
                            if 'cls' in n or 'head' in n:
                                print("cls param", n, "norm:", p.detach().norm().item())
                except Exception as e:
                    print("skip classifier param debug:", e)

                # 6) sanity check: assigned_labels 是否越界或都相同
                if len(pos_labels_all) > 0:
                    if (pos_labels_cat.max().item() >= self.num_classes) or (pos_labels_cat.min().item() < 0):
                        print("LABEL INDEX ERROR: assigned label outside [0, num_classes-1]:", pos_labels_cat.min().item(), pos_labels_cat.max().item())
                    unique_labels = torch.unique(pos_labels_cat)
                    if unique_labels.numel() == 1:
                        print("WARNING: positives assigned to only one label in this batch:", unique_labels.item())

        return {
            'loss_iou': self.loss_iou_weight * loss_iou,
            'loss_cls': self.loss_cls_weight * loss_cls,
            'loss_obj': self.loss_obj_weight * loss_obj,
            'loss_dfl': self.loss_dfl_weight * loss_dfl}



class YOLOv10Loss(nn.Module):
    def __init__(
        self,
        strides: List[int] = [8, 16, 32],
        num_classes: int = 80,
        dfl_bins: int = 16,
        loss_cls_weight: float = 0.5,
        loss_iou_weight: float = 7.5,
        loss_dfl_weight: float = 1.5,
        tal_topk: int = 10,
        use_focal: bool = True,          # focal for BCE path
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        debug: bool = False,
        # hard-neg 采样和正负样本权重
        neg_pos_ratio: int = 3,
        neg_thresh: float = 0.05,
        pos_weight: float = 2.0,
        neg_weight: float = 0.5,
        # new control params
        pos_loss_type: str = 'bce',   # 'bce' (default) or 'ce' (softmax CE focal)
        use_neg_loss: bool = True,
        neg_selection: str = 'per_image',
        label_smoothing = 0.1,
        o2m_weight: float = .8,
    ):
        super().__init__()
        self.o2m_weight = o2m_weight
        # one2one: prefer single-label CE -> pass pos_loss_type='ce' when creating
        self.one2one_loss = YOLOv8Loss(
            strides = strides,
            num_classes = num_classes,
            dfl_bins = dfl_bins,
            loss_cls_weight = loss_cls_weight,
            loss_iou_weight = loss_iou_weight,
            loss_dfl_weight = loss_dfl_weight,
            use_focal = use_focal,
            focal_alpha = focal_alpha,
            focal_gamma = focal_gamma,
            debug = debug,
            neg_pos_ratio = neg_pos_ratio,
            neg_thresh = neg_thresh,
            pos_weight = pos_weight,
            neg_weight = neg_weight,
            pos_loss_type = 'ce',
            use_neg_loss = use_neg_loss,
            neg_selection = neg_selection,
            tal_topk = 1,
            label_smoothing = label_smoothing)
        self.one2many_loss = YOLOv8Loss(
            strides = strides,
            num_classes = num_classes,
            dfl_bins = dfl_bins,
            loss_cls_weight = loss_cls_weight,
            loss_iou_weight = loss_iou_weight,
            loss_dfl_weight = loss_dfl_weight,
            use_focal = use_focal,
            focal_alpha = focal_alpha,
            focal_gamma = focal_gamma,
            debug = debug,
            neg_pos_ratio = neg_pos_ratio,
            neg_thresh = neg_thresh,
            pos_weight = pos_weight,
            neg_weight = neg_weight,
            pos_loss_type = pos_loss_type,
            use_neg_loss = use_neg_loss,
            neg_selection = neg_selection,
            tal_topk = 5,
            label_smoothing = label_smoothing)

    def forward(self, preds, targs):
        # preds: {'one2one': [...], 'one2many': [...]}
        one2one_loss = self.one2one_loss(preds['one2one'], targs)
        one2many_loss = self.one2many_loss(preds['one2many'], targs)
        loss = {}
        for k, v in one2one_loss.items():
            loss[f'loss_one2one_{k}'] = v * (1 - self.o2m_weight)
        for k, v in one2many_loss.items():
            loss[f'loss_one2many_{k}'] = v * self.o2m_weight

        return loss
