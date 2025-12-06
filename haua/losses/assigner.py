import torch
import torch.nn.functional as F
from typing import Tuple, Optional

from ..models.utils import centers_of_boxes, pairwise_iou, bbox_iou


def atss_assign(
    anchors: torch.Tensor,
    gt_boxes: torch.Tensor,
    num_candidates: int = 9,
    topk: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ATSS label assignment (per image).
    Args:
        anchors: (N,4) anchor boxes / grid boxes in x1,y1,x2,y2
        gt_boxes: (M,4) ground-truth boxes in same coords
        num_candidates: number of nearest anchors per GT to consider (default 9)
        topk: alias for num_candidates (kept for clarity)
    Returns:
        assigned_gt_inds: (N,) long tensor, -1 means negative/background, >=0 is index of assigned gt in gt_boxes
        assigned_ious: (N,) float tensor, IoU of assigned gt (0 if negative)
    Notes:
        This implements the core ATSS idea:
         - For each GT, select K nearest anchors (by center distance)
         - Compute IoUs of those K anchors to this GT, take mean+std as threshold
         - Anchors whose IoU >= threshold and whose center is inside GT are positives for that GT.
         - If multiple GTs assign same anchor, choose GT with highest IoU.
    """
    if gt_boxes.numel() == 0:
        N = anchors.size(0)
        return (
            torch.full((N,),-1, dtype=torch.long, device=anchors.device),
            torch.zeros(N, device=anchors.device))

    N = anchors.size(0)
    M = gt_boxes.size(0)
    # centers
    anc_centers = centers_of_boxes(anchors)  # (N,2)
    gt_centers = centers_of_boxes(gt_boxes)  # (M,2)

    # pairwise center distance
    dist = torch.cdist(gt_centers, anc_centers)  # (M,N)
    # for each gt, choose k nearest anchors
    k = num_candidates if topk is None else topk
    k = min(k, N)
    topk_ids = dist.topk(k, largest=False, dim=1).indices  # (M,k)

    # compute IoU matrix (N,M) or (M,N) -> keep (N,M) for convenience
    ious = pairwise_iou(anchors, gt_boxes)  # (N,M)
    ious_t = ious.T  # (M,N)

    # threshold per GT: mean + std of top-k ious
    candidate_ious = torch.gather(ious_t, 1, topk_ids)  # (M,k)
    mean_per_gt = candidate_ious.mean(1)
    std_per_gt = candidate_ious.std(1)
    thr = mean_per_gt + std_per_gt  # (M,)

    # for each GT mark anchors whose IoU >= thr and whose center is inside GT box
    # center in gt:
    anc_x = anc_centers[:,0]
    anc_y = anc_centers[:,1]
    gt_x1, gt_y1, gt_x2, gt_y2 = gt_boxes[:,0], gt_boxes[:,1], gt_boxes[:,2], gt_boxes[:,3]  # (M,)

    # expand to (M,N)
    anc_x_expand = anc_x[None, :].repeat(M,1)
    anc_y_expand = anc_y[None, :].repeat(M,1)
    inside_gt = (anc_x_expand >= gt_x1[:,None]) & (anc_x_expand <= gt_x2[:,None]) & \
                (anc_y_expand >= gt_y1[:,None]) & (anc_y_expand <= gt_y2[:,None])

    ious_mask = (ious_t >= thr[:,None]) & inside_gt  # (M,N)
    # assign: for each anchor, pick gt with highest IoU among candidates, else -1
    # convert back to (N,M) for easier gather
    ious_NM = ious  # (N,M)
    # for anchors that have any True in ious_mask[:,n], select the gt with max iou
    matched_gt_inds = torch.full((N,), -1, dtype=torch.long, device=anchors.device)
    matched_ious = torch.zeros((N,), dtype=anchors.dtype, device=anchors.device)

    for m in range(M):
        mask_m = ious_mask[m]  # (N,)
        if mask_m.any():
            # candidate anchors for this gt
            candidate_idxs = mask_m.nonzero(as_tuple=False).squeeze(1)
            # for those anchors, try to set gt index if IoU larger than previous assigned
            cand_ious = ious_NM[candidate_idxs, m]  # IoUs
            # compare with existing assigned IoUs
            prev_ious = matched_ious[candidate_idxs]
            update_mask = cand_ious > prev_ious
            if update_mask.any():
                to_update = candidate_idxs[update_mask]
                matched_gt_inds[to_update] = m
                matched_ious[to_update] = cand_ious[update_mask]

    return matched_gt_inds, matched_ious


def simota_assign(
    anchors: torch.Tensor,
    pred_cls_logits: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: Optional[torch.Tensor] = None,
    center_radius: float = 2.5,
    topk: int = 10,
    cls_weight: float = 1.0,
    iou_weight: float = 3.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    SimOTA-like assigner (per image).
    Args:
        anchors: (N,4) candidate boxes (x1,y1,x2,y2) or grid centers but boxes expected.
        pred_cls_logits: (N,C) predicted class logits for each anchor (before sigmoid)
        gt_boxes: (M,4) gt boxes
        gt_labels: (M,) long labels (optional, used for computing classification cost); if None classification cost ignored
        center_radius: radius (in units of anchor stride) to select candidate anchors per GT. If anchors are arbitrary, code uses center distance normalized by average stride; if uncertain, keep it large.
        topk: number of top candidate anchors by center distance to consider per GT
        cls_weight, iou_weight: weights for cost terms
    Returns:
        matched_gt_inds: (N,) long tensor, -1 background, >=0 gt index
        matched_ious: (N,) float tensor of IoU with matched gt or 0
    Notes:
        This is a simplified implementation of SimOTA/OTA dynamic-k matching.
    """
    device = anchors.device
    N = anchors.size(0)
    M = gt_boxes.size(0)
    if M == 0:
        return torch.full((N,), -1, dtype=torch.long, device=device), torch.zeros((N,), device=device)

    # 1) compute pairwise iou (N,M)
    ious = pairwise_iou(anchors, gt_boxes)  # (N,M)

    # 2) compute center distances and pre-select candidates per GT
    anc_centers = centers_of_boxes(anchors)  # (N,2)
    gt_centers = centers_of_boxes(gt_boxes)  # (M,2)
    center_dist = torch.cdist(gt_centers, anc_centers)  # (M,N)
    k = min(topk, N)
    candidate_idxs = center_dist.topk(k, largest=False, dim=1).indices  # (M,k)

    # 3) compute cost matrix for candidates: classification cost + iou cost
    # classification cost: if gt_labels provided, use BCE between pred prob and one-hot, else ignore cls cost
    # convert logits -> prob (sigmoid)
    cls_cost = torch.zeros((N, M), device=device)
    if gt_labels is not None and pred_cls_logits is not None:
        # pred_cls_logits: (N,C)
        pred_prob = pred_cls_logits.sigmoid()  # (N,C)
        # for each gt m, classification cost = -pred_prob[:,gt_label] (higher prob smaller cost)
        for m in range(M):
            label = int(gt_labels[m].item())
            # negative log-likelihood like cost: use -log(p) ~ but we can use -p to keep simple and consistent
            cls_cost[candidate_idxs[m], m] = -pred_prob[candidate_idxs[m], label]
    # iou cost: -log(iou)
    iou_cost = -torch.log(ious.clamp(min=1e-7))  # (N,M)

    # combine cost only for candidate positions; others stay large
    INF = 1e9
    cost = torch.full((N, M), INF, device=device)
    for m in range(M):
        idxs = candidate_idxs[m]  # (k,)
        cost[idxs, m] = cls_weight * cls_cost[idxs, m] + iou_weight * iou_cost[idxs, m]

    # 4) dynamic K matching (greedy)
    matched_gt_inds = torch.full((N,), -1, dtype=torch.long, device=device)
    matched_ious = torch.zeros((N,), dtype=anchors.dtype, device=device)

    # for each gt, select dynamic top anchors
    # step: for each gt, determine dynamic_k = max(1, int(sum(top_ious)))
    topk_ious, _ = ious[candidate_idxs, torch.arange(M)[:,None]].topk(k=min(10, k), dim=1, largest=True) if k>0 else (torch.zeros((M,0), device=device), None)
    dynamic_ks = (topk_ious.sum(1).int().clamp(min=1)).tolist()  # list of M ints

    # now for each gt, select dynamic_k anchors with smallest cost
    # but need to resolve conflicts: anchors assigned multiple times -> keep lowest cost (or highest iou)
    for m in range(M):
        idxs = candidate_idxs[m]  # (k,)
        if idxs.numel() == 0:
            continue
        c = cost[idxs, m]  # (k,)
        k_m = dynamic_ks[m]
        k_m = min(k_m, idxs.numel())
        _, topk_idx_in_c = torch.topk(-c, k=k_m, largest=True)  # smallest cost -> largest -cost
        chosen = idxs[topk_idx_in_c]  # anchor indices chosen for this gt

        # assign: if anchor already assigned, compare cost and update if this gt gives lower cost
        for a in chosen:
            a = int(a.item())
            prev = matched_gt_inds[a].item()
            if prev == -1:
                matched_gt_inds[a] = m
                matched_ious[a] = ious[a, m]
            else:
                # conflict: keep gt giving higher IoU (or lower cost)
                if ious[a, m] > matched_ious[a]:
                    matched_gt_inds[a] = m
                    matched_ious[a] = ious[a, m]

    return matched_gt_inds, matched_ious


def tal_assign(
    pred_scores: torch.Tensor,
    pred_bboxes: torch.Tensor,
    gt_bboxes: torch.Tensor,
    gt_labels: torch.Tensor,
    topk: int = 10,
    cls_power: float = 1.0,
    iou_power: float = 3.0,
    use_center_constraint: bool = True,
    center_radius: float = 0.5,
    use_softmax: bool = False,
    min_iou_for_candidate: float = 1e-6,
    debug: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    更稳健的 Task-Aligned Assigner 实现（兼容 logits 或 probs 输入，带数值保护）
    返回:
        target_scores: (B, N, C)  — only assigned class positions non-zero (scaled by weight)
        target_bboxes: (B, N, 4)
        fg_mask: (B, N) bool
        matched_gt_inds: (B, N) long, -1 表示背景 (index into original gt list for that image)
    重要改动与策略：
      - 候选选取后会删除 IoU 极低的匹配（min_iou_for_pos，默认为至少 0.12，除非用户传了更大）
      - 对每个 gt 限制最大 assigned anchors（max_per_gt = min(topk, 9)）
      - 当 topk == 1 时，使用贪心质量排序保证一对一匹配（不允许一个 anchor 分配给多个 gt）
      - 质量度量 quality = (pred_score_for_assigned_class ** cls_power) * (IoU ** iou_power)
    """
    device = pred_scores.device
    B, N, C = pred_scores.shape

    target_scores = torch.zeros_like(pred_scores, device=device)
    target_bboxes = torch.zeros((B, N, 4), device=device)
    fg_mask = torch.zeros((B, N), dtype=torch.bool, device=device)
    matched_gt_inds = torch.full((B, N), -1, dtype=torch.long, device=device)

    eps = 1e-8
    # ensure minimal IoU for positives: prefer at least 0.12 unless caller requests higher threshold
    min_iou_for_pos = max(min_iou_for_candidate, 0.12)

    def is_prob_tensor(x: torch.Tensor):
        return float(x.min()) >= -1e-6 and float(x.max()) <= 1.0 + 1e-6

    for b in range(B):
        gt_b_all = gt_bboxes[b]  # (G,4) or maybe empty tensor
        gt_l_all = gt_labels[b]  # (G,)
        valid_mask = gt_l_all >= 0
        if not valid_mask.any():
            if debug:
                print(f"[TAL] batch {b}: no valid gt")
            continue

        valid_idx = valid_mask.nonzero(as_tuple=False).squeeze(1)
        gt_b = gt_b_all[valid_idx]   # (Gv,4)
        gt_l = gt_l_all[valid_idx]   # (Gv,)
        Gv = gt_b.shape[0]

        pb = pred_bboxes[b]   # (N,4)
        ps = pred_scores[b]   # (N,C)

        # IoU matrix (N, Gv)
        ious = bbox_iou(pb, gt_b).clamp(min=0.0, max=1.0)

        # classification probability safe conversion
        if use_softmax:
            if is_prob_tensor(ps) and torch.allclose(ps.sum(dim=1), torch.ones(ps.shape[0], device=device), atol=1e-3):
                prob = ps
            else:
                prob = F.softmax(ps, dim=1)
        else:
            if is_prob_tensor(ps):
                prob = ps
            else:
                prob = torch.sigmoid(ps)

        # cls_score: for each anchor and each gt, probability of gt's class
        gt_one_hot = F.one_hot(gt_l.long(), num_classes=C).float().to(device)  # (Gv, C)
        cls_score = prob @ gt_one_hot.t()  # (N, Gv)
        cls_score = cls_score.clamp(min=eps, max=1.0)

        # alignment metric: cls^cls_power * iou^iou_power
        align_metric = (cls_score ** cls_power) * (ious.clamp(min=eps) ** iou_power)
        align_metric = torch.nan_to_num(align_metric, nan=0.0, posinf=1e6, neginf=0.0)

        # prepare for topk selection: filter by candidate IoU threshold
        am_for_topk = align_metric.clone()
        am_for_topk[ious < min_iou_for_candidate] = -1e9

        # candidate_mask: (N, Gv)
        candidate_mask = torch.zeros_like(am_for_topk, dtype=torch.bool)

        # anchor centers and gt centers/sizes for center constraint
        pb_centers = (pb[:, :2] + pb[:, 2:]) / 2.0
        gt_centers = (gt_b[:, :2] + gt_b[:, 2:]) / 2.0
        gt_wh = (gt_b[:, 2:] - gt_b[:, :2]).clamp(min=1e-6)
        gt_radius = gt_wh.max(dim=1)[0] * center_radius

        k = min(topk, N)
        # per-gt candidate selection (topk with center constraint)
        for gi in range(Gv):
            if use_center_constraint:
                d = ((pb_centers - gt_centers[gi]) ** 2).sum(dim=1).sqrt()
                center_ok = d <= gt_radius[gi]
                inside_x = (pb_centers[:, 0] >= gt_b[gi, 0]) & (pb_centers[:, 0] <= gt_b[gi, 2])
                inside_y = (pb_centers[:, 1] >= gt_b[gi, 1]) & (pb_centers[:, 1] <= gt_b[gi, 3])
                inside_box = inside_x & inside_y
                allow = center_ok | inside_box
            else:
                allow = torch.ones((N,), dtype=torch.bool, device=device)

            vals = am_for_topk[:, gi].clone()
            vals[~allow] = -1e9
            if (vals > -1e8).sum() == 0:
                continue
            topk_vals, topk_idx = vals.topk(k=k, largest=True)
            good_mask = topk_vals > -1e8
            if good_mask.any():
                candidate_mask[topk_idx[good_mask], gi] = True

        # if no candidates -> continue
        pos_mask_any = candidate_mask.any(dim=1)
        if pos_mask_any.sum() == 0:
            if debug:
                print(f"[TAL] batch {b}: no candidate anchors after topk/center")
            continue

        # Build list of candidate pairs and their quality for conflict resolution
        # quality = align_metric (already computed), but we will combine with IoU again if needed
        # Candidate indices:
        cand_anchor_idx, cand_gt_idx = torch.where(candidate_mask)  # tensors of same length M
        if cand_anchor_idx.numel() == 0:
            if debug:
                print(f"[TAL] batch {b}: no candidates after topk filtering")
            continue

        qualities = align_metric[cand_anchor_idx, cand_gt_idx]  # (M,)

        # We'll perform matching differently depending on topk:
        # - if topk == 1: enforce one-to-one matching via greedy by quality
        # - if topk > 1: allow multiple anchors per gt but cap via max_per_gt (min(topk,9))
        max_per_gt = min(max(1, topk), 9)

        # Prepare structures to collect accepted matches
        accepted_anchor_idx = []
        accepted_gt_idx = []
        accepted_quality = []

        if topk == 1:
            # Greedy one-to-one: sort candidate pairs by quality desc, accept if anchor free and gt free
            M = qualities.shape[0]
            if M == 0:
                continue
            # sort descending
            sorted_vals, order = torch.sort(qualities, descending=True)
            anc_sorted = cand_anchor_idx[order]
            gt_sorted = cand_gt_idx[order]
            anchor_taken = torch.zeros((N,), dtype=torch.bool, device=device)
            gt_taken = torch.zeros((Gv,), dtype=torch.bool, device=device)
            for i_idx in range(M):
                a = int(anc_sorted[i_idx].item()); g = int(gt_sorted[i_idx].item())
                if anchor_taken[a] or gt_taken[g]:
                    continue
                # check IoU quality threshold again
                if ious[a, g] < min_iou_for_pos:
                    continue
                accepted_anchor_idx.append(a)
                accepted_gt_idx.append(g)
                accepted_quality.append(float(sorted_vals[i_idx].item()))
                anchor_taken[a] = True
                gt_taken[g] = True

        else:
            # topk > 1: allow multiple anchors per gt but cap max_per_gt
            # For each gt, select up to max_per_gt anchors by quality, additionally ensure these anchors pass min_iou_for_pos
            for gi in range(Gv):
                # anchors candidate for this gt
                mask_for_gt = cand_gt_idx == gi
                if mask_for_gt.sum().item() == 0:
                    continue
                a_idxs = cand_anchor_idx[mask_for_gt]
                q_vals = qualities[mask_for_gt]
                # filter by min_iou_for_pos
                keep_mask = ious[a_idxs, gi] >= min_iou_for_pos
                if keep_mask.sum().item() == 0:
                    continue
                a_keep = a_idxs[keep_mask]
                q_keep = q_vals[keep_mask]
                k2 = min(max_per_gt, a_keep.numel())
                topk_vals, topk_idx = torch.topk(q_keep, k=k2, largest=True)
                chosen_anchors = a_keep[topk_idx]
                for ii in range(chosen_anchors.numel()):
                    accepted_anchor_idx.append(int(chosen_anchors[ii].item()))
                    accepted_gt_idx.append(gi)
                    accepted_quality.append(float(topk_vals[ii].item()))

            # Note: the same anchor may be accepted for multiple gts here (rare if min_iou_for_pos is set).
            # We'll resolve duplicates by keeping the GT with highest quality for that anchor.
            if len(accepted_anchor_idx) > 0:
                # convert to tensors
                anc = torch.tensor(accepted_anchor_idx, dtype=torch.long, device=device)
                gtids = torch.tensor(accepted_gt_idx, dtype=torch.long, device=device)
                quals = torch.tensor(accepted_quality, dtype=torch.float, device=device)
                # group by anchor and keep best gt per anchor
                unique_anchors = torch.unique(anc)
                final_anc = []
                final_gt = []
                final_q = []
                for ua in unique_anchors:
                    mask = (anc == ua)
                    sub_q = quals[mask]
                    sub_gt = gtids[mask]
                    best_idx = torch.argmax(sub_q)
                    final_anc.append(int(ua.item()))
                    final_gt.append(int(sub_gt[best_idx].item()))
                    final_q.append(float(sub_q[best_idx].item()))
                accepted_anchor_idx = final_anc
                accepted_gt_idx = final_gt
                accepted_quality = final_q

        # now populate matched arrays for this image
        if len(accepted_anchor_idx) == 0:
            if debug:
                print(f"[TAL] batch {b}: no accepted matches after resolution")
            continue

        anc_tensor = torch.tensor(accepted_anchor_idx, dtype=torch.long, device=device)
        gt_tensor = torch.tensor(accepted_gt_idx, dtype=torch.long, device=device)
        qual_tensor = torch.tensor(accepted_quality, dtype=torch.float, device=device)

        # map gt_tensor (index into valid_gt list) back to original gt indices in batch
        matched_orig_idx = valid_idx[gt_tensor]  # indices in original gt list (for this image)
        matched_gt_inds[b, anc_tensor] = matched_orig_idx.long()

        # fill target_bboxes and target_scores for assigned anchors
        # target_bboxes should contain the assigned gt box (using valid local index)
        target_bboxes[b, anc_tensor] = gt_b[gt_tensor]

        # assigned labels
        assigned_labels = gt_l[gt_tensor].long()  # local label ids
        # set only the assigned-class position to 1 (others remain 0)
        target_scores[b, anc_tensor, assigned_labels] = 1.0

        # turn qual_tensor into normalized weights (per-image)
        w = qual_tensor.clone().detach()
        w = torch.clamp(w, min=0.0)
        maxw = w.max() if w.numel() > 0 else 1.0
        if maxw > 0:
            w = w / (maxw + eps)
        else:
            w = torch.ones_like(w)
        # stabilize distribution (sqrt)
        w = torch.sqrt(w)

        # multiply only the assigned class position (target_scores currently has 1 at assigned class)
        # using broadcasting: target_scores[b, anc_tensor] is (K, C) with one-hot; multiply rows by w
        target_scores[b, anc_tensor] = target_scores[b, anc_tensor] * w.unsqueeze(-1)

        # mark fg_mask for these anchors
        fg_mask[b, anc_tensor] = True

        if debug:
            print(f"[TAL debug] batch {b}: accepted {len(accepted_anchor_idx)} positives, max quality {float(qual_tensor.max()):.6f}")

    return target_scores, target_bboxes, fg_mask, matched_gt_inds
