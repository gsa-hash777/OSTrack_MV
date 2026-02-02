# train_fusion_only.py
import os
import argparse
import random
from typing import List, Tuple

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.test.tracker.ostrack import OSTrack
from lib.test.evaluation import Tracker
from lib.utils.box_ops import clip_box


# -----------------------------
# 1) 读取GT（与你现有脚本一致）
# -----------------------------
def read_initial_groundtruth(video_path):
    gt_file = os.path.join(video_path, 'groundtruth.txt')
    with open(gt_file, 'r') as f:
        line = f.readline().strip()
        if ',' in line:
            init_rect = list(map(int, line.split(',')))
        else:
            init_rect = list(map(int, line.split()))
    return init_rect  # [x,y,w,h]

def read_all_groundtruth(video_path):
    gt_boxes = []
    gt_file = os.path.join(video_path, 'groundtruth.txt')
    with open(gt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ',' in line:
                rect = list(map(int, line.split(',')))
            else:
                rect = list(map(int, line.split()))
            gt_boxes.append(rect)
    return gt_boxes  # list of [x,y,w,h] per frame


# -----------------------------
# 2) 轻量可学习融合模块
# -----------------------------
class LiteCrossViewFusion(nn.Module):
    """
    可学习的跨视角后期融合（轻量）：
    - query: 当前视角 cat_features
    - key/value: 其他高置信度视角 cat_features
    - bottleneck attention + 小MLP + 可学习gate
    """
    def __init__(self, dim=768, bottleneck=128, num_heads=4, mlp_ratio=2.0, drop=0.0):
        super().__init__()
        self.dim = dim
        self.d = bottleneck
        self.h = num_heads
        assert bottleneck % num_heads == 0

        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.ln_out = nn.LayerNorm(bottleneck)

        self.q_proj = nn.Linear(dim, bottleneck, bias=False)
        self.k_proj = nn.Linear(dim, bottleneck, bias=False)
        self.v_proj = nn.Linear(dim, bottleneck, bias=False)
        self.out_proj = nn.Linear(bottleneck, dim, bias=False)

        hidden = int(bottleneck * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(bottleneck, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, bottleneck),
            nn.Dropout(drop),
        )

        # gate输入：(query_conf, mean_k, max_k, min_k) -> scalar gate
        self.gate_mlp = nn.Sequential(
            nn.Linear(4, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.scale = (bottleneck // num_heads) ** -0.5

    def _split(self, x):  # (B,N,d) -> (B,h,N,dh)
        B, N, d = x.shape
        dh = d // self.h
        return x.view(B, N, self.h, dh).permute(0, 2, 1, 3)

    def _merge(self, x):  # (B,h,N,dh) -> (B,N,d)
        B, h, N, dh = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B, N, h * dh)

    def forward(self, query_feat, key_feat_list, key_conf_list, query_conf, temperature=1.0, top_k=None):
        if len(key_feat_list) == 0:
            gate = torch.tensor(0.0, device=query_feat.device, dtype=query_feat.dtype)
            return query_feat, gate

        q0 = self.ln_q(query_feat)
        kv = torch.cat(key_feat_list, dim=1)  # (B, Nt, C)
        kv0 = self.ln_kv(kv)

        q = self.q_proj(q0)      # (B,N,d)
        k = self.k_proj(kv0)     # (B,Nt,d)
        v = self.v_proj(kv0)     # (B,Nt,d)

        qh = self._split(q)
        kh = self._split(k)
        vh = self._split(v)

        attn = (qh @ kh.transpose(-2, -1)) * (self.scale / max(temperature, 1e-6))  # (B,h,N,Nt)

        if top_k is not None and top_k < attn.shape[-1]:
            topv, topi = torch.topk(attn, k=top_k, dim=-1)
            mask = torch.ones_like(attn, dtype=torch.bool)
            mask.scatter_(-1, topi, False)
            attn = attn.masked_fill(mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        fused = attn @ vh                # (B,h,N,dh)
        fused = self._merge(fused)       # (B,N,d)
        fused = fused + self.mlp(self.ln_out(fused))

        # gate
        kc = torch.tensor(key_conf_list, device=query_feat.device, dtype=query_feat.dtype)
        stats = torch.stack([
            torch.tensor(float(query_conf), device=query_feat.device, dtype=query_feat.dtype),
            kc.mean() if kc.numel() > 0 else torch.tensor(0., device=query_feat.device, dtype=query_feat.dtype),
            kc.max() if kc.numel() > 0 else torch.tensor(0., device=query_feat.device, dtype=query_feat.dtype),
            kc.min() if kc.numel() > 0 else torch.tensor(0., device=query_feat.device, dtype=query_feat.dtype),
        ], dim=0)

        gate = torch.sigmoid(self.gate_mlp(stats)[0])  # scalar
        fusedC = self.out_proj(fused)                  # (B,N,C)

        out = (1.0 - gate) * query_feat + gate * fusedC
        return out, gate


# def fuse_low_confidence_views_trainable(
#     fusion_net,
#     features_list: List[torch.Tensor],  # each (1,N,C)
#     confidences: List[float],
#     score_threshold=0.5,
#     temperature=0.7,
#     top_k=None,
# ):
#     updated = list(features_list)
#     gates = [None] * len(features_list)
#     low_mask = [c < score_threshold for c in confidences]
#
#     high_idx = [i for i, c in enumerate(confidences) if c >= score_threshold]
#     for i, is_low in enumerate(low_mask):
#         if not is_low:
#             continue
#         key_feats = [features_list[j] for j in high_idx if j != i]
#         key_confs = [confidences[j] for j in high_idx if j != i]
#         if len(key_feats) == 0:
#             continue
#         out_i, gate_i = fusion_net(
#             query_feat=features_list[i],
#             key_feat_list=key_feats,
#             key_conf_list=key_confs,
#             query_conf=confidences[i],
#             temperature=temperature,
#             top_k=top_k,
#         )
#         updated[i] = out_i
#         gates[i] = float(gate_i.detach().cpu().item())
#
#     return updated, gates, low_mask


# -----------------------------
# 3) 冻结 tracker + 可微loss（crop坐标系）
# -----------------------------
def freeze_tracker(tracker: OSTrack):
    tracker.network.eval()
    for p in tracker.network.parameters():
        p.requires_grad = False

def forward_head_and_box(tracker: OSTrack, feat_last: torch.Tensor) -> torch.Tensor:
    """
    返回 pred_box_crop: Tensor(4,) [cx,cy,w,h] in crop coords
    与你推理逻辑一致，但不转tolist，保证可反传
    """
    out = tracker.network.forward_head(feat_last, None)
    pred_score_map = out["score_map"]
    response = tracker.output_window * pred_score_map

    pred_boxes = tracker.network.box_head.cal_bbox(response, out["size_map"], out["offset_map"])
    pred_boxes = pred_boxes.view(-1, 4)
    pred_box = pred_boxes.mean(dim=0) * tracker.params.search_size / tracker.resize_factor
    return pred_box  # Tensor(4,)

def map_box_to_crop_torch(gt_xywh: torch.Tensor, prev_state_xywh: torch.Tensor, search_size: int, resize_factor: float):
    """
    这是你 map_box_back 的严格逆变换：
    map_box_back:
      half_side = 0.5*search_size/resize_factor
      cx_real = cx + (cx_prev - half_side)
      x = cx_real - 0.5*w
    逆：
      cx = cx_real - (cx_prev - half_side)
      w = w_gt
    """
    cx_prev = prev_state_xywh[0] + 0.5 * prev_state_xywh[2]
    cy_prev = prev_state_xywh[1] + 0.5 * prev_state_xywh[3]
    half_side = 0.5 * float(search_size) / float(resize_factor)

    cx_real = gt_xywh[0] + 0.5 * gt_xywh[2]
    cy_real = gt_xywh[1] + 0.5 * gt_xywh[3]

    cx = cx_real - (cx_prev - half_side)
    cy = cy_real - (cy_prev - half_side)
    w = gt_xywh[2]
    h = gt_xywh[3]
    return torch.stack([cx, cy, w, h], dim=0)

def feature_delta_loss(updated_feat: torch.Tensor, original_feat: torch.Tensor):
    return (updated_feat - original_feat).pow(2).mean()


# -----------------------------
# 4) 数据读取：按md目录->多视角img/ + groundtruth.txt
# -----------------------------
def list_md_dirs(root_dir: str) -> List[str]:
    md_list = sorted([
        d for d in os.listdir(root_dir)
        if d.startswith("md") and os.path.isdir(os.path.join(root_dir, d))
    ])
    return md_list

def discover_view_dirs(md_dir: str) -> List[str]:
    """
    md_dir下面通常有多个view子目录（如 camera1/camera2 或 view1/view2 等）
    你原脚本里是遍历 md 下的子目录，并取 vd/img
    这里我们通用化：找出包含 img/ 和 groundtruth.txt 的子目录
    """
    views = []
    for vd in sorted(os.listdir(md_dir)):
        p = os.path.join(md_dir, vd)
        if not os.path.isdir(p):
            continue
        img_dir = os.path.join(p, "img")
        gt_file = os.path.join(p, "groundtruth.txt")
        if os.path.isdir(img_dir) and os.path.isfile(gt_file):
            views.append(vd)
    return views

def load_image_paths(img_dir: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = sorted([
        os.path.join(img_dir, f) for f in os.listdir(img_dir)
        if f.lower().endswith(exts)
    ])
    return files

def read_frame_bgr(path: str):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return im

def bgr_to_rgb_uint8(im_bgr):
    return cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)


# -----------------------------
# 5) 单个md序列的训练（逐帧）
# -----------------------------
def train_one_md(
    md_path: str,
    tracker_name: str,
    tracker_param: str,
    device: str,
    fusion_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    score_threshold=0.5,
    temperature=0.7,
    top_k=None,
    w_l1=1.0,
    w_delta=0.05,
    max_frames=None,
    seed=0,
    verbose=False,
    use_only_high_as_kv: bool = False,   # True=只用高置信度当kv；False=用所有其它视角当kv（更稳）
    loss_log: list = None,
):
    import os, random
    import torch
    random.seed(seed)

    view_dirs = discover_view_dirs(md_path)
    if len(view_dirs) == 0:
        if verbose:
            print(f"[Skip] no valid views in {md_path}")
        return 0.0, 0

    paths = [os.path.join(md_path, vd) for vd in view_dirs]
    img_dirs = [os.path.join(p, "img") for p in paths]
    img_files_per_view = [load_image_paths(d) for d in img_dirs]

    num_frames = min(len(x) for x in img_files_per_view)
    if max_frames is not None:
        num_frames = min(num_frames, int(max_frames))

    optional_boxes = [read_initial_groundtruth(p) for p in paths]  # [x,y,w,h]
    all_gt_boxes   = [read_all_groundtruth(p)    for p in paths]   # list frames

    gt_frames = min(len(g) for g in all_gt_boxes)
    num_frames = min(num_frames, gt_frames)
    if num_frames <= 1:
        if verbose:
            print(f"[Skip] too short sequence {md_path}")
        return 0.0, 0

    # ---------- 创建 trackers（每视角一个） ----------
    trackers = []
    for _ in img_dirs:
        tracker_instance = Tracker(tracker_name, tracker_param, "image_sequence")
        params = tracker_instance.get_parameters()
        if not hasattr(params, "debug"):
            params.debug = 0
        trk = OSTrack(params, "image_sequence")
        freeze_tracker(trk)  # 彻底冻结网络参数
        trackers.append(trk)

    # ---------- 初始化（第0帧） ----------
    first_frames_rgb = []
    for v in range(len(trackers)):
        im_bgr = read_frame_bgr(img_files_per_view[v][0])
        im_rgb = bgr_to_rgb_uint8(im_bgr)
        first_frames_rgb.append(im_rgb)

    for v, trk in enumerate(trackers):
        init_box = optional_boxes[v]
        trk.initialize(first_frames_rgb[v], {"init_bbox": init_box})

    total_loss_val = 0.0
    total_steps = 0

    # ---------- 逐帧训练（从1开始） ----------
    for t in range(1, num_frames):
        frames_rgb_list = []
        gt_img_xywh_list = []
        for v in range(len(trackers)):
            im_bgr = read_frame_bgr(img_files_per_view[v][t])
            im_rgb = bgr_to_rgb_uint8(im_bgr)
            frames_rgb_list.append(im_rgb)

            gt_xywh = all_gt_boxes[v][t]
            gt_img_xywh_list.append(torch.tensor(gt_xywh, device=device, dtype=torch.float32))

        # ============================================================
        # (1) track：只取 feature/conf —— 强制 no_grad
        # ============================================================
        features = []
        confidences = []
        with torch.no_grad():
            for v, trk in enumerate(trackers):
                out = trk.track(frames_rgb_list[v])  # 你的track内部也no_grad，不影响
                score_map = out.get("score_map", None)     # response
                feat = out.get("cat_features", None)       # backbone_feat (B, HWt+HWs, C)

                if score_map is None or feat is None:
                    features = []
                    break

                confidences.append(float(score_map.max().item()))
                features.append(feat.to(device))

        if len(features) != len(trackers):
            continue

        low_mask = [c < score_threshold for c in confidences]
        if sum(low_mask) == 0:
            # 没有低置信度视角，不训练
            continue

        # ============================================================
        # (2) fusion + head + loss：强制 enable_grad（防止外层no_grad污染）
        # ============================================================
        with torch.enable_grad():
            fusion_net.train()

            updated_features = list(features)
            gates = [None] * len(features)

            # ----- fusion（只对低置信度视角）-----
            for i in range(len(features)):
                if not low_mask[i]:
                    continue

                if use_only_high_as_kv:
                    kv_idx = [j for j, c in enumerate(confidences) if (c >= score_threshold and j != i)]
                else:
                    kv_idx = [j for j in range(len(features)) if j != i]  # ✅ 推荐：所有其它视角

                if len(kv_idx) == 0:
                    continue

                key_feats = [features[j] for j in kv_idx]
                key_confs = [confidences[j] for j in kv_idx]

                out_i, gate_i = fusion_net(
                    query_feat=features[i],
                    key_feat_list=key_feats,
                    key_conf_list=key_confs,
                    query_conf=confidences[i],
                    temperature=temperature,
                    top_k=top_k
                )
                updated_features[i] = out_i
                gates[i] = float(gate_i.detach().cpu().item())

            # ----- loss + backward -----
            optimizer.zero_grad(set_to_none=True)

            loss_sum = None
            used = 0

            for v, trk in enumerate(trackers):
                if not low_mask[v]:
                    continue

                fused_feat = updated_features[v]

                # 保险：如果这个视角没进fusion图（几乎只会在你kv_idx为空时发生），跳过
                if not fused_feat.requires_grad:
                    continue

                # pred_box_crop 必须是 Tensor（且不tolist/detach）
                pred_box_crop = forward_head_and_box(trk, fused_feat)

                # GT映射到crop coords（与你map_box_back严格互逆）
                prev_state = torch.tensor(trk.state, device=pred_box_crop.device, dtype=pred_box_crop.dtype)
                gt_img_xywh = gt_img_xywh_list[v].to(device=pred_box_crop.device, dtype=pred_box_crop.dtype)

                gt_box_crop = map_box_to_crop_torch(
                    gt_xywh=gt_img_xywh,
                    prev_state_xywh=prev_state,
                    search_size=trk.params.search_size,
                    resize_factor=trk.resize_factor
                )

                loss_l1 = torch.abs(pred_box_crop - gt_box_crop).mean()
                loss_reg = feature_delta_loss(fused_feat, features[v])

                loss = w_l1 * loss_l1 + w_delta * loss_reg
                loss_sum = loss if (loss_sum is None) else (loss_sum + loss)
                used += 1

            if used == 0:
                continue

            loss_sum = loss_sum / used

            # 最后一道保险：不可导就跳过，绝不 backward
            if not loss_sum.requires_grad:
                if verbose:
                    print(f"[Warn] loss has no grad at {os.path.basename(md_path)} t={t}, skip")
                continue

            loss_sum.backward()
            optimizer.step()

            if loss_log is not None:
                loss_log.append(float(loss_sum.detach().cpu().item()))

            total_loss_val += float(loss_sum.detach().cpu().item())
            total_steps += 1

            if verbose and (t % 20 == 0):
                mean_conf = sum(confidences) / len(confidences)
                print(f"[{os.path.basename(md_path)} t={t:4d}] loss={float(loss_sum):.4f} used={used} low={sum(low_mask)} mean_conf={mean_conf:.3f} gates={gates}")

    return total_loss_val, total_steps



# -----------------------------
# 6) 主训练入口
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True, help="数据根目录，包含 md* 子目录")
    parser.add_argument("--tracker_name", type=str, default="ostrack")
    parser.add_argument("--tracker_param", type=str, default="vitb_256_mae_32x4_ep300")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)

    parser.add_argument("--w_l1", type=float, default=1.0)
    parser.add_argument("--w_delta", type=float, default=0.05)

    parser.add_argument("--max_mds", type=int, default=-1)
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--save_path", type=str, default="fusion_net.pth")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available but device is cuda.")

    # fusion_net（只训练它）
    fusion_net = LiteCrossViewFusion(dim=768, bottleneck=128, num_heads=4).to(device)
    optimizer = torch.optim.AdamW(fusion_net.parameters(), lr=args.lr, weight_decay=1e-4)

    train_losses = []  # 用于画 loss 曲线

    md_list = list_md_dirs(args.root_dir)
    if args.max_mds > 0:
        md_list = md_list[:args.max_mds]

    print(f"Found {len(md_list)} md sequences under {args.root_dir}")

    for ep in range(args.epochs):
        random.shuffle(md_list)
        ep_loss = 0.0
        ep_steps = 0

        for md in md_list:
            md_path = os.path.join(args.root_dir, md)
            loss_val, steps = train_one_md(
                md_path=md_path,
                tracker_name=args.tracker_name,
                tracker_param=args.tracker_param,
                device=device,
                fusion_net=fusion_net,
                optimizer=optimizer,
                score_threshold=args.score_threshold,
                temperature=args.temperature,
                top_k=(args.top_k if args.top_k > 0 else None),
                w_l1=args.w_l1,
                w_delta=args.w_delta,
                max_frames=(args.max_frames if args.max_frames > 0 else None),
                seed=args.seed + ep,
                verbose=args.verbose,
                loss_log=train_losses,
            )
            ep_loss += loss_val
            ep_steps += steps

        mean_loss = ep_loss / max(ep_steps, 1)
        print(f"[Epoch {ep+1}/{args.epochs}] steps={ep_steps} mean_loss={mean_loss:.6f}")

        # 保存
        torch.save(
            {
                "fusion_net": fusion_net.state_dict(),
                "epoch": ep,
                "args": vars(args),
            },
            args.save_path
        )
        print(f"Saved fusion_net to {args.save_path}")

    import matplotlib.pyplot as plt
    import numpy as np

    if len(train_losses) > 0:
        plt.figure(figsize=(10, 4))
        plt.plot(train_losses, label="Train Loss", linewidth=1)

        # 可选：滑动平均（更好看）
        win = 20
        if len(train_losses) > win:
            smooth = np.convolve(train_losses, np.ones(win) / win, mode='valid')
            plt.plot(range(win - 1, win - 1 + len(smooth)), smooth,
                     label=f"Moving Avg ({win})", linewidth=2)

        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.title("FusionNet Training Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        save_path = os.path.join(os.getcwd(), "train_loss_curve.png")
        plt.savefig(save_path, dpi=150)
        plt.show()

        print(f"Loss curve saved to: {save_path}")

    print("Training finished.")


if __name__ == "__main__":
    main()
