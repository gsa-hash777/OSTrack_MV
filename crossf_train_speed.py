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
    scaler,                              # GradScaler
    T_low: float = 0.25,                 # 需要救援阈值
    T_high: float = 0.50,                # 老师可靠阈值
    temperature: float = 0.7,
    top_k=None,
    w_l1: float = 1.0,
    w_delta: float = 0.05,
    w_gate: float = 0.10,                # gate 约束权重
    min_gate_good: float = 0.30,         # 好老师时 gate 至少这么大
    max_frames=None,
    seed: int = 0,
    verbose: bool = False,
    prefetch_workers: int = 4,
    loss_log: list = None,
):
    """
    两视角专用训练：
      - 只训练 fusion_net
      - 仅在 (ci < T_low && cj >= T_high) 时，对 i 进行融合与 box loss
      - 若 ci < T_low 但 cj < T_high：跳过该帧（不融合不训练）——与你推理一致
    """
    import os, random, cv2, torch
    from concurrent.futures import ThreadPoolExecutor

    random.seed(seed)

    view_dirs = discover_view_dirs(md_path)
    if len(view_dirs) == 0:
        if verbose:
            print(f"[Skip] no valid views in {md_path}")
        return 0.0, 0

    paths = [os.path.join(md_path, vd) for vd in view_dirs]
    img_dirs = [os.path.join(p, "img") for p in paths]
    img_files_per_view = [load_image_paths(d) for d in img_dirs]

    # 只支持两视角（你当前场景）
    if len(img_files_per_view) != 2:
        raise RuntimeError(f"Expected 2 views, got {len(img_files_per_view)} in {md_path}")

    num_frames = min(len(x) for x in img_files_per_view)
    if max_frames is not None:
        num_frames = min(num_frames, int(max_frames))

    optional_boxes = [read_initial_groundtruth(p) for p in paths]
    all_gt_boxes   = [read_all_groundtruth(p)    for p in paths]

    gt_frames = min(len(g) for g in all_gt_boxes)
    num_frames = min(num_frames, gt_frames)
    if num_frames <= 1:
        if verbose:
            print(f"[Skip] too short sequence {md_path}")
        return 0.0, 0

    # ---------- trackers ----------
    trackers = []
    for _ in img_dirs:
        tracker_instance = Tracker(tracker_name, tracker_param, "image_sequence")
        params = tracker_instance.get_parameters()
        if not hasattr(params, "debug"):
            params.debug = 0
        trk = OSTrack(params, "image_sequence")
        freeze_tracker(trk)  # 冻结 tracker.network 参数
        trackers.append(trk)

    # ---------- init frame 0 ----------
    first_frames_rgb = []
    for v in range(2):
        im_bgr = read_frame_bgr(img_files_per_view[v][0])
        im_rgb = bgr_to_rgb_uint8(im_bgr)
        first_frames_rgb.append(im_rgb)

    for v, trk in enumerate(trackers):
        trk.initialize(first_frames_rgb[v], {"init_bbox": optional_boxes[v]})

    # ---------- prefetch ----------
    pool = ThreadPoolExecutor(max_workers=prefetch_workers)

    def load_frame_two_view(t):
        frames = []
        for v in range(2):
            im_bgr = cv2.imread(img_files_per_view[v][t], cv2.IMREAD_COLOR)
            if im_bgr is None:
                raise RuntimeError(f"Failed to read image: {img_files_per_view[v][t]}")
            frames.append(cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB))
        return frames

    future = pool.submit(load_frame_two_view, 1)

    total_loss_val = 0.0
    total_steps = 0

    for t in range(1, num_frames):
        frames_rgb_list = future.result()
        if t + 1 < num_frames:
            future = pool.submit(load_frame_two_view, t + 1)

        # GT（原图xywh tensor）
        gt_img_xywh_list = [
            torch.tensor(all_gt_boxes[v][t], device=device, dtype=torch.float32)
            for v in range(2)
        ]

        # ============================================================
        # (1) track：no_grad 取 features/conf
        # ============================================================
        with torch.no_grad():
            out0 = trackers[0].track(frames_rgb_list[0])
            out1 = trackers[1].track(frames_rgb_list[1])

            sm0, f0 = out0.get("score_map", None), out0.get("cat_features", None)
            sm1, f1 = out1.get("score_map", None), out1.get("cat_features", None)

            if (sm0 is None) or (sm1 is None) or (f0 is None) or (f1 is None):
                continue

            c0 = float(sm0.max().item())
            c1 = float(sm1.max().item())

            features = [f0.to(device), f1.to(device)]
            confidences = [c0, c1]

        # 需要救援的视角
        need0 = (c0 < T_low)
        need1 = (c1 < T_low)

        # 只在 low-high 情况训练（与你推理一致）
        train_pairs = []
        if need0 and (c1 >= T_high):
            train_pairs.append((0, 1))
        if need1 and (c0 >= T_high):
            train_pairs.append((1, 0))

        if len(train_pairs) == 0:
            # 你提出的规则：另一视角 < 0.5 就跳过这帧不融合
            continue

        # ============================================================
        # (2) fusion + head + loss：enable_grad + AMP
        # ============================================================
        with torch.enable_grad():
            fusion_net.train()
            optimizer.zero_grad(set_to_none=True)

            loss_sum = None
            used = 0

            with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
                for (i, j) in train_pairs:
                    # i: 低于T_low 需要救援
                    # j: >=T_high 可靠老师

                    updated_feat, gate_i = fusion_net(
                        query_feat=features[i],  # student：需要梯度
                        key_feat_list=[features[j].detach()],  # teacher：显式不参与梯度
                        key_conf_list=[confidences[j]],
                        query_conf=confidences[i],
                        temperature=temperature,
                        top_k=top_k
                    )

                    # updated_feat, gate_i = fusion_net(
                    #     query_feat=features[i],
                    #     key_feat_list=[features[j]],          # 两视角：只有一个kv
                    #     key_conf_list=[confidences[j]],
                    #     query_conf=confidences[i],
                    #     temperature=temperature,
                    #     top_k=top_k
                    # )

                    # pred_box_crop（Tensor(4,)）
                    pred_box_crop = forward_head_and_box(trackers[i], updated_feat)

                    prev_state = torch.tensor(trackers[i].state, device=pred_box_crop.device, dtype=pred_box_crop.dtype)
                    gt_img_xywh = gt_img_xywh_list[i].to(device=pred_box_crop.device, dtype=pred_box_crop.dtype)
                    gt_box_crop = map_box_to_crop_torch(
                        gt_xywh=gt_img_xywh,
                        prev_state_xywh=prev_state,
                        search_size=trackers[i].params.search_size,
                        resize_factor=trackers[i].resize_factor
                    )

                    loss_l1 = torch.abs(pred_box_crop - gt_box_crop).mean()
                    loss_reg = feature_delta_loss(updated_feat, features[i])

                    # 好老师时 gate 不要太小，否则融合不起作用
                    loss_gate = torch.relu(min_gate_good - gate_i).mean()

                    loss = w_l1 * loss_l1 + w_delta * loss_reg + w_gate * loss_gate
                    loss_sum = loss if (loss_sum is None) else (loss_sum + loss)
                    used += 1

            if used == 0 or loss_sum is None or (not loss_sum.requires_grad):
                continue

            loss_sum = loss_sum / used

            scaler.scale(loss_sum).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss_val += float(loss_sum.detach().cpu().item())
            total_steps += 1
            if loss_log is not None:
                loss_log.append(float(loss_sum.detach().cpu().item()))

            if verbose and (t % 20 == 0):
                mean_conf = 0.5 * (c0 + c1)
                print(f"[{os.path.basename(md_path)} t={t:4d}] loss={float(loss_sum):.4f} pairs={train_pairs} conf=({c0:.3f},{c1:.3f}) mean_conf={mean_conf:.3f}")

    pool.shutdown(wait=True)
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

    # parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--T_low", type=float, default=0.25,
                        help="Low confidence threshold: ci < T_low triggers rescue training")
    parser.add_argument("--T_high", type=float, default=0.50,
                        help="High confidence threshold: teacher view must satisfy cj >= T_high")

    parser.add_argument("--w_gate", type=float, default=0.10,
                        help="Weight for gate regularization loss")
    parser.add_argument("--min_gate_good", type=float, default=0.30,
                        help="In good-teacher case, encourage gate >= this value")

    parser.add_argument("--prefetch_workers", type=int, default=4,
                        help="Number of threads for image prefetching")

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
    scaler = torch.cuda.amp.GradScaler(enabled=(device.startswith("cuda")))

    train_losses = []  # 用于画 loss 曲线
    epoch_losses = []

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
                scaler=scaler,
                T_low=args.T_low,
                T_high=args.T_high,
                temperature=args.temperature,
                top_k=(args.top_k if args.top_k > 0 else None),
                # loss 权重
                w_l1=args.w_l1,
                w_delta=args.w_delta,
                # gate 约束（推荐显式传入，方便你调参）
                w_gate=args.w_gate,
                min_gate_good=args.min_gate_good,

                max_frames=(args.max_frames if args.max_frames > 0 else None),
                seed=args.seed + ep,
                verbose=args.verbose,
                prefetch_workers=args.prefetch_workers,
                loss_log=train_losses,
            )

            ep_loss += loss_val
            ep_steps += steps

        # mean_loss = ep_loss / max(ep_steps, 1)
        # print(f"[Epoch {ep+1}/{args.epochs}] steps={ep_steps} mean_loss={mean_loss:.6f}")

        if ep_steps > 0:
            epoch_mean_loss = ep_loss / ep_steps
        else:
            epoch_mean_loss = float("nan")

        epoch_losses.append(epoch_mean_loss)

        print(f"[Epoch {ep + 1}/{args.epochs}] "
              f"steps={ep_steps}, mean_loss={epoch_mean_loss:.6f}")

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

    if len(epoch_losses) > 0:
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(epoch_losses) + 1),
                 epoch_losses, marker="o", linewidth=2)

        plt.xlabel("Epoch")
        plt.ylabel("Mean Loss")
        plt.title("Training Loss vs Epoch")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("train_loss_epoch.png", dpi=150)
        plt.show()

        print("Saved epoch-level loss curve to train_loss_epoch.png")

    print("Training finished.")


if __name__ == "__main__":
    main()
