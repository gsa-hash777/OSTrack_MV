import os
import sys
import argparse
import numpy as np
from lib.test.tracker.ostrack import OSTrack  # 修改为正确的导入
from lib.utils.box_ops import clip_box
import cv2 as cv,cv2
from lib.test.evaluation import Tracker
import xml.etree.ElementTree as ET
import json
from scipy.spatial import distance  # 用于计算中心点距离
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import deque
from utils_with_auc import aggregate_dataset_by_md,save_md_results,aggregate_whole_dataset,aggregate_dataset_by_view


# =========================
# Learnable fusion net (inference-time)
# =========================

def auc_success(iou_list, step=0.05):
    """AUC of success plot: mean success rate over IoU thresholds [0,1]."""
    if iou_list is None or len(iou_list) == 0:
        return 0.0
    thresholds = [i * step for i in range(int(1/step) + 1)]
    succ = []
    n = len(iou_list)
    for th in thresholds:
        succ.append(sum(1 for x in iou_list if x >= th) / n)
    return float(sum(succ) / len(succ))

class LiteCrossViewFusion(nn.Module):
    """Lightweight learnable cross-view fusion used only at inference/test."""
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

        # gate: (q_conf, mean_k, max_k, min_k) -> [0,1]
        self.gate_mlp = nn.Sequential(
            nn.Linear(4, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.scale = (bottleneck // num_heads) ** -0.5

    def _split(self, x):
        B, N, d = x.shape
        dh = d // self.h
        return x.view(B, N, self.h, dh).permute(0, 2, 1, 3)

    def _merge(self, x):
        B, h, N, dh = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B, N, h * dh)

    def forward(self, query_feat, key_feat_list, key_conf_list, query_conf, temperature=1.0, top_k=None):
        if len(key_feat_list) == 0:
            gate = torch.tensor(0.0, device=query_feat.device, dtype=query_feat.dtype)
            return query_feat, gate

        q0 = self.ln_q(query_feat)
        kv = torch.cat(key_feat_list, dim=1)
        kv0 = self.ln_kv(kv)

        q = self.q_proj(q0)
        k = self.k_proj(kv0)
        v = self.v_proj(kv0)

        qh = self._split(q)
        kh = self._split(k)
        vh = self._split(v)

        attn = (qh @ kh.transpose(-2, -1)) * (self.scale / max(float(temperature), 1e-6))

        if top_k is not None and top_k < attn.shape[-1]:
            topv, topi = torch.topk(attn, k=top_k, dim=-1)
            mask = torch.ones_like(attn, dtype=torch.bool)
            mask.scatter_(-1, topi, False)
            attn = attn.masked_fill(mask, float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        fused = attn @ vh
        fused = self._merge(fused)
        fused = fused + self.mlp(self.ln_out(fused))

        kc = torch.tensor(key_conf_list, device=query_feat.device, dtype=query_feat.dtype)
        stats = torch.stack([
            torch.tensor(float(query_conf), device=query_feat.device, dtype=query_feat.dtype),
            kc.mean() if kc.numel() > 0 else torch.tensor(0., device=query_feat.device, dtype=query_feat.dtype),
            kc.max() if kc.numel() > 0 else torch.tensor(0., device=query_feat.device, dtype=query_feat.dtype),
            kc.min() if kc.numel() > 0 else torch.tensor(0., device=query_feat.device, dtype=query_feat.dtype),
        ], dim=0)
        gate = torch.sigmoid(self.gate_mlp(stats)[0])
        fusedC = self.out_proj(fused)
        out = (1.0 - gate) * query_feat + gate * fusedC
        return out, gate

def load_fusion_net(ckpt_path, device="cuda"):
    fusion_net = LiteCrossViewFusion(dim=768, bottleneck=128, num_heads=4).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["fusion_net"] if isinstance(ckpt, dict) and "fusion_net" in ckpt else ckpt
    fusion_net.load_state_dict(state, strict=True)
    fusion_net.eval()
    return fusion_net

@torch.no_grad()
def fuse_low_confidence_views_learned(features, confidences, fusion_net,
                                   T_low=0.25, T_high=0.5,
                                   temperature=0.7, top_k=50):
    """Inference-time learned fusion (teacher->student).

    Rules:
      - High-confidence views (c >= T_high) are kept unchanged.
      - A view is considered 'student' only if c < T_low.
      - Student i is enhanced ONLY using teachers with c >= T_high.
      - If no teacher exists (e.g., 2-view case where the other view is mid/low), skip fusion for i.
    Returns:
      updated_features (list), gates (list[float|None]).
    """
    num_views = len(features)
    updated = list(features)
    gates = [None] * num_views

    # Precompute teacher indices
    teacher_idx = [j for j, c in enumerate(confidences) if c >= T_high]

    for i in range(num_views):
        ci = confidences[i]

        # keep high-confidence view unchanged
        if ci >= T_high:
            continue

        # only rescue very low confidence views
        if ci >= T_low:
            continue

        # teachers cannot include itself
        kv_idx = [j for j in teacher_idx if j != i]
        if len(kv_idx) == 0:
            # no reliable teacher -> do not fuse
            continue

        # IMPORTANT: detach teacher feats so teacher stays fixed (consistent with your training intent)
        key_feats = [features[j].detach() for j in kv_idx]
        key_confs = [confidences[j] for j in kv_idx]

        out_i, gate_i = fusion_net(
            query_feat=features[i],
            key_feat_list=key_feats,
            key_conf_list=key_confs,
            query_conf=ci,
            temperature=temperature,
            top_k=top_k,
        )
        updated[i] = out_i
        gates[i] = float(gate_i.detach().cpu().item())

    return updated, gates
@torch.no_grad()
def forward_head_and_update_state(tracker, feat_last):
    """Decode box from feat_last via forward_head, and update tracker.state (image xywh)."""
    out = tracker.network.forward_head(feat_last, None)
    pred_score_map = out['score_map']
    response = tracker.output_window * pred_score_map
    pred_boxes = tracker.network.box_head.cal_bbox(response, out['size_map'], out['offset_map'])
    pred_boxes = pred_boxes.view(-1, 4)
    pred_box = (pred_boxes.mean(dim=0) * tracker.params.search_size / tracker.resize_factor).tolist()
    tracker.state = clip_box(tracker.map_box_back(pred_box, tracker.resize_factor), tracker.H, tracker.W, margin=10)
    return tracker.state

def read_initial_groundtruth(video_path):
    """读取 groundtruth 文件的第一行作为初始框，自动处理逗号或空格分隔"""
    gt_file = os.path.join(video_path, 'groundtruth.txt')
    with open(gt_file, 'r') as f:
        line = f.readline().strip()  # 读取第一行
        # 先尝试用逗号分隔，如果失败则用空格分隔
        if ',' in line:
            init_rect = list(map(int, line.split(',')))
        else:
            init_rect = list(map(int, line.split()))
    return init_rect

def read_all_groundtruth(video_path):
    """读取所有帧的 groundtruth 框，自动处理逗号或空格分隔"""
    gt_boxes = []
    gt_file = os.path.join(video_path, 'groundtruth.txt')
    with open(gt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            # 先尝试用逗号分隔，如果失败则用空格分隔
            if ',' in line:
                rect = list(map(int, line.split(',')))
            else:
                rect = list(map(int, line.split()))
            gt_boxes.append(rect)
    return gt_boxes




class CameraCalibration:
    def __init__(self, root):
        self.root = root  # 根路径

        # 文件名对应相机的索引
        self.camera_files = ['camera1.json', 'camera2.json', 'camera3.json',
                             'camera5.json', 'camera6.json', 'camera7.json']

        # self.camera_files = ['camera2.json', 'camera3.json', 'camera4.json',
        #                      'camera5.json', 'camera6.json', 'camera7.json']

    def get_intrinsic_extrinsic_matrix(self, camera_i):
        # 获取相机 JSON 文件的路径
        json_file_path = os.path.join(self.root, 'calibrations', self.camera_files[camera_i])

        # 打开并读取 JSON 文件
        with open(json_file_path, 'r') as f:
            calibration_data = json.load(f)

        # 获取内参矩阵
        intrinsic_matrix = np.array(calibration_data['camera_intrinsic'], dtype=np.float32)
        # 获取外参矩阵
        extrinsic_matrix = np.array(calibration_data['camera_extrinsic'], dtype=np.float32)
        # extrinsic_matrix = np.array(calibration_data['global_camera'], dtype=np.float32)

        return intrinsic_matrix, extrinsic_matrix

def calculate_iou(boxA, boxB):
    # boxA 和 boxB 的格式为 [x, y, width, height]

    # 计算两个框的交集
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    # 计算交集面积
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # 计算两个框的面积
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    # 计算交并比 (IoU)
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

#满足任意视角数的拼接
def stack_frames(frames, layout=(2, 3)):
    # Calculate total frames needed for the layout
    total_frames = layout[0] * layout[1]

    # Create blank frame if needed for padding
    blank_frame = np.zeros_like(frames[0]) if frames else np.zeros((360, 640, 3), dtype=np.uint8)

    # Pad frames with blank frames to ensure we have enough frames
    frames += [blank_frame] * (total_frames - len(frames))

    # Stack frames as per the layout
    rows = []
    for i in range(layout[0]):
        row_frames = frames[i * layout[1]:(i + 1) * layout[1]]
        row_frames = [cv2.resize(f, (640, 360)) for f in row_frames]
        rows.append(cv2.hconcat(row_frames))
    return cv2.vconcat(rows)

def parse_bbox(bbox_str):
    """Parse bounding box string into a tuple of floats."""
    try:
        bbox = tuple(map(float, bbox_str.split(',')))
        if len(bbox) != 4:
            raise ValueError("Bounding box must be in the format 'x,y,w,h'.")
        return np.array(bbox)  # Convert to numpy array
    except ValueError as e:
        print(f"Error parsing bounding box: {e}")
        return None



def compute_confidence(score_map):
    """计算置信度"""
    return score_map.max().item()  # 假设使用 score_map 的最大值作为置信度


def cross_view_attention_fusion_v2(
        query_feat, key_feat_list, key_weights=None, num_heads=8, alpha=0.5,
        sim_threshold=None, top_k=None, temperature=1.0
):
    """
    Cross-attention 融合：增加相似度阈值、top-k 筛选、温度调节。

    Args:
        query_feat: Tensor (1, N, C)，主视角特征
        key_feat_list: List[Tensor]，其他视角特征
        key_weights: List[float]，每个 key_feat 的置信度权重
        num_heads: int，多头注意力头数
        alpha: float，融合比例
        sim_threshold: float，相似度阈值（None 表示不启用）
        top_k: int，只保留前 k 个相似度最高的 key（None 表示不启用）
        temperature: float，softmax 温度（<1 更尖锐）

    Returns:
        enhanced_feat: (1, N, C)
    """
    assert query_feat.dim() == 3
    N, C = query_feat.shape[1], query_feat.shape[2]

    if len(key_feat_list) == 0:
        return query_feat  # 没有其他视角

    key_feat = torch.cat(key_feat_list, dim=1)  # (1, N_total, C)
    value_feat = key_feat.clone()

    # 置信度加权
    if key_weights is not None:
        start = 0
        for i, w in enumerate(key_weights):
            seg_len = key_feat_list[i].shape[1]
            key_feat[:, start:start + seg_len, :] *= w
            value_feat[:, start:start + seg_len, :] *= w
            start += seg_len

    def split_heads(x, num_heads):
        return x.reshape(1, x.shape[1], num_heads, C // num_heads).permute(0, 2, 1, 3)

    def combine_heads(x):
        return x.permute(0, 2, 1, 3).reshape(1, N, C)

    Q = split_heads(query_feat, num_heads)
    K = split_heads(key_feat, num_heads)
    V = split_heads(value_feat, num_heads)

    scale = (C // num_heads) ** 0.5
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (scale * temperature)

    # ---- 相似度阈值筛选 ----
    if sim_threshold is not None:
        scores = scores.masked_fill(scores < sim_threshold, -1e9)

    # ---- Top-k 筛选 ----
    if top_k is not None and top_k < scores.shape[-1]:
        topk_values, topk_indices = torch.topk(scores, top_k, dim=-1)
        mask = torch.ones_like(scores, dtype=torch.bool)
        mask.scatter_(-1, topk_indices, False)
        scores = scores.masked_fill(mask, -1e9)

    attn = torch.softmax(scores, dim=-1)
    fused = torch.matmul(attn, V)
    fused = combine_heads(fused)
    enhanced_feat = (1 - alpha) * query_feat + alpha * fused
    return enhanced_feat


def fuse_low_confidence_views(
        features, confidences, score_threshold=0.5, num_heads=8, alpha=0.5,
        sim_threshold=None, top_k=None, temperature=1.0
):
    """
    仅替换置信度低的视角特征，用 cross-attention v2 融合结果增强。

    Args:
        features: List[Tensor], 每个为 (1, 320, 768)
        confidences: List[float], 每个视角的置信度
        score_threshold: float, 用于区分高低置信度视角
        num_heads: int, 注意力头数
        alpha: float, 融合强度
        sim_threshold: float, 相似度阈值
        top_k: int, top-k 筛选
        temperature: float, softmax 温度

    Returns:
        updated_features: List[Tensor], 更新后的特征列表
    """
    updated_features = []
    num_views = len(features)

    for i in range(num_views):
        query_feat = features[i]

        # 保持高置信度视角原特征
        if confidences[i] >= score_threshold:
            updated_features.append(query_feat)
            continue

        # 低置信度视角：用其他高置信度视角特征增强
        key_feat_list, key_weights = [], []
        for j in range(num_views):
            if j != i and confidences[j] >= score_threshold:
                key_feat_list.append(features[j])
                key_weights.append(confidences[j])

        if len(key_feat_list) == 0:
            updated_features.append(query_feat)
        else:
            fused_feat = cross_view_attention_fusion_v2(
                query_feat,
                key_feat_list,
                key_weights=key_weights,
                num_heads=num_heads,
                alpha=alpha,
                sim_threshold=sim_threshold,
                top_k=top_k,
                temperature=temperature
            )
            updated_features.append(fused_feat)

    return updated_features


def run_multiple_trackers(tracker_name, tracker_param, image_folders, optional_boxes, all_gt_boxes, debug=None, save_results=False,
                        use_learned_fusion=False, fusion_ckpt_path=None, device='cuda',
                        fusion_T_low=0.25, fusion_T_high=0.5, fusion_temperature=0.7, fusion_top_k=50):
    """Run multiple trackers on sequences of images in multiple folders."""

    # 创建跟踪器并加载参数
    trackers = []
    for _ in image_folders:
        tracker_instance = Tracker(tracker_name, tracker_param, "image_sequence")
        params = tracker_instance.get_parameters()  # 获取配置参数
        params.debug = debug if debug is not None else getattr(params, 'debug', 0)
        # params.debug = debug if debug is not None else False
        tracker = OSTrack(params, "image_sequence")  # 将参数传递给 OSTrack
        trackers.append(tracker)

    # 排序并加载每个文件夹中的图片文件名
    image_files_per_folder = [
        sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))])
        for folder in image_folders]

    # 验证每个文件夹是否包含图片
    for i, image_files in enumerate(image_files_per_folder):
        if not image_files:
            print(f"No images found in folder: {image_folders[i]}")
            return

    # ===== load trained fusion net for inference (optional) =====
    fusion_net = None
    if use_learned_fusion:
        assert fusion_ckpt_path is not None, "use_learned_fusion=True but fusion_ckpt_path is None"
        fusion_net = load_fusion_net(fusion_ckpt_path, device=device)

    # 初始化跟踪器
    for i, tracker in enumerate(trackers):

        first_image = cv.imread(image_files_per_folder[i][0])
        first_image_rgb = cv.cvtColor(first_image, cv.COLOR_BGR2RGB)
        tracker.initialize(first_image_rgb, {'init_bbox': optional_boxes[i]})


    # 定义六种不同的颜色 (BGR 颜色空间)
    colors = [(255, 0, 0),  # 蓝色
              (0, 255, 0),  # 绿色
              (0, 0, 255),  # 红色
              (255, 255, 0),  # 青色
              (255, 0, 255),  # 洋红
              (0, 255, 255)]  # 黄色

    frame_indices = [0] * 6  # 用于跟踪每个视频的帧索引
    gt_color = (255, 255, 255)  # 白色用于绘制GT框和点
    # 初始化跟踪指标数据
    tracking_metrics = {
        i: {
            'total_frames': 0,
            'successful_tracks': 0,
            'total_distance_error': 0,
            'initializations': 0,
            'precise_tracks': 0,
            'total_normalized_accuracy': 0
        }
        for i in range(len(image_files_per_folder))
    }
    iou_threshold = 0.5  # IOU 阈值，用于判断是否成功跟踪
    distance_threshold = 20  # 精确度阈值

    # 初始化 IoU 列表
    ious = []
    # 初始化每个视角的 IoU 列表
    num_views = 6
    ious_per_view = {i: [] for i in range(1, num_views + 1)}  # 假设视角从 1 开始，num_views 是视角总数

    # 处理图片序列
    for frame_idx in range(len(image_files_per_folder[0])):  # 假设所有文件夹有相同数量的图片
        frames = []

        foot_views = []  # 存储对应的视角编号
        foot_colors = []  # 存储对应的颜色

        confidences = []
        features = []
        score_maps = []

        for i, tracker in enumerate(trackers):
            image_file = image_files_per_folder[i][frame_idx]
            frame = cv.imread(image_file)
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            # 获取当前帧的 gt 框
            gt_rect = all_gt_boxes[i][frame_indices[i]]
            frames.append(frame)

            # 显示当前帧数在左上角
            frame_text = f"Frame: {frame_indices[i]}"
            cv2.putText(frame, frame_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

            # 判断如果gt_rect为[0, 0, 0, 0]，则跳过该帧的脚点计算，但仍然显示该帧
            if gt_rect == [0, 0, 0, 0]:

                # 跟踪器继续正常跟踪
                out = tracker.track(frame_rgb)
                bbox = [int(s) for s in out['target_bbox']]
                score_map = out.get("score_map", None)
                feature = out.get("cat_features", None)

                if score_map is not None:
                    # 计算最大值和熵值
                    max_score = score_map.max().item()
                    confidences.append(max_score)
                    score_maps.append(score_map)
                    features.append(feature)

                # 绘制跟踪框
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), colors[i], 3)

                # 绘制中心点
                center0 = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
                cv2.circle(frame, center0, 10, colors[i], -1)


                foot_views.append(i + 1)  # 记录视角编号
                foot_colors.append(colors[i])  # 记录颜色

                frame_indices[i] += 1  # 增加帧索引


            # 如果有 GT 框，继续 GT 的绘制和计算
            cv2.rectangle(frame, (gt_rect[0], gt_rect[1]),
                          (gt_rect[0] + gt_rect[2], gt_rect[1] + gt_rect[3]),
                          gt_color, 2)

            # 计算并绘制中心点
            gt_center = (gt_rect[0] + gt_rect[2] // 2, gt_rect[1] + gt_rect[3] // 2)
            cv2.circle(frame, gt_center, 10, gt_color, -1)  # 绘制中心点


            # 跟踪目标
            out = tracker.track(frame_rgb)
            bbox = [int(s) for s in out['target_bbox']]
            score_map = out.get("score_map", None)
            feature = out.get("cat_features", None)

            if score_map is not None:
                # 计算最大值和熵值
                max_score = score_map.max().item()
                # print(max_score)
                confidences.append(max_score)
                score_maps.append(score_map)
                features.append(feature)


            # 计算 IOU 和中心点距离
            iou = calculate_iou(bbox, gt_rect)
            tracking_metrics[i]['total_frames'] += 1
            if iou > iou_threshold:
                tracking_metrics[i]['successful_tracks'] += 1
            ious.append(iou)
            ious_per_view[i + 1].append(iou)  # 记录每个视角的 IoU

            # 计算精度（中心点距离误差）
            track_center = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
            center_distance = distance.euclidean(gt_center, track_center)
            tracking_metrics[i]['total_distance_error'] += center_distance

            # 跟踪精确判定
            if center_distance <= distance_threshold:
                tracking_metrics[i]['precise_tracks'] += 1

            # 归一化精确度计算
            gt_object_diagonal = np.sqrt(gt_rect[2] ** 2 + gt_rect[3] ** 2)
            normalized_accuracy = 1 - (
                    center_distance / gt_object_diagonal) if gt_object_diagonal > 0 else 0
            tracking_metrics[i]['total_normalized_accuracy'] += normalized_accuracy

            # 绘制跟踪框
            cv.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), colors[i], 3)
            cv.putText(frame, f'Cam {i + 1}', (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 3)
            # 计算中心点
            center0 = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
            # 画出中心点
            cv2.circle(frame, center0, 10, (0, 0, 255), -1)


            # 将帧添加到列表中
            frame_indices[i] += 1  # 增加帧索引


        score_threshold = 0.25


        # ===== Feature fusion (post-hoc) =====
        if use_learned_fusion and (fusion_net is not None):
            updated_features, gates = fuse_low_confidence_views_learned(
                features, confidences, fusion_net,
                T_low=fusion_T_low, T_high=fusion_T_high,
                temperature=fusion_temperature,
                top_k=fusion_top_k
            )
        else:
            updated_features = fuse_low_confidence_views(
                features,
                confidences,
                score_threshold=0.5,
                num_heads=8,
                alpha=0.5,
                sim_threshold=0.3,
                top_k=50,
                temperature=0.7
            )
            gates = None

        if updated_features is not None:
            # 将增强后的特征更新回低置信度的 tracker
            for i, tracker in enumerate(trackers):
                if confidences[i] < score_threshold:
                    tracker.feat_last = updated_features[i]
                    corrected_bbox = forward_head_and_update_state(tracker, tracker.feat_last)
                    # 可视化修正的跟踪框
                    color = (255, 165, 0)  # 用于区分修正框
                    cv2.rectangle(frames[i],
                                  (int(corrected_bbox[0]), int(corrected_bbox[1])),
                                  (int(corrected_bbox[0]) + int(corrected_bbox[2]), int(corrected_bbox[1]) + int(corrected_bbox[3])),
                                  color, 3)


    # 在所有帧处理完成后计算最终指标
    # 初始化用于计算平均值的变量
    total_views = len(tracking_metrics)
    sum_success_rate = 0
    sum_precise_rate = 0
    sum_average_accuracy = 0
    sum_normalized_accuracy = 0
    sum_avg_iou = 0
    valid_iou_views = 0

    for i, metrics in tracking_metrics.items():
        total_frames = metrics['total_frames']
        successful_tracks = metrics['successful_tracks']
        total_distance_error = metrics['total_distance_error']
        precise_tracks = metrics['precise_tracks']
        total_normalized_accuracy = metrics['total_normalized_accuracy']

        success_rate = successful_tracks / total_frames if total_frames > 0 else 0
        precise_rate = precise_tracks / total_frames if total_frames > 0 else 0
        average_accuracy = total_distance_error / successful_tracks if successful_tracks > 0 else float('inf')
        normalized_accuracy = total_normalized_accuracy / total_frames if total_frames > 0 else 0

        # 累加各项指标
        sum_success_rate += success_rate
        sum_precise_rate += precise_rate
        if average_accuracy != float('inf'):
            sum_average_accuracy += average_accuracy
        sum_normalized_accuracy += normalized_accuracy


    # 计算并打印每个视角的平均 IoU
    for view_id, ious in ious_per_view.items():
        if ious:
            avg_iou = sum(ious) / len(ious)
            sum_avg_iou += avg_iou
            valid_iou_views += 1

    # ================= 新增：结构化返回每个视角的指标 =================
    results = {}

    for view_id, ious in ious_per_view.items():
        if not ious:
            continue

        metrics = tracking_metrics[view_id - 1]
        total_frames = metrics['total_frames']

        SR = metrics['successful_tracks'] / total_frames if total_frames else 0
        PR20 = metrics['precise_tracks'] / total_frames if total_frames else 0
        NP = metrics['total_normalized_accuracy'] / total_frames if total_frames else 0
        IoU = sum(ious) / len(ious)
        AUC = auc_success(ious, step=0.05)

        results[view_id] = {
            "SR": SR,
            "PR20": PR20,
            "NP": NP,
            "IoU": IoU,
            "AUC": AUC
        }

    return results


def run_one_md(md, args, ROOT_DIR):
    base_dir = os.path.join(ROOT_DIR, md)

    # ===== 自动发现所有视角 =====
    view_dirs = sorted([
        d for d in os.listdir(base_dir)
        if d.startswith(md + "-") and
           os.path.isdir(os.path.join(base_dir, d))
    ])

    if len(view_dirs) == 0:
        print(f"[Skip] {md}: no views found")
        return None

    video_paths = []
    paths = []

    for vd in view_dirs:
        img_dir = os.path.join(base_dir, vd, "img")
        if not os.path.isdir(img_dir):
            print(f"[Skip] {md}: {vd} missing img/")
            return None
        video_paths.append(img_dir)
        paths.append(os.path.join(base_dir, vd))

    optional_boxes = [read_initial_groundtruth(p) for p in paths]
    all_gt_boxes = [read_all_groundtruth(p) for p in paths]

    results = run_multiple_trackers(
        args.tracker_name,
        args.tracker_param,
        video_paths,
        optional_boxes,
        all_gt_boxes,
        args.debug,
        args.save_results,
        use_learned_fusion=args.use_learned_fusion,
        fusion_ckpt_path=args.fusion_ckpt,
        device=args.device,
        fusion_T_low=args.fusion_T_low,
        fusion_T_high=args.fusion_T_high,
        fusion_temperature=args.fusion_temperature,
        fusion_top_k=args.fusion_top_k
    )

    save_md_results(md, results)
    return results

from multiprocessing import Pool, cpu_count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracker_name', type=str, default="ostrack")
    parser.add_argument('--tracker_param', type=str, default="vitb_256_mae_32x4_ep300")
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_learned_fusion', action='store_true', help='use trained fusion_net at test time')
    parser.add_argument('--fusion_ckpt', type=str, default='fusion_net.pth')
    parser.add_argument('--fusion_T_low', type=float, default=0.25, help='Student threshold: ci < T_low triggers fusion')
    parser.add_argument('--fusion_T_high', type=float, default=0.5, help='Teacher threshold: only views with c >= T_high can be KV')
    parser.add_argument('--fusion_temperature', type=float, default=0.7)
    parser.add_argument('--fusion_top_k', type=int, default=50)
    args = parser.parse_args()

    ROOT_DIR = "E:/BaiduNetdiskDownload/Two-MDOT/two_test1/two"

    md_list = sorted([
        d for d in os.listdir(ROOT_DIR)
        if d.startswith("md") and os.path.isdir(os.path.join(ROOT_DIR, d))
    ])

    print(f"Found {len(md_list)} md sequences")

    # with Pool(processes=min(cpu_count(), 4)) as pool:
    with Pool(processes=1) as pool:
        all_results = pool.starmap(
            run_one_md,
            [(md, args, ROOT_DIR) for md in md_list]
        )

    valid_md = []
    valid_results = []

    for md, res in zip(md_list, all_results):
        if res is not None:
            valid_md.append(md)
            valid_results.append(res)

    # aggregate_dataset_by_md(valid_md, valid_results)
    dataset_by_md = aggregate_dataset_by_md(valid_md, valid_results)
    dataset_avg = aggregate_whole_dataset(dataset_by_md)
    dataset_by_view = aggregate_dataset_by_view(valid_md, valid_results)

    print("===== Overall Dataset Results =====")
    for k, v in dataset_avg.items():
        print(f"{k}: {v:.4f}")

    # ===== Additional metric: AUC (success plot) =====
    # valid_results: list of dict(view_id -> metrics dict containing 'AUC' if enabled)
    all_auc = []
    for res in valid_results:
        if res is None:
            continue
        for vid, m in res.items():
            if isinstance(m, dict) and ("AUC" in m):
                all_auc.append(m["AUC"])
    if len(all_auc) > 0:
        print(f"AUC: {float(sum(all_auc)/len(all_auc)):.4f}")
    else:
        print("AUC: (not computed) - run with --use_learned_fusion to include AUC in per-view results")
    
    print("All evaluations finished.")

    print("===== Dataset Results by View =====")
    for v, m in dataset_by_view.items():
        extra = ""
        if 'AUC' in m:
            extra = f", AUC={m['AUC']:.4f}"
        print(f"View {v}: SR={m['SR']:.4f}, PR20={m['PR20']:.4f}, NP={m['NP']:.4f}, IoU={m['IoU']:.4f}{extra}")


if __name__ == '__main__':
    main()
