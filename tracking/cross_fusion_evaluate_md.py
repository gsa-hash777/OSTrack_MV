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
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import deque
from utils import aggregate_dataset_by_md,save_md_results,aggregate_whole_dataset,aggregate_dataset_by_view
from feature_fusion_visualization import visualize_multiview_feats,visualize_score_maps,visualize_scoremaps

# 定义相机内参和外参文件名列表
intrinsic_camera_matrix_filenames = ['intr_CVLab1.xml', 'intr_CVLab2.xml', 'intr_CVLab3.xml', 'intr_IDIAP1.xml',
                                     'intr_IDIAP2.xml', 'intr_IDIAP3.xml']
extrinsic_camera_matrix_filenames = ['extr_CVLab1.xml', 'extr_CVLab2.xml', 'extr_CVLab3.xml', 'extr_IDIAP1.xml',
                                     'extr_IDIAP2.xml', 'extr_IDIAP3.xml']


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

# 从图像坐标转换为世界坐标 (假设 z = 0)
def get_worldcoord_from_imagecoord(image_coord, intrinsic_mat, extrinsic_mat):
    # 计算投影矩阵，投影矩阵是内参矩阵与外参矩阵的乘积
    project_mat = intrinsic_mat @ extrinsic_mat
    # 将投影矩阵的第三列删除，并对其求逆
    project_mat = np.linalg.inv(np.delete(project_mat, 2, 1))
    # 在图像坐标后添加一行全为1的数组，用于齐次坐标转换
    image_coord = np.concatenate([image_coord, np.ones([1, image_coord.shape[1]])], axis=0)
    # 计算世界坐标
    world_coord = project_mat @ image_coord
    # 将世界坐标的前两行除以第三行，进行齐次坐标归一化
    world_coord = world_coord[:2, :] / world_coord[2, :]
    return world_coord

# 假设公共平面的大小为 480x1440
def project_to_common_plane(world_coord, scale=1, offset_x=0, offset_y=0):
    """
    将世界坐标 (x, y) 投影到公共平面上。
    参数:
    - world_coord: 世界坐标 (2, N)
    - scale: 缩放因子，将世界坐标放缩到公共平面大小
    - offset_x, offset_y: 用于将中心点移动到公共平面中央
    返回:
    - 投影到公共平面的点 (plane_x, plane_y)
    """
    plane_x = int(world_coord[0, 0] * scale + offset_x)
    plane_y = int(world_coord[1, 0] * scale + offset_y)
    return plane_x, plane_y

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

# 设置保存可视化图像的目录
VISUALIZATION_DIR = "score_map_response_6_0.7"
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

def visualize_score_maps(score_maps, frame_idx):
    """
    同时绘制并保存包含所有视角的完整画布
    """
    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(score_maps):
            # 处理数据
            score_map = score_maps[i].squeeze().cpu().numpy()

            # 绘制主图
            im = ax.imshow(score_map, cmap='jet')

            # 添加colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)  # 关键修改：使用fig.colorbar而不是plt.colorbar

            ax.set_title(f"View {i}")
            ax.axis("off")
        else:
            ax.axis("off")  # 隐藏多余子图

    plt.suptitle(f"Frame {frame_idx}")
    plt.tight_layout()

    # 一次性保存完整画布
    save_path = os.path.join(VISUALIZATION_DIR, f"frame_{frame_idx:04d}.png")
    plt.savefig(save_path, bbox_inches='tight')  # bbox_inches防止内容被裁剪
    # plt.show()
    plt.close(fig)  # 明确关闭当前figure释放内存

def check_if_outlier(current_bbox, previous_boxes):
    """
    判断当前检测框是否为离群点。
    历史平均判断离散
    Args:
        current_bbox (list): 当前检测框 [x, y, w, h]。
        previous_boxes (deque): 存储前几帧的检测框。

    Returns:
        bool: 如果是离群点，返回 True；否则返回 False。
    """
    if not previous_boxes:
        return False  # 如果没有历史框，则当前框不是离群点

    # 计算当前框与历史框平均值的差异
    avg_bbox = np.mean(previous_boxes, axis=0)
    bbox_diff = np.abs(np.array(current_bbox) - avg_bbox)
    # print(bbox_diff)
    # 设置一个阈值，根据应用调整
    threshold = [30, 30, 15, 15]  # x, y, w, h 的离群点阈值

    return any(bbox_diff > threshold)


def compute_confidence(score_map):
    """计算置信度"""
    return score_map.max().item()  # 假设使用 score_map 的最大值作为置信度

def compute_apce(score_map):
    """
    计算 APCE
    Args:
        score_map: 响应图 (PyTorch Tensor, shape=[H, W])
    Returns:
        apce: 平均峰值相关能量
    """
    score_map = score_map.squeeze()
    F_max = score_map.max()
    F_min = score_map.min()
    numerator = (F_max - F_min).pow(2)
    denominator = (score_map - F_min).pow(2).mean()
    apce = numerator / denominator if denominator > 0 else float('inf')
    return apce.item()



def generate_video(output_video_path, fps=1):
    """将 score_map 可视化图像合成视频"""
    images = sorted(os.listdir(VISUALIZATION_DIR))
    if not images:
        print("没有找到可视化图像，无法生成视频")
        return

    # 读取第一张图片以获取尺寸
    first_image_path = os.path.join(VISUALIZATION_DIR, images[0])
    frame = cv2.imread(first_image_path)
    h, w, _ = frame.shape

    # 设置视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 格式
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    for image_name in images:
        image_path = os.path.join(VISUALIZATION_DIR, image_name)
        frame = cv2.imread(image_path)
        video_writer.write(frame)  # 写入视频

    video_writer.release()
    print(f"Score Map 可视化视频已保存: {output_video_path}")


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


def run_multiple_trackers(tracker_name, tracker_param, image_folders, optional_boxes, all_gt_boxes, debug=None, save_results=False):
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

    # 初始化跟踪器
    for i, tracker in enumerate(trackers):
        first_image = cv.imread(image_files_per_folder[i][0])
        first_image_rgb = cv.cvtColor(first_image, cv.COLOR_BGR2RGB)
        tracker.initialize(first_image_rgb, {'init_bbox': optional_boxes[i]})



    calibration = CameraCalibration("D:/MultiViewDataset/wildtrack/WildTrackSeq1")
    # calibration = CameraCalibration("D:/MultiViewDataset/stairs2Seq4")

    # 加载相机内外参
    camera_matrices = [calibration.get_intrinsic_extrinsic_matrix(i) for i in range(6)]
    # 定义六种不同的颜色 (BGR 颜色空间)
    colors = [(255, 0, 0),  # 蓝色
              (0, 255, 0),  # 绿色
              (0, 0, 255),  # 红色
              (255, 255, 0),  # 青色
              (255, 0, 255),  # 洋红
              (0, 255, 255)]  # 黄色

    # 创建显示公共平面的窗口
    common_plane_size = (1080, 1920)
    # common_plane = np.ones((*common_plane_size, 3), dtype=np.uint8) * 255  # 创建一个公共平面窗口，白色背景
    common_plane = np.ones((*common_plane_size, 3), dtype=np.uint8)  # 创建一个公共平面窗口，黑色背景
    # 创建VideoWriter对象，保存视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4编码
    # output_video_common_plane = cv2.VideoWriter('common_plane_output_ostrack.mp4', fourcc, 2.0,
    #                                              (common_plane_size[1], common_plane_size[0]))
    # output_video_tracking = cv2.VideoWriter('multiview_tracking_output_fusion_D6.mp4', fourcc, 1.0,
    #                                          (1920, 720))  # 假设多视频窗口大小为 1920x1080

    # cv2.namedWindow('Multi-Video Tracking', cv2.WINDOW_NORMAL)
    # # cv2.namedWindow('Common Plane', cv2.WINDOW_NORMAL)
    # 创建显示窗口
    # display_name = 'Multi-Tracker Display'
    # cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
    # cv.resizeWindow(display_name, 1280, 720)

    frame_indices = [0] * 6  # 用于跟踪每个视频的帧索引
    gt_color = (255, 255, 255)  # 白色用于绘制GT框和点

    # previous_boxes = [None] * len(image_files_per_folder)  # 用于记录上一帧的跟踪框
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
    # score_threshold = 0.5
    # 初始化计时器和帧计数器
    start_time = time.time()
    frame_count = 0
    # 初始化 IoU 列表
    ious = []
    # 初始化每个视角的 IoU 列表
    num_views = 6
    ious_per_view = {i: [] for i in range(1, num_views + 1)}  # 假设视角从 1 开始，num_views 是视角总数

    #初始化 previous_boxes 为包含各视角 deque 的列表
    previous_boxes = [deque(maxlen=5) for _ in range(len(image_files_per_folder))]

    # 处理图片序列
    for frame_idx in range(len(image_files_per_folder[0])):  # 假设所有文件夹有相同数量的图片
        frames = []
        foot_points_gt = []  # 存储所有gt脚点坐标
        foot_points = []  # 存储所有脚点坐标

        foot_views = []  # 存储对应的视角编号
        foot_colors = []  # 存储对应的颜色

        confidences = []
        features = []
        score_maps = []

        # 逐个处理每个跟踪器和文件夹中的图片
        common_plane = np.ones((*common_plane_size, 3), dtype=np.uint8)  # 黑色背景
        intrinsic_matrices = [camera_matrix[0] for camera_matrix in camera_matrices]  # 提取所有内参
        extrinsic_matrices = [camera_matrix[1] for camera_matrix in camera_matrices]  # 提取所有外参

        for i, (tracker, (intrinsic_mat, extrinsic_mat)) in enumerate(zip(trackers, camera_matrices)):
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
                # feature = out.get("features", None)

                if score_map is not None:
                    # 计算最大值和熵值
                    max_score = score_map.max().item()
                    confidences.append(max_score)
                    score_maps.append(score_map)
                    features.append(feature)


                    # if max_score >= score_threshold:
                    #     print(f"视角 {i} 未遮挡，保留特征")
                    #     features.append(feature)
                    # else:
                    #     print(f"视角 {i} 遮挡，跳过特征")

                # 绘制跟踪框
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), colors[i], 3)

                # 绘制中心点
                center0 = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
                cv2.circle(frame, center0, 10, colors[i], -1)

                # 计算脚点
                foot_point = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3])

                # 投影脚点到 common_plane
                tracker_world_coord = get_worldcoord_from_imagecoord(np.array([[foot_point[0]], [foot_point[1]]]),
                                                                     intrinsic_mat, extrinsic_mat)
                plane_x, plane_y = project_to_common_plane(tracker_world_coord)
                foot_points.append([plane_x, plane_y])  # 收集投影后的脚点坐标

                foot_views.append(i + 1)  # 记录视角编号
                foot_colors.append(colors[i])  # 记录颜色

                # 绘制投影后的脚点在 common_plane 上
                cv2.circle(common_plane, (plane_x, plane_y), 10, colors[i], -1)

                # frames.append(frame)  # 添加帧到列表
                frame_indices[i] += 1  # 增加帧索引

                # # 保存当前帧的跟踪框，供下帧参考
                # if bbox is not None:
                #     previous_boxes[i] = bbox  # 在跟踪完成后更新 `previous_boxes`

                # 检测当前框是否为离群点
                is_outlier = check_if_outlier(bbox, previous_boxes[i])

                # 如果不是离群点，则存入 previous_boxes
                if not is_outlier:
                    previous_boxes[i].append(bbox)

                continue  # 跳过脚点计算

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
            # feature = out.get("features", None)


            if score_map is not None:
                # 计算最大值和熵值
                max_score = score_map.max().item()
                # print(max_score)
                confidences.append(max_score)
                score_maps.append(score_map)
                features.append(feature)


                # if max_score >= score_threshold:
                #     print(f"视角 {i} 未遮挡，保留特征")
                #     features.append(feature)
                #     # valid_trackers.append(tracker)
                # else:
                #     print(f"视角 {i} 遮挡，跳过特征")

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

            # # 在执行 previous_boxes 更新之前检查 bbox 是否有效
            # if bbox is not None:
            #     previous_boxes[i] = bbox

            # 检测当前框是否为离群点
            is_outlier = check_if_outlier(bbox, previous_boxes[i])

            # 如果不是离群点，则存入 previous_boxes
            if not is_outlier:
                previous_boxes[i].append(bbox)

        # if score_maps:
        #     visualize_score_maps(score_maps, frame_idx)
        score_threshold = 0.3

        # updated_features = fuse_low_confidence_views(
        #     features, confidences, score_threshold=0.5, num_heads=8, alpha=0.5
        # )

        updated_features = fuse_low_confidence_views(
            features,
            confidences,
            score_threshold=0.5,
            num_heads=8,
            alpha=0.5,
            sim_threshold=0.3,  # 相似度阈值
            top_k=50,  # 仅保留前k个最相似的patch
            temperature=0.7  # softmax温度
        )

        # visualize_multiview_feats(
        #     original_feats=features,
        #     fused_feats=updated_features,
        #     # images=rgb_images,
        #     # view_names=['Cam1', 'Cam2', 'Cam3'],
        #     # save_prefix='output/vis/frame_005_fusion'
        # )

        if updated_features is not None:
            # 将增强后的特征更新回低置信度的 tracker
            for i, tracker in enumerate(trackers):
                # tracker.feat_last = updated_features[i]
                # is_outlier = check_if_outlier(boxes[i], previous_boxes[i])
                # 如果置信度低于0.5，说明跟踪可能不稳定，替换特征
                # if confidences[i] < score_threshold and is_outlier:
                # if confidences[i] < score_threshold or is_outlier:
                if confidences[i] < score_threshold:
                # if is_outlier:
                #     print(f"替换视角 {i} 的特征",confidences[i])
                    # visualize_multiview_feats(
                    #     original_feats=features[i],
                    #     fused_feats=updated_features[i],
                    #     view=i
                    #     # images=rgb_images,
                    #     # view_names=['Cam1', 'Cam2', 'Cam3'],
                    #     # save_prefix='output/vis/frame_005_fusion'
                    # )
                    # 直接替换 `feat_last`
                    tracker.feat_last = updated_features[i]
                    # tracker.feat_last = tracker.feat_last * 0.5 + 0.5 * final_feature

                    # 将替换后的特征通过 forward_head 生成新的结果
                    out = tracker.network.forward_head(tracker.feat_last, None)
                    # print(out.keys())
                    # 更新输出到 tracker
                    # tracker.out_dict = out
                    pred_score_map = out['score_map']
                    response = tracker.output_window * pred_score_map

                    # visualize_scoremaps(
                    #     scoremaps_before=score_maps[i],
                    #     scoremaps_after=response,
                    #     view=i
                    #     # view_names=["Cam1", "Cam2", "Cam3"],
                    #     # save_prefix="output/scoremap/frame_015"
                    # )

                    pred_boxes = tracker.network.box_head.cal_bbox(response, out['size_map'], out['offset_map'])
                    pred_boxes = pred_boxes.view(-1, 4)
                    # Baseline: Take the mean of all pred boxes as the final result
                    pred_box = (pred_boxes.mean(
                        dim=0) * tracker.params.search_size / tracker.resize_factor).tolist()  # (cx, cy, w, h) [0,1]
                    # get the final box result
                    tracker.state = clip_box(tracker.map_box_back(pred_box, tracker.resize_factor), tracker.H, tracker.W, margin=10)
                    corrected_bbox = tracker.state
                    # print(corrected_bbox)
                    # 可视化修正的跟踪框
                    color = (255, 165, 0)  # 用于区分修正框
                    cv2.rectangle(frames[i],
                                  (int(corrected_bbox[0]), int(corrected_bbox[1])),
                                  (int(corrected_bbox[0]) + int(corrected_bbox[2]), int(corrected_bbox[1]) + int(corrected_bbox[3])),
                                  color, 3)

                # else:
                #     print(f"保持视角 {i} 的原始特征")

        # 生成 score_map 视频
        # generate_video("score_map_visualization.mp4", fps=1)

        # combined_frame = stack_frames(frames, layout=(2, 3))
        # # 显示合并的帧
        # cv.imshow(display_name, combined_frame)
        # # cv2.imshow('Common Plane', common_plane)
        # key = cv.waitKey(1)
        # if key == ord('q'):
        #     break
    # # # #
    #     # 将每一帧的common_plane和combined_frame写入视频
    #     # output_video_common_plane.write(common_plane)
    #     output_video_tracking.write(combined_frame)
    #
    # # 释放VideoWriter对象
    # # output_video_common_plane.release()
    # output_video_tracking.release()
    # # cv2.destroyAllWindows()
    #
    # cv.destroyAllWindows()

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

        # print(f"View {i + 1}:")
        # print(f"  Success Rate: {success_rate:.4f}")
        # print(f"  precise Rate: {precise_rate:.4f}")
        # print(f"  Average Accuracy (Distance Error): {average_accuracy:.4f} pixels")
        # print(f"  normalized_accuracy: {normalized_accuracy:.4f}")

    # 计算并打印每个视角的平均 IoU
    for view_id, ious in ious_per_view.items():
        if ious:
            avg_iou = sum(ious) / len(ious)
            sum_avg_iou += avg_iou
            valid_iou_views += 1
        #     print(f"Average IoU for View {view_id}: {avg_iou:.4f}")
        # else:
        #     print(f"No IoU data for View {view_id}.")

    # # 计算并打印所有视角的平均指标
    # if total_views > 0:
    #     print("\nAverage across all views:")
    #     print(f"SR: {sum_success_rate / total_views:.4f}")
    #     print(f"PR20: {sum_precise_rate / total_views:.4f}")
    #     if sum_average_accuracy > 0:  # 确保不是所有视角都是inf
    #         print(f"DE: {sum_average_accuracy / total_views:.4f}")
    #     print(f"NP: {sum_normalized_accuracy / total_views:.4f}")
    #
    # if valid_iou_views > 0:
    #     print(f"IOU: {sum_avg_iou / valid_iou_views:.4f}")

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

        results[view_id] = {
            "SR": SR,
            "PR20": PR20,
            "NP": NP,
            "IoU": IoU
        }

    return results


# def run_one_md(md, args, ROOT_DIR):
#     base_dir = os.path.join(ROOT_DIR, md)
#
#     video_paths = [
#         os.path.join(base_dir, f"{md}-1", "img"),
#         os.path.join(base_dir, f"{md}-2", "img")
#     ]
#     paths = [
#         os.path.join(base_dir, f"{md}-1"),
#         os.path.join(base_dir, f"{md}-2")
#     ]
#
#     if not all(os.path.exists(p) for p in video_paths + paths):
#         print(f"[Skip] {md} missing data")
#         return None
#
#     optional_boxes = [read_initial_groundtruth(p) for p in paths]
#     all_gt_boxes = [read_all_groundtruth(p) for p in paths]
#
#     results = run_multiple_trackers(
#         args.tracker_name,
#         args.tracker_param,
#         video_paths,
#         optional_boxes,
#         all_gt_boxes,
#         args.debug,
#         args.save_results
#     )
#
#     save_md_results(md, results)
#     return results

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
        args.save_results
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
    args = parser.parse_args()

    # ROOT_DIR = "E:/BaiduNetdiskDownload/Two-MDOT/two_test1/two"
    ROOT_DIR = "E:/BaiduNetdiskDownload/Three-MDOT/three_test1/three"

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

    print("All evaluations finished.")

    print("===== Dataset Results by View =====")
    for v, m in dataset_by_view.items():
        print(f"View {v}: SR={m['SR']:.4f}, PR20={m['PR20']:.4f}, NP={m['NP']:.4f}, IoU={m['IoU']:.4f}")


if __name__ == '__main__':
    main()
