from scipy.spatial.distance import mahalanobis
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
# 定义相机内参和外参文件名列表
intrinsic_camera_matrix_filenames = ['intr_CVLab1.xml', 'intr_CVLab2.xml', 'intr_CVLab3.xml', 'intr_IDIAP1.xml',
                                     'intr_IDIAP2.xml', 'intr_IDIAP3.xml']
extrinsic_camera_matrix_filenames = ['extr_CVLab1.xml', 'extr_CVLab2.xml', 'extr_CVLab3.xml', 'extr_IDIAP1.xml',
                                     'extr_IDIAP2.xml', 'extr_IDIAP3.xml']

# def read_groundtruth(video_path):
#     groundtruth_path = os.path.join(video_path, 'groundtruth.txt')
#     with open(groundtruth_path, 'r') as f:
#         # 读取第一行，并将其解析为整数列表
#         line = f.readline().strip()
#         rect = list(map(int, line.split()))  #格式为 x y width height
#     return rect

def read_initial_groundtruth(video_path):
    """读取 groundtruth 文件的第一行作为初始框"""
    gt_file = os.path.join(video_path, 'groundtruth.txt')
    with open(gt_file, 'r') as f:
        line = f.readline().strip()  # 读取第一行
        init_rect = list(map(int, line.split()))  # 格式为 "x y width height"
    return init_rect

def read_all_groundtruth(video_path):
    """读取所有帧的 groundtruth 框"""
    gt_boxes = []
    gt_file = os.path.join(video_path, 'groundtruth.txt')
    with open(gt_file, 'r') as f:
        for line in f:
            rect = list(map(int, line.strip().split()))  # 假设格式为 "x y width height"
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

# 从世界坐标转换为图像坐标
def get_imagecoord_from_worldcoord(world_coord, intrinsic_mat, extrinsic_mat):
    # 计算投影矩阵，投影矩阵是内参矩阵与外参矩阵的乘积
    project_mat = intrinsic_mat @ extrinsic_mat
    # 将投影矩阵的第三列删除
    project_mat = np.delete(project_mat, 2, 1)
    # 在世界坐标后添加一行全为1的数组，用于齐次坐标转换
    world_coord = np.concatenate([world_coord, np.ones([1, world_coord.shape[1]])], axis=0)
    # 计算图像坐标
    image_coord = project_mat @ world_coord
    # 将图像坐标的前两行除以第三行，进行齐次坐标归一化
    image_coord = image_coord[:2, :] / image_coord[2, :]
    return image_coord


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
        # 获取畸变系数
        dist_coeffs = np.array(calibration_data['dist'], dtype=np.float32)

        return intrinsic_matrix, extrinsic_matrix, dist_coeffs


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

# 计算欧几里得距离
def calculate_euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

#距离最进的三个点的中心
def calculate_closest_three_points_midpoint(foot_points, foot_colors, foot_views, frame_index):
    min_distance = float('inf')
    closest_points = []
    closest_colors = []
    closest_views = []

    # 循环遍历所有点对的组合，寻找距离最近的三个点
    for i in range(len(foot_points)):
        for j in range(i + 1, len(foot_points)):
            for k in range(j + 1, len(foot_points)):
                # 计算点对之间的欧氏距离
                dist_ij = calculate_euclidean_distance(foot_points[i], foot_points[j])
                dist_ik = calculate_euclidean_distance(foot_points[i], foot_points[k])
                dist_jk = calculate_euclidean_distance(foot_points[j], foot_points[k])
                total_distance = dist_ij + dist_ik + dist_jk

                # 如果这三个点的总距离比当前最小距离更小，则更新
                if total_distance < min_distance:
                    min_distance = total_distance
                    closest_points = [foot_points[i], foot_points[j], foot_points[k]]
                    closest_colors = [foot_colors[i], foot_colors[j], foot_colors[k]]
                    closest_views = [foot_views[i], foot_views[j], foot_views[k]]

    # 计算最近三个点的中心点
    midpoint = np.mean(closest_points, axis=0).astype(int)

    # 打印调试信息
    # print(f"Frame {frame_index}: Closest three points are from views {closest_views[0]}, {closest_views[1]}, {closest_views[2]}, "
    #       f"with colors {closest_colors[0]}, {closest_colors[1]}, {closest_colors[2]}")
    # print(f"Points: {closest_points[0]} and {closest_points[1]} and {closest_points[2]}")
    # print(f"Midpoint: {midpoint}")

    return midpoint,closest_views

#计算离散点
def detect_outliers_mahalanobis(foot_points, foot_colors, foot_views, frame_index, m=1, k_base=1, std_threshold=50):
    """
    使用马氏距离检测离群点，根据当前帧的协方差矩阵动态调整 k 值。
    """
    foot_points_array = np.array(foot_points)

    # 计算质心
    centroid = np.mean(foot_points_array, axis=0)

    # 协方差矩阵及其逆
    try:
        cov_matrix = np.cov(foot_points_array, rowvar=False)
        inv_cov_matrix = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        # 协方差矩阵不可逆时使用伪逆
        inv_cov_matrix = np.linalg.pinv(np.cov(foot_points_array, rowvar=False))

    # 计算每个脚点的马氏距离
    distances = [mahalanobis(point, centroid, inv_cov_matrix) for point in foot_points_array]

    # 计算中间点（如三点最近中点）
    midpoint, closest_views = calculate_closest_three_points_midpoint(foot_points, foot_colors, foot_views, frame_index)

    # 计算每个点到中间点的马氏距离（也可以只对centroid进行判断）
    distances1 = [mahalanobis(point, midpoint, inv_cov_matrix) for point in foot_points_array]

    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    # 动态调整 k 值
    if std_distance > std_threshold:
        k = min(1.75, k_base + 0.25 * (std_distance / mean_distance))
    else:
        k = max(0.15, k_base - 0.25 * (std_distance / mean_distance))

    T = m * mean_distance + k * std_distance

    outliers = []
    for i, distance in enumerate(distances1):
        if distance > T:
            outliers.append((foot_views[i], foot_colors[i], distance))

    # if outliers:
        # print(f"Frame {frame_index}: Found {len(outliers)} outliers with Mahalanobis k={k:.2f}")
        # for outlier in outliers:
        #     print(f"Outlier in view {outlier[0]} with color {outlier[1]} and distance {outlier[2]:.2f}")

    return centroid, T, outliers

def detect_outliers_adaptive(foot_points, foot_colors, foot_views, frame_index, m=1, k_base=1, std_threshold=50):
    """
    根据当前帧的标准差动态调整 k 值以检测离群点。
    """
    foot_points_array = np.array(foot_points)
    # 计算质心 (所有脚点的均值)
    centroid = np.mean(foot_points_array, axis=0)
    # 计算每个脚点到质心的距离
    distances = np.linalg.norm(foot_points_array - centroid, axis=1)
    midpoint,closest_views = calculate_closest_three_points_midpoint(foot_points, foot_colors, foot_views, frame_index)
    # 计算每个脚点到中心的距离
    distances1 = np.linalg.norm(foot_points_array - midpoint, axis=1)
    # print(distances1)
    # 计算距离的均值和标准差
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    # 根据当前标准差调整 k 值，设置上下限
    if std_distance > std_threshold:
        k = min(1.75, k_base + 0.25 * (std_distance / mean_distance))
    else:
        k = max(0.15, k_base - 0.25 * (std_distance / mean_distance))
    # 设定阈值 T = 均值 + k * 标准差
    T = m * mean_distance + k * std_distance

    # 输出离群点的信息
    outliers = []
    for i, distance in enumerate(distances1):
        if distance > T:
            outliers.append((foot_views[i], foot_colors[i], distance))
    # 打印离群点的信息
    # if outliers:
    #     print(f"Frame {frame_index}: Found {len(outliers)} outliers with k={k:.2f}")
    #     for outlier in outliers:
    #         print(f"Outlier in view {outlier[0]} with color {outlier[1]} and distance {outlier[2]:.2f}")

    return centroid, T, outliers

def correct_outliers_mean(foot_points,foot_colors, foot_views, frame_index, intrinsic_matrices, extrinsic_matrices, frames, outliers,trackers, previous_boxes,tracking_metrics, dist_coeffs):
    """
        使用最近三个点的几何中心来修正离群点，并调整跟踪器。

    """
    for outlier_view, outlier_color, outlier_point in outliers:
        outlier_index = foot_views.index(outlier_view)
        # 使用最近三个点的几何中心来修正离群点
        midpoint, closest_views = calculate_closest_three_points_midpoint(foot_points, foot_colors, foot_views, frame_index)
        # print(f"midpoint{midpoint}")
        # 获取世界坐标中的修正点
        midpoint_world_coord = np.array([[midpoint[0]], [midpoint[1]]])
        # print(f"midpoint_world_coord{midpoint_world_coord}")
        # 获取该离群点对应视角的内参和外参矩阵
        intrinsic_mat = intrinsic_matrices[outlier_view - 1]  # 假设视角编号从1开始
        extrinsic_mat = extrinsic_matrices[outlier_view - 1]
        # 将修正点从世界坐标逆投影回图像坐标
        # image_coord = get_imagecoord_from_worldcoord(midpoint_world_coord, intrinsic_mat, extrinsic_mat, dist_coeffs)
        image_coord = get_imagecoord_from_worldcoord(midpoint_world_coord, intrinsic_mat, extrinsic_mat)

        # print(image_coord)
        # print(f"image_coord{image_coord}")
        # print(intrinsic_mat, extrinsic_mat)
        # 将 image_coord 转换为整数，并确保是一个元组
        image_coord = (int(image_coord[0, 0]), int(image_coord[1, 0]))
        # 绘制修正后的点在对应视角图像中
        cv2.circle(frames[outlier_index], image_coord, 10, (0, 0, 255), -1)  # 绿色表示修正点
        # Calculate average box size over the last five frames for this view
        if previous_boxes[outlier_index]:
            avg_width = int(np.mean([box[2] for box in previous_boxes[outlier_index]]))
            avg_height = int(np.mean([box[3] for box in previous_boxes[outlier_index]]))
        else:
            avg_width, avg_height = 0, 0  # Fallback values if no history exists

        # Use avg_width and avg_height in place of previous width and height
        corrected_bbox = (image_coord[0] - avg_width // 2, image_coord[1] - avg_height, avg_width, avg_height)

        # 直接更新跟踪器的 state 属性来修改跟踪框
        trackers[outlier_index].state = corrected_bbox

        # 更新 previous_boxes 以保持框位置同步
        previous_boxes[outlier_view-1].append(corrected_bbox)

        # 可视化修正的跟踪框
        color = (255,165,0)  # 用于区分修正框
        cv2.rectangle(frames[outlier_index],
                      (corrected_bbox[0], corrected_bbox[1]),
                      (corrected_bbox[0] + corrected_bbox[2], corrected_bbox[1] + corrected_bbox[3]),
                      color, 3)

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

    # 设置一个阈值，根据应用调整
    threshold = [30, 30, 15, 15]  # x, y, w, h 的离群点阈值
    # threshold = [50, 50, 25, 25]  # x, y, w, h 的离群点阈值

    return any(bbox_diff > threshold)

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

def update_foot_point_from_bbox(
    corrected_bbox, view_idx,
    intrinsic_matrices, extrinsic_matrices,
    foot_points, foot_views
):
    foot_x = corrected_bbox[0] + corrected_bbox[2] / 2
    foot_y = corrected_bbox[1] + corrected_bbox[3]
    foot_img = np.array([[foot_x], [foot_y]])

    world = get_worldcoord_from_imagecoord(
        foot_img,
        intrinsic_matrices[view_idx],
        extrinsic_matrices[view_idx]
    )

    plane_x, plane_y = project_to_common_plane(world)

    view_id = view_idx + 1
    if view_id in foot_views:
        idx = foot_views.index(view_id)
        foot_points[idx] = [plane_x, plane_y]


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



    calibration = CameraCalibration("D:/MultiViewDataset/wildtrack/wildtrack_initial/WildTrackSeq1")
    # # 加载相机内外参
    # camera_matrices = [calibration.get_intrinsic_extrinsic_matrix(i) for i in range(6)]
    # 加载相机内外参和畸变系数
    camera_calibrations = [calibration.get_intrinsic_extrinsic_matrix(i) for i in range(6)]
    camera_matrices = [(calib[0], calib[1]) for calib in camera_calibrations]
    dist_coeffs_list = [calib[2] for calib in camera_calibrations]

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
    # output_video_common_plane = cv2.VideoWriter('common_plane_output_ostrack_sizetest1.mp4', fourcc, 1.0,
    #                                              (common_plane_size[1], common_plane_size[0]))
    # output_video_tracking = cv2.VideoWriter('multiview_tracking_data2_without-distortion.mp4', fourcc, 1.0,
    #                                          (1920, 720))  # 假设多视频窗口大小为 1920x1080
    #
    # # cv2.namedWindow('Multi-Video Tracking', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('Common Plane', cv2.WINDOW_NORMAL)
    # # # 创建显示窗口
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
    distance_threshold = 20 # 精确度阈值

    # 初始化计时器和帧计数器
    start_time = time.time()
    frame_count = 0
    # 初始化 IoU 列表
    ious = []
    # 初始化每个视角的 IoU 列表
    num_views = 6
    ious_per_view = {i: [] for i in range(1, num_views + 1)}  # 假设视角从 1 开始，num_views 是视角总数

    #############################################
    #初始化 previous_boxes 为包含各视角 deque 的列表
    previous_boxes = [deque(maxlen=5) for _ in range(len(image_files_per_folder))]

    # 处理图片序列
    for frame_idx in range(len(image_files_per_folder[0])):  # 假设所有文件夹有相同数量的图片
        frames = []
        foot_points_gt = []  # 存储所有gt脚点坐标
        foot_points = []    # 存储所有脚点坐标
        foot_views = []  # 存储对应的视角编号
        foot_colors = []  # 存储对应的颜色

        confidences = []
        features = []
        score_maps = []


        # 逐个处理每个跟踪器和文件夹中的图片
        common_plane = np.ones((*common_plane_size, 3), dtype=np.uint8)        # 黑色背景
        intrinsic_matrices = [camera_matrix[0] for camera_matrix in camera_matrices]  # 提取所有内参
        extrinsic_matrices = [camera_matrix[1] for camera_matrix in camera_matrices]  # 提取所有外参


        for i, (tracker, (intrinsic_mat, extrinsic_mat),dist_coeffs) in enumerate(zip(trackers, camera_matrices, dist_coeffs_list)):
            image_file = image_files_per_folder[i][frame_idx]
            frame = cv.imread(image_file)
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            # 获取当前帧的 gt 框
            gt_rect = all_gt_boxes[i][frame_indices[i]]

            # 显示当前帧数在左上角
            frame_text = f"Frame: {frame_indices[i]}"
            cv2.putText(frame, frame_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

            # 在 common_plane 左上角显示当前帧数
            common_plane_text = f"Frame: {frame_indices[i]}"
            cv2.putText(common_plane, common_plane_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

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
                    # print(max_score)
                    confidences.append(max_score)
                    score_maps.append(score_map)
                    features.append(feature)



                # 绘制跟踪框
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), colors[i], 3)

                # 绘制中心点
                center0 = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
                cv2.circle(frame, center0, 10, colors[i], -1)

                # 计算脚点
                foot_point = (bbox[0]+ bbox[2] // 2, bbox[1] + bbox[3])

                # 投影脚点到 common_plane
                # tracker_world_coord = get_worldcoord_from_imagecoord(np.array([[foot_point[0]], [foot_point[1]]]),
                #                                                      intrinsic_mat, extrinsic_mat, dist_coeffs)
                tracker_world_coord = get_worldcoord_from_imagecoord(np.array([[foot_point[0]], [foot_point[1]]]),
                                                                     intrinsic_mat, extrinsic_mat)
                plane_x, plane_y = project_to_common_plane(tracker_world_coord)

                foot_points.append([plane_x, plane_y])  # 收集投影后的脚点坐标

                foot_views.append(i + 1)  # 记录视角编号
                foot_colors.append(colors[i])  # 记录颜色

                # 绘制投影后的脚点在 common_plane 上
                cv2.circle(common_plane, (plane_x, plane_y), 10, colors[i], -1)

                frames.append(frame)  # 添加帧到列表
                frame_indices[i] += 1  # 增加帧索引

                # # 检测当前框是否为离群点
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

            # 计算GT脚点
            foot_point = (gt_rect[0] + gt_rect[2] // 2, gt_rect[1] + gt_rect[3])  # 脚点坐标
            cv2.circle(frame, foot_point, 10, gt_color, -1)  # 绘制脚点

            # 投影 GT 脚点到 common_plane
            # gt_world_coord = get_worldcoord_from_imagecoord(np.array([[foot_point[0]], [foot_point[1]]]),
            #                                                 intrinsic_mat, extrinsic_mat, dist_coeffs)
            gt_world_coord = get_worldcoord_from_imagecoord(np.array([[foot_point[0]], [foot_point[1]]]),
                                                            intrinsic_mat, extrinsic_mat)
            plane_x, plane_y = project_to_common_plane(gt_world_coord)
            foot_points_gt.append([plane_x, plane_y])  # 收集投影后的GT脚点坐标
            # cv2.circle(common_plane, (plane_x, plane_y), 10, gt_color, -1)  # 在公共平面上绘制 GT 脚点

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



            # 绘制跟踪框
            cv.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), colors[i], 3)
            cv.putText(frame, f'Cam {i + 1}', (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 3)

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

            # 计算中心点
            center0 = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
            # 画出中心点
            cv2.circle(frame, center0, 10, (0, 0, 255), -1)
            # 脚点
            center = np.array([[bbox[0] + bbox[2] // 2], [bbox[1] + bbox[3]]])

            # world_coord = get_worldcoord_from_imagecoord(center, intrinsic_mat, extrinsic_mat, dist_coeffs)
            world_coord = get_worldcoord_from_imagecoord(center, intrinsic_mat, extrinsic_mat)


            # 将投影到公共平面的点可视化 (假设 x, y 映射到图像坐标)
            plane_x, plane_y = project_to_common_plane(world_coord)

            foot_points.append([plane_x, plane_y])  # 收集投影后的脚点坐标

            foot_views.append(i + 1)  # 记录视角编号
            foot_colors.append(colors[i])  # 记录颜色
            cv2.circle(common_plane, (plane_x, plane_y), 10, colors[i], -1)

            # # 打印坐标和颜色调试
            # print(f"Camera {i}: plane_x = {plane_x}, plane_y = {plane_y}")
            # print(f"Drawing point for camera {i} with color {colors[i]}")

            # 检查坐标是否在公共平面范围内，再绘制
            # if 0 <= plane_x < 1920 and 0 <= plane_y < 1080:
            cv2.circle(common_plane, (plane_x, plane_y), 10, colors[i], -1)  # 使用每个摄像机的颜色绘制点

            # 将帧添加到列表中
            frames.append(frame)
            frame_indices[i] += 1  # 增加帧索引

            # 检测当前框是否为离群点
            is_outlier = check_if_outlier(bbox, previous_boxes[i])

            # 如果不是离群点，则存入 previous_boxes
            if not is_outlier:
                previous_boxes[i].append(bbox)


        score_threshold = 0.3

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

        if updated_features is not None:
            # 将增强后的特征更新回低置信度的 tracker
            for i, tracker in enumerate(trackers):
                if confidences[i] < score_threshold:
                    # 直接替换 `feat_last`
                    tracker.feat_last = updated_features[i]
                    # 将替换后的特征通过 forward_head 生成新的结果
                    out = tracker.network.forward_head(tracker.feat_last, None)
                    # print(out.keys())
                    pred_score_map = out['score_map']
                    response = tracker.output_window * pred_score_map

                    pred_boxes = tracker.network.box_head.cal_bbox(response, out['size_map'], out['offset_map'])
                    pred_boxes = pred_boxes.view(-1, 4)
                    # Baseline: Take the mean of all pred boxes as the final result
                    pred_box = (pred_boxes.mean(
                        dim=0) * tracker.params.search_size / tracker.resize_factor).tolist()  # (cx, cy, w, h) [0,1]
                    # get the final box result
                    tracker.state = clip_box(tracker.map_box_back(pred_box, tracker.resize_factor), tracker.H, tracker.W, margin=10)
                    corrected_bbox = tracker.state
                    # print(corrected_bbox)
                    #更新 foot_points
                    update_foot_point_from_bbox(
                        corrected_bbox,
                        i,
                        intrinsic_matrices,
                        extrinsic_matrices,
                        foot_points,
                        foot_views
                    )
                    # 可视化修正的跟踪框
                    color = (255, 165, 0)  # 用于区分修正框
                    cv2.rectangle(frames[i],
                                  (int(corrected_bbox[0]), int(corrected_bbox[1])),
                                  (int(corrected_bbox[0]) + int(corrected_bbox[2]), int(corrected_bbox[1]) + int(corrected_bbox[3])),
                                  color, 3)

        # # 在每一帧结束时更新帧计数器
        # frame_count += 1
        # # 计算 FPS
        # elapsed_time = time.time() - start_time
        # if elapsed_time > 0:
        #     fps = frame_count / elapsed_time
        #     # 打印或显示 FPS
        #     print(f"Current FPS: {fps:.2f}")
        if foot_points:
            # 计算最近三个点的中心点并可视化
            midpoint,closest_views = calculate_closest_three_points_midpoint(foot_points, foot_colors, foot_views, frame_indices[0])

            # 在 common_plane 上绘制中心点
            cv2.circle(common_plane, (midpoint[0], midpoint[1]), 10, (255, 105, 180), -1)  # 绘制粉色中心点
            # centroid, T, outliers = detect_outliers_adaptive(foot_points, foot_colors, foot_views, frame_indices[0])
            centroid, T, outliers = detect_outliers_mahalanobis(foot_points, foot_colors, foot_views, frame_indices[0])


            # 修正离群点并将修正后的点绘制在图像上
            correct_outliers_mean(foot_points,foot_colors, foot_views, frame_indices[0], intrinsic_matrices,
                             extrinsic_matrices,
                             frames, outliers, trackers, previous_boxes, tracking_metrics, dist_coeffs)


        # if foot_points:
        #     # 检测离群点并获取质心
        #     centroid, T, outliers = detect_outliers_adaptive(foot_points, foot_colors, foot_views, frame_indices[0])

            # # 在 common_plane 上绘制质心
            # centroid = centroid.astype(int)
            # cv2.circle(common_plane, (centroid[0], centroid[1]), 10, (255, 255, 255), -1)  # 绘制白色质心

            # 在 common_plane 上标记离群点
            for outlier_view, outlier_color, _ in outliers:
                outlier_index = foot_views.index(outlier_view)
                outlier_point = foot_points[outlier_index]
                cv2.circle(common_plane, (outlier_point[0], outlier_point[1]), 5, (0, 0, 0), -1)  # 绘制离群点

        if foot_points_gt:
            foot_points_array = np.array(foot_points_gt)
            foot_center = foot_points_array.mean(axis=0).astype(int)  # 计算中心点 (x, y)
            # 在公共平面上绘制中心点
            cv2.circle(common_plane, (foot_center[0], foot_center[1]), 10, (255, 255, 255), -1)  # 绘制白色中心点


        # # 按 2x3 布局拼接帧
        # row1 = np.hstack(frames[:3])  # 前三帧
        # row2 = np.hstack(frames[3:])  # 后三帧
        # combined_frame = np.vstack([row1, row2])  # 垂直拼接两行
    #
        # combined_frame = stack_frames(frames, layout=(2, 3))
        # # 显示合并的帧
        # cv.imshow(display_name, combined_frame)
        # cv2.imshow('Common Plane', common_plane)
        # key = cv.waitKey(1)
        # if key == ord('q'):
        #     break
    # # #
        # 将每一帧的common_plane和combined_frame写入视频
        # output_video_common_plane.write(common_plane)
        # output_video_tracking.write(combined_frame)

    # 释放VideoWriter对象
    # output_video_common_plane.release()
    # output_video_tracking.release()
    # cv2.destroyAllWindows()
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

    # 计算并打印所有视角的平均指标
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


def run_one_sequence(seq_name, args, ROOT_DIR, dataset_type):
    seq_dir = os.path.join(ROOT_DIR, seq_name)

    if dataset_type == "wildtrack":
        view_infos = parse_wildtrack_sequence(seq_dir)
    elif dataset_type == "md":
        view_infos = parse_md_sequence(seq_dir, seq_name)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    if len(view_infos) == 0:
        print(f"[Skip] {seq_name}: no valid views")
        return None

    video_paths = [v["img_dir"] for v in view_infos]
    anno_paths = [v["anno_dir"] for v in view_infos]

    optional_boxes = [read_initial_groundtruth(p) for p in anno_paths]
    all_gt_boxes = [read_all_groundtruth(p) for p in anno_paths]

    results = run_multiple_trackers(
        args.tracker_name,
        args.tracker_param,
        video_paths,
        optional_boxes,
        all_gt_boxes,
        args.debug,
        args.save_results
    )

    save_md_results(seq_name, results)
    return results

def parse_wildtrack_sequence(seq_dir):
    """
    seq_dir:
      D:/MultiViewDataset/wildtrack/wildtrack_initial/WildTrackSeq1
    """

    view_infos = []

    camera_dirs = sorted([
        d for d in os.listdir(seq_dir)
        if d.startswith("camera") and
           os.path.isdir(os.path.join(seq_dir, d))
    ])

    for cam in camera_dirs:
        cam_id = int(cam.replace("camera", ""))
        img_dir = os.path.join(seq_dir, cam, "img")

        if not os.path.isdir(img_dir):
            continue

        # 这里 GT 在 seq_dir/cameraX/ 下
        view_infos.append({
            "view_id": cam_id,
            "img_dir": img_dir,
            "anno_dir": os.path.join(seq_dir, cam)
        })

    return view_infos

def parse_md_sequence(seq_dir, md_name):
    view_infos = []

    view_dirs = sorted([
        d for d in os.listdir(seq_dir)
        if d.startswith(md_name + "-") and
           os.path.isdir(os.path.join(seq_dir, d))
    ])

    for vd in view_dirs:
        view_id = int(vd.split("-")[-1])
        view_infos.append({
            "view_id": view_id,
            "img_dir": os.path.join(seq_dir, vd, "img"),
            "anno_dir": os.path.join(seq_dir, vd)
        })

    return view_infos

from multiprocessing import Pool, cpu_count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracker_name', type=str, default="ostrack")
    parser.add_argument('--tracker_param', type=str, default="vitb_256_mae_32x4_ep300")
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--save_results', action='store_true')
    args = parser.parse_args()

    # 相机内外参
    calibration = CameraCalibration("D:/MultiViewDataset/wildtrack/wildtrack_initial/WildTrackSeq1")

    # 加载相机内外参
    camera_matrices = [calibration.get_intrinsic_extrinsic_matrix(i) for i in range(6)]

    # ROOT_DIR = "E:/BaiduNetdiskDownload/Two-MDOT/two_train/test1"

    ROOT_DIR = "D:/MultiViewDataset/wildtrack/wildtrack_initial"

    DATASET_TYPE = "wildtrack"

    seq_list = sorted([
        d for d in os.listdir(ROOT_DIR)
        if d.startswith("WildTrackSeq") and
           os.path.isdir(os.path.join(ROOT_DIR, d))
    ])

    print(f"Found {len(seq_list)} md sequences")

    # md_list = sorted([
    #     d for d in os.listdir(ROOT_DIR)
    #     if d.startswith("md") and os.path.isdir(os.path.join(ROOT_DIR, d))
    # ])
    #
    # print(f"Found {len(md_list)} md sequences")

    # with Pool(processes=min(cpu_count(), 4)) as pool:
    # with Pool(processes=1) as pool:
    #     all_results = pool.starmap(
    #         run_one_md,
    #         [(md, args, ROOT_DIR) for md in md_list]
    #     )

    with Pool(processes=1) as pool:
        all_results = pool.starmap(
            run_one_sequence,
            [(seq, args, ROOT_DIR, DATASET_TYPE) for seq in seq_list]
        )

    valid_md = []
    valid_results = []

    # for md, res in zip(md_list, all_results):
    #     if res is not None:
    #         valid_md.append(md)
    #         valid_results.append(res)

    for md, res in zip(seq_list, all_results):
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
