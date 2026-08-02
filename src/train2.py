import _init_paths
import os
import json
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter


# =========================
# Global Config
# =========================
WINDOW_SIZE = 3.0
BATCH_SIZE = 32
LEARNING_RATE = 6e-4
EPOCHS = 500
USE_OCR = False


# =========================
# Model
# =========================
from FusionNet import ImportanceScorer,TemporalImportanceScorer
# except Exception:
#     class ImportanceScorer(nn.Module):
#         def __init__(self, input_dim=5):
#             super().__init__()
#             self.net = nn.Sequential(
#                 nn.Linear(input_dim, 64),
#                 nn.ReLU(),
#                 nn.Dropout(0.2),
#                 nn.Linear(64, 32),
#                 nn.ReLU(),
#                 nn.Linear(32, 1)
#             )

#         def forward(self, x):
#             return self.net(x)


# =========================
# IO Utils
# =========================
def safe_load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_audio_feature_file(audio_dir, video_name):
    candidates = [
        f"{video_name}_audio_features.json",
        f"{video_name}_transcript_audio_features.json",
        f"{video_name}_transcript_audio_features_.json",
    ]

    for name in candidates:
        path = os.path.join(audio_dir, name)
        if os.path.exists(path):
            return path

    return None


# =========================
# 异常值处理函数
# =========================
def detect_and_clip_outliers(values, method="iqr", multiplier=3.0):
    """
    检测并裁剪异常值
    
    参数:
        values: numpy array 或 list
        method: "iqr" 或 "std"
        multiplier: IQR 倍数（默认3）或标准差倍数（默认3）
    
    返回:
        裁剪后的值，以及原始有效值
    """
    if len(values) == 0:
        return values, []
    
    arr = np.array(values)
    
    # 方法1：IQR（四分位距）
    if method == "iqr":
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        
        if iqr == 0:
            # 如果所有值相同，不裁剪
            return arr.tolist(), []
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
    else:
        # 方法2：标准差
        mean = np.mean(arr)
        std = np.std(arr)
        
        if std == 0:
            return arr.tolist(), []
        
        lower_bound = mean - multiplier * std
        upper_bound = mean + multiplier * std
    
    # 裁剪异常值
    clipped = np.clip(arr, lower_bound, upper_bound)
    
    # 标记哪些是异常值
    outliers_mask = (arr < lower_bound) | (arr > upper_bound)
    outlier_indices = np.where(outliers_mask)[0].tolist()
    outlier_values = arr[outliers_mask].tolist()
    
    return clipped.tolist(), outlier_values


# =========================
# Window Utils
# =========================
def get_num_windows(max_time):
    return int(np.ceil(max_time / WINDOW_SIZE))


def point_to_window_idx(timestamp):
    return int(np.floor(float(timestamp) / WINDOW_SIZE))


def interval_to_window_indices(start, end):
    start = float(start)
    end = float(end)

    if end <= start:
        return [point_to_window_idx(start)]

    eps = 1e-6

    first = int(np.floor(start / WINDOW_SIZE))
    last = int(np.floor((end - eps) / WINDOW_SIZE))

    return list(range(max(0, first), max(0, last) + 1))


def add_point_value(sum_arr, count_arr, idx, value):
    if value is None:
        return
    try:
        value = float(value)
    except Exception:
        return

    if idx < 0 or idx >= len(sum_arr):
        return

    sum_arr[idx] += value
    count_arr[idx] += 1.0


def add_interval_value(sum_arr, count_arr, start, end, value):
    if value is None:
        return
    try:
        value = float(value)
    except Exception:
        return

    indices = interval_to_window_indices(start, end)

    for idx in indices:
        if 0 <= idx < len(sum_arr):
            sum_arr[idx] += value
            count_arr[idx] += 1.0


def finalize_1d(sum_arr, count_arr):
    """
    返回：平均值数组（缺失为 NaN），有效标志数组
    """
    out = np.full_like(sum_arr, np.nan, dtype=np.float32)  # ← 关键修改
    valid = count_arr > 0
    out[valid] = sum_arr[valid] / count_arr[valid]
    return out, valid


# =========================
# Feature Loaders (增加时间戳校准)
# =========================
def load_visual_records(json_path, value_key):
    data = safe_load_json(json_path)
    if data is None:
        print(f"警告: 文件不存在 {json_path}")
        return []

    records = []

    for item in data:
        timestamp = item.get("timestamp", item.get("start", None))
        start = item.get("start", timestamp)
        end = item.get("end", timestamp)
        value = item.get(value_key, None)

        if timestamp is None or value is None:
            continue

        start = float(start)
        end = float(end) if end is not None else start

        records.append({
            "type": "point" if end <= start else "interval",
            "timestamp": float(timestamp),
            "start": start,
            "end": end,
            "value": value
        })

    return records


def load_ocr_records(json_path):
    data = safe_load_json(json_path)
    if data is None:
        return []

    records = []

    for item in data:
        timestamp = item.get("timestamp", item.get("start", None))
        if timestamp is None:
            continue

        start = item.get("start", timestamp)
        end = item.get("end", timestamp)
        value = item.get("norm_text_len", 0.0)

        start = float(start)
        end = float(end) if end is not None else start

        records.append({
            "type": "point" if end <= start else "interval",
            "timestamp": float(timestamp),
            "start": start,
            "end": end,
            "value": value
        })

    return records


def load_audio_records(json_path):
    data = safe_load_json(json_path)
    if data is None:
        print(f"警告: 音频特征文件不存在 {json_path}")
        return [], []

    scores = data.get("scores", {})

    audio_emotion = []
    audio_feature = []

    for item in scores.get("audio_emotion", []):
        start = item.get("start", None)
        end = item.get("end", None)
        value = item.get("audio_emotion", None)

        if start is None or value is None:
            continue

        start = float(start)
        end = float(end) if end is not None else start + WINDOW_SIZE

        audio_emotion.append({
            "type": "interval",
            "start": start,
            "end": end,
            "timestamp": start,
            "value": value
        })

    for item in scores.get("audio_feature", []):
        start = item.get("start", None)
        end = item.get("end", None)
        value = item.get("audio_feature", None)

        if start is None or value is None:
            continue

        start = float(start)
        end = float(end) if end is not None else start + WINDOW_SIZE

        audio_feature.append({
            "type": "interval",
            "start": start,
            "end": end,
            "timestamp": start,
            "value": value
        })

    return audio_emotion, audio_feature


# =========================
# Label Alignment
# =========================
def load_label_records(label_path):
    """
    支持格式：
    {
        "meta": {...},
        "data": [
            {"timestamp": 2109.664, "score": 0.6},
            {"timestamp": 2111.661, "score": 0.6}
        ]
    }
    或直接列表:
    [
        {"timestamp": 2109.664, "score": 0.6}
    ]

    算法：
    1. 按 timestamp 排序。
    2. 若没有 start/end，则用相邻 timestamp 的中点推断区间。
    3. 若 label 区间大于 1s，则复制到覆盖的 1s 窗口。
    4. 若多个 label 落到同一个 1s 窗口，则求均值。
    """
    raw_data = safe_load_json(label_path)
    
    # 如果 raw_data 是 None，直接返回空列表
    if raw_data is None:
        print(f"警告: 无法读取标签文件 {label_path}")
        return []
    
    # 兼容两种格式
    if isinstance(raw_data, dict) and "data" in raw_data:
        raw_data = raw_data["data"]
    elif not isinstance(raw_data, list):
        print(f"警告: 标签文件格式未知 {label_path}")
        return []

    point_items = []
    interval_items = []

    for item in raw_data:
        score = item.get("score", item.get("importance", item.get("label", 0.0)))

        if "start" in item and "end" in item:
            interval_items.append({
                "start": float(item["start"]),
                "end": float(item["end"]),
                "value": float(score)
            })
        else:
            timestamp = item.get("timestamp", item.get("time", None))
            if timestamp is None:
                continue

            point_items.append({
                "timestamp": float(timestamp),
                "value": float(score)
            })

    point_items.sort(key=lambda x: x["timestamp"])

    inferred_intervals = []

    if len(point_items) == 1:
        t = point_items[0]["timestamp"]
        inferred_intervals.append({
            "start": t,
            "end": t + WINDOW_SIZE,
            "value": point_items[0]["value"]
        })

    elif len(point_items) > 1:
        timestamps = [x["timestamp"] for x in point_items]

        gaps = np.diff(timestamps)
        median_gap = float(np.median(gaps)) if len(gaps) > 0 else WINDOW_SIZE
        median_gap = max(median_gap, WINDOW_SIZE)

        for i, item in enumerate(point_items):
            t = item["timestamp"]

            if i == 0:
                left = t
            else:
                left = (timestamps[i - 1] + t) / 2.0

            if i == len(point_items) - 1:
                right = t + median_gap / 2.0
            else:
                right = (t + timestamps[i + 1]) / 2.0

            if right <= left:
                right = left + WINDOW_SIZE

            inferred_intervals.append({
                "start": left,
                "end": right,
                "value": item["value"]
            })

    return interval_items + inferred_intervals


# =========================
# Dynamic Duration
# =========================
def collect_max_time(*record_lists):
    max_time = 0.0

    for records in record_lists:
        for r in records:
            if "end" in r:
                max_time = max(max_time, float(r["end"]))
            elif "timestamp" in r:
                max_time = max(max_time, float(r["timestamp"]))

    return max_time


def build_feature_vector(records, num_windows):
    sum_arr = np.zeros(num_windows, dtype=np.float32)
    count_arr = np.zeros(num_windows, dtype=np.float32)

    for r in records:
        if r["type"] == "point":
            idx = point_to_window_idx(r["timestamp"])
            add_point_value(sum_arr, count_arr, idx, r["value"])
        else:
            add_interval_value(sum_arr, count_arr, r["start"], r["end"], r["value"])

    return finalize_1d(sum_arr, count_arr)


def build_label_vector(label_records, num_windows):
    sum_arr = np.zeros(num_windows, dtype=np.float32)
    count_arr = np.zeros(num_windows, dtype=np.float32)

    for r in label_records:
        add_interval_value(sum_arr, count_arr, r["start"], r["end"], r["value"])

    return finalize_1d(sum_arr, count_arr)

def calibrate_visual_records_to_reference(records, reference_max_time):
    """
    将错误时间轴下的视觉 point 记录校准到真实时间轴，并转成 interval。

    核心逻辑：
    1. 原 emotion.json / gesture.json 的 timestamp 是错误压缩时间。
    2. 使用 audio 或 label 的真实最大时间作为 reference_max_time。
    3. 将视觉 timestamp 线性拉伸到 reference 时间轴。
    4. 将视觉 point 转换为 interval，使其覆盖到下一个视觉采样点。
    """
    if not records or reference_max_time <= 0:
        return records

    records = sorted(records, key=lambda x: x["timestamp"])

    raw_times = np.array([float(r["timestamp"]) for r in records], dtype=np.float32)
    raw_min = 0
    raw_max = float(raw_times.max())

    if raw_max <= raw_min:
        return records

    # 线性校准到 [0, reference_max_time]
    calibrated_times = (raw_times - raw_min) / (raw_max - raw_min) * reference_max_time

    calibrated_records = []

    for i, r in enumerate(records):
        start = float(calibrated_times[i])

        if i < len(records) - 1:
            end = float(calibrated_times[i + 1])
        else:
            # 最后一帧延续一个 median step，或者最多到 reference_max_time
            if len(calibrated_times) > 1:
                median_step = float(np.median(np.diff(calibrated_times)))
            else:
                median_step = WINDOW_SIZE
            end = min(start + median_step, reference_max_time)

        if end <= start:
            end = start + WINDOW_SIZE

        calibrated_records.append({
            "type": "interval",
            "timestamp": start,
            "start": start,
            "end": end,
            "value": r["value"]
        })

    return calibrated_records

# =========================
# Dataset
# =========================
class LessonDataset(Dataset):
    def __init__(self, label_dir, visual_dir, audio_dir):
        self.samples = []

        label_files = [
            f for f in os.listdir(label_dir)
            if f.endswith("_keyframes.json")
        ]

        print(f"找到 {len(label_files)} 个标签文件")

        for label_file in label_files:
            video_name = label_file.replace("_keyframes.json", "")
            print(f"\n处理视频: {video_name}")

            v_dir = os.path.join(visual_dir, video_name)

            emotion_path = os.path.join(v_dir, "emotion.json")
            gesture_path = os.path.join(v_dir, "gesture.json")
            ocr_path = os.path.join(v_dir, "ocr.json")
            label_path = os.path.join(label_dir, label_file)
            audio_path = find_audio_feature_file(audio_dir, video_name)

            required = [
                emotion_path,
                gesture_path,
                label_path,
                audio_path
            ]

            if USE_OCR:
                required.append(ocr_path)

            if any(p is None or not os.path.exists(p) for p in required):
                print(f"  ⚠️ 跳过 {video_name}: 存在缺失文件")
                continue

            visual_emotion = load_visual_records(emotion_path, "arousal_score")
            visual_gesture = load_visual_records(gesture_path, "arm_raise")
            audio_emotion, audio_feature = load_audio_records(audio_path)
            label_records = load_label_records(label_path)

            if USE_OCR:
                ocr_records = load_ocr_records(ocr_path)
            else:
                ocr_records = []

            # ==========================================
            # 🔥 单独检测 emotion 时间并校准
            # ==========================================
            emotion_max_time = collect_max_time(visual_emotion) if visual_emotion else 0
            gesture_max_time = collect_max_time(visual_gesture) if visual_gesture else 0
            audio_emo_max_time = collect_max_time(audio_emotion) if audio_emotion else 0
            audio_feat_max_time = collect_max_time(audio_feature) if audio_feature else 0
            label_max_time = collect_max_time(label_records) if label_records else 0

            print(f"  📊 各模态最大时间:")
            print(f"     emotion: {emotion_max_time:.1f}s")
            print(f"     gesture: {gesture_max_time:.1f}s")
            print(f"     audio_emotion: {audio_emo_max_time:.1f}s")
            print(f"     audio_feature: {audio_feat_max_time:.1f}s")
            print(f"     label: {label_max_time:.1f}s")

            # 以 label 为基准校准各模态
            if label_max_time > 0:
                reference_max_time = max(
                        collect_max_time(audio_emotion) if audio_emotion else 0,
                        collect_max_time(audio_feature) if audio_feature else 0,
                        collect_max_time(label_records) if label_records else 0
                    )
                visual_emotion = calibrate_visual_records_to_reference(
                    visual_emotion,
                    reference_max_time
                )

                visual_gesture = calibrate_visual_records_to_reference(
                    visual_gesture,
                    reference_max_time
                )

                # audio_feature = calibrate_visual_records_to_reference(
                #     audio_feature,
                #     reference_max_time
                # )

                # audio_emotion = calibrate_visual_records_to_reference(
                #     audio_emotion,
                #     reference_max_time
                # )

            # 重新计算各模态时间
            emotion_max_time = collect_max_time(visual_emotion) if visual_emotion else 0
            gesture_max_time = collect_max_time(visual_gesture) if visual_gesture else 0
            audio_emo_max_time = collect_max_time(audio_emotion) if audio_emotion else 0
            audio_feat_max_time = collect_max_time(audio_feature) if audio_feature else 0

            print(f"  📊 校准后各模态最大时间:")
            print(f"     emotion: {emotion_max_time:.1f}s")
            print(f"     gesture: {gesture_max_time:.1f}s")
            print(f"     audio_emotion: {audio_emo_max_time:.1f}s")
            print(f"     audio_feature: {audio_feat_max_time:.1f}s")

            max_time = collect_max_time(
                visual_emotion,
                visual_gesture,
                audio_emotion,
                audio_feature,
                ocr_records,
                label_records
            )

            if max_time <= 0:
                print(f"  ⚠️ 跳过 {video_name}: 无有效时间信息")
                continue

            num_windows = get_num_windows(max_time)
            
            # ==========================================
            # 🔥 构建特征向量（缺失值保持为 NaN）
            # ==========================================
            vemo, vemo_valid = build_feature_vector(visual_emotion, num_windows)
            vges, vges_valid = build_feature_vector(visual_gesture, num_windows)
            aemo, aemo_valid = build_feature_vector(audio_emotion, num_windows)
            afeat, afeat_valid = build_feature_vector(audio_feature, num_windows)
            labels, label_valid = build_label_vector(label_records, num_windows)
            # print(emotion_path, vemo)

            if USE_OCR:
                ocr, ocr_valid = build_feature_vector(ocr_records, num_windows)
                features = np.stack([vemo, vges, aemo, afeat, ocr], axis=1)
                feature_valid = vemo_valid | vges_valid | aemo_valid | afeat_valid | ocr_valid
            else:
                features = np.stack([vemo, vges, aemo, afeat], axis=1)
                feature_valid = vemo_valid | vges_valid | aemo_valid | afeat_valid

            # ==========================================
            # 🔥 缺失值填充：用列均值填充（而不是 0）
            # ==========================================
            for col_idx in range(features.shape[1]):
                col_values = features[:, col_idx]
                valid_mask = ~np.isnan(col_values)
                valid_values = col_values[valid_mask]
                
                if len(valid_values) > 0:
                    # 用列均值填充缺失值
                    col_mean = np.mean(valid_values)
                    features[:, col_idx] = np.where(np.isnan(features[:, col_idx]), col_mean, features[:, col_idx])
                    print(f"  模态 {col_idx} 列均值: {col_mean:.4f}, 非NaN比例: {valid_mask.sum()/len(valid_mask):.2%}")
                else:
                    print(f"  ⚠️ 模态 {col_idx} 全为 NaN，填充为 0")
                    features[:, col_idx] = 0.0

            # ==========================================
            # 🔥 异常值处理（按列处理）
            # ==========================================
            for col_idx in range(features.shape[1]):
                col_values = features[:, col_idx]
                valid_mask = ~np.isnan(col_values)
                valid_values = col_values[valid_mask]
                
                if len(valid_values) > 0:
                    clipped_values, outliers = detect_and_clip_outliers(valid_values, method="iqr", multiplier=3.0)
                    if outliers:
                        print(f"  📌 模态 {col_idx} 检测到 {len(outliers)} 个异常值，已裁剪")
                    features[valid_mask, col_idx] = np.array(clipped_values)

            # 最后处理剩余的 NaN（如果有的话）
            features = np.nan_to_num(features, nan=0.0)

            # ==========================================
            # 🔥 数据清洗后再构建样本
            # ==========================================

            # 1. 清洗特征
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
            features = np.clip(features, 0.0, 1.0)

            # 2. 清洗标签
            labels = np.nan_to_num(labels, nan=0.0, posinf=1.0, neginf=0.0)
            labels = np.clip(labels, 0.0, 1.0)

            # 3. 打印统计信息
            print(f"📊 特征清洗后: 最大值={features.max():.4f}, 最小值={features.min():.4f}")
            print(f"📊 标签清洗后: 最大值={labels.max():.4f}, 最小值={labels.min():.4f}")

            # 4. 构建样本
            video_count = 0
            for idx in range(num_windows):
                if not feature_valid[idx] and not label_valid[idx]:
                    continue
                
                # 检查单个样本是否有 NaN
                feat = features[idx]
                lbl = labels[idx]
                
                if np.isnan(feat).any() or np.isnan(lbl):
                    continue

                self.samples.append({
                    "features": torch.tensor(feat, dtype=torch.float32),
                    "label": torch.tensor([lbl], dtype=torch.float32),
                    "video_name": video_name,
                    "window_idx": idx,
                    "start": idx * WINDOW_SIZE,
                    "end": (idx + 1) * WINDOW_SIZE
                })
                video_count += 1

            print(f"  提取 {video_count} 个 {WINDOW_SIZE}s 训练窗口")
            

        print(f"\n数据集构建完成，共 {len(self.samples)} 个样本")
        # 统计标签分布
        label_values = [s["label"].item() for s in self.samples]
        positive_count = sum(1 for v in label_values if v > 0.5)
        zero_count = sum(1 for v in label_values if v == 0)
        print(f"📊 标签分布: 正样本(>0.5): {positive_count}, 零样本(=0): {zero_count}, 总计: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return item["features"], item["label"]


# =========================
# Training
# =========================
def train_model():
    BASE_DIR = "/home/featurize/work/LessonNet/Lesson_Net/data/processed"
    VISUAL_DIR = os.path.join(BASE_DIR, "visual", "masks")
    AUDIO_DIR = os.path.join(BASE_DIR, "audio", "transcribed")
    LABEL_DIR = "/home/featurize/work/LessonNet/Lesson_Net/data/raw/training_data/training_data"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    dataset = LessonDataset(
        label_dir=LABEL_DIR,
        visual_dir=VISUAL_DIR,
        audio_dir=AUDIO_DIR
    )
    all_features = torch.stack([s["features"] for s in dataset.samples])
    all_labels = torch.tensor([s["label"].item() for s in dataset.samples])

    print("📊 特征与标签的相关系数:")
    for i in range(all_features.shape[1]):
        corr = np.corrcoef(all_features[:, i].numpy(), all_labels.numpy())[0, 1]
        print(f"  模态 {i}: {corr:.4f}")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    if val_size == 0:
        train_size = len(dataset) - 1
        val_size = 1

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    input_dim = 5 if USE_OCR else 4
    model = ImportanceScorer(input_dim=input_dim).to(device)
    # model = TemporalImportanceScorer(input_dim=input_dim).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5
    )

    writer = SummaryWriter("./runs/lessonnet_training")

    best_val_loss = float("inf")
    best_model_weights = copy.deepcopy(model.state_dict())

    print("\n开始训练...")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            # print(features,labels)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)

        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)

                outputs = model(features)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * features.size(0)

        val_loss /= len(val_dataset)

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)

        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())

    writer.close()

    save_path = os.path.join(BASE_DIR, "best_importance_scorer.pth")
    torch.save(best_model_weights, save_path)

    print(f"\n训练结束，最优模型已保存至: {save_path}")
    print(f"最优验证集 Loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    train_model()