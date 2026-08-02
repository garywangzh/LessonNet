import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import copy
from torch.utils.tensorboard import SummaryWriter
import sys
from pathlib import Path
from .FusionNet import ImportanceScorer, fuse_multimodal_features, MAX_DURATION, WINDOW_SIZE


# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入 FusionNet 中的模型
from src.FusionNet import ImportanceScorer


# ==========================================
# 配置参数（放在全局，供所有函数使用）
# ==========================================
WINDOW_SIZE = 10      # 时间粒度 (秒)
MAX_DURATION = 600    # 视频最大处理时长 (秒)，可根据实际视频调整


# ==========================================
# 数据加载函数
# ==========================================
def load_visual_data(json_path, value_key):
    """通用视觉数据加载函数 (用于情绪和手势)"""
    if not os.path.exists(json_path):
        print(f"警告: 文件不存在 {json_path}")
        return []
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    points = []
    for item in data:
        time = item.get('start')
        value = item.get(value_key)
        if time is not None and value is not None:
            points.append((float(time), float(value)))
    
    points.sort(key=lambda x: x[0])
    return points


def load_ocr_data(json_path):
    """
    加载 OCR 数据（实际格式：包含 start 字段，没有 end）
    """
    if not os.path.exists(json_path):
        print(f"警告: OCR文件不存在 {json_path}")
        return []
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    points = []
    for item in data:
        # 实际 OCR 数据使用 timestamp 或 start
        time = item.get('timestamp')
        if time is None:
            time = item.get('start')
        # 使用 norm_text_len，如果不存在则默认为 0
        value = item.get('norm_text_len', 0)
        
        if time is not None:
            points.append((float(time), float(value)))
            
    points.sort(key=lambda x: x[0])
    if points:
        print(f"  -> 成功加载 {len(points)} 个 OCR 数据点")
    return points


def load_audio_data(json_path):
    """
    加载音频数据（适配嵌套的 scores 结构）
    """
    if not os.path.exists(json_path):
        print(f"警告: 音频特征文件不存在 {json_path}")
        return [], []
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    scores = data.get("scores", {})
    emo_list = scores.get("audio_emotion", [])
    feat_list = scores.get("audio_feature", [])
    
    emo_points = [(x.get('start'), x.get('audio_emotion')) for x in emo_list if x.get('start') is not None]
    feat_points = [(x.get('start'), x.get('audio_feature')) for x in feat_list if x.get('start') is not None]
    
    emo_points.sort(key=lambda x: x[0])
    feat_points.sort(key=lambda x: x[0])
    
    return emo_points, feat_points


def fill_matrix(points, features_matrix, col_idx, max_duration, window_size):
    """将散点数据填充到网格矩阵中"""
    for time, score in points:
        if time >= max_duration:
            continue
        grid_idx = int(time // window_size)
        if np.isnan(features_matrix[grid_idx, col_idx]):
            features_matrix[grid_idx, col_idx] = score
        else:
            features_matrix[grid_idx, col_idx] = max(features_matrix[grid_idx, col_idx], score)


def fuse_multimodal_features(visual_emo_path, visual_ges_path, audio_path, ocr_path):
    """
    5模态融合：顺序 [Vis_Emo, Vis_Ges, Aud_Emo, Aud_Feat, OCR]
    """
    print("正在融合多模态特征...")
    
    # 1. 加载数据
    vis_emo_points = load_visual_data(visual_emo_path, 'arousal_score')
    vis_ges_points = load_visual_data(visual_ges_path, 'arm_raise')
    aud_emo_points, aud_feat_points = load_audio_data(audio_path)
    ocr_points = load_ocr_data(ocr_path)
    
    # 2. 初始化网格
    num_grids = int(MAX_DURATION // WINDOW_SIZE)
    features_matrix = np.full((num_grids, 5), np.nan)
    
    # 3. 填充数据
    fill_matrix(vis_emo_points, features_matrix, 0, MAX_DURATION, WINDOW_SIZE)
    fill_matrix(vis_ges_points, features_matrix, 1, MAX_DURATION, WINDOW_SIZE)
    fill_matrix(aud_emo_points, features_matrix, 2, MAX_DURATION, WINDOW_SIZE)
    fill_matrix(aud_feat_points, features_matrix, 3, MAX_DURATION, WINDOW_SIZE)
    fill_matrix(ocr_points, features_matrix, 4, MAX_DURATION, WINDOW_SIZE)
    
    # 4. 缺失值处理
    features_matrix = np.nan_to_num(features_matrix, nan=0.0)
    
    print(f"特征融合完成。矩阵形状: {features_matrix.shape}")
    return features_matrix


# ==========================================
# 数据集类
# ==========================================
class LessonDataset(Dataset):
    def __init__(self, label_dir, visual_dir, audio_dir):
        self.samples = []
        
        print("正在扫描并对齐数据集...")
        
        # 获取所有视频名称（从标签文件）
        label_files = [f for f in os.listdir(label_dir) if f.endswith("_keyframes.json")]
        print(f"找到 {len(label_files)} 个标签文件")
        
        for label_file in label_files:
            video_name = label_file.replace("_keyframes.json", "")
            print(f"\n处理视频: {video_name}")
            
            # 1. 构造该视频所有特征的路径
            v_dir = os.path.join(visual_dir, video_name)
            a_dir = audio_dir  # 音频特征在同一个目录下
            
            vis_emo_path = os.path.join(v_dir, "emotion.json")
            vis_ges_path = os.path.join(v_dir, "gesture.json")
            ocr_path = os.path.join(v_dir, "ocr.json")
            audio_feat_path = os.path.join(a_dir, f"{video_name}_transcript_audio_features.json")
            label_path = os.path.join(label_dir, label_file)
            
            # 检查必要文件是否存在
            missing_files = []
            for path, name in [(vis_emo_path, "emotion"), (vis_ges_path, "gesture"), 
                               (ocr_path, "ocr"), (audio_feat_path, "audio_feat"), 
                               (label_path, "label")]:
                if not os.path.exists(path):
                    missing_files.append(name)
            
            if missing_files:
                print(f"  ⚠️ 跳过 {video_name}: 缺少 {', '.join(missing_files)}")
                continue
            
            # 2. 提取并融合 5 维特征矩阵
            features_matrix = fuse_multimodal_features(vis_emo_path, vis_ges_path, audio_feat_path, ocr_path)
            num_grids = features_matrix.shape[0]
            
            # 3. 加载标签（从 keyframes.json）
            target_matrix = np.zeros(num_grids)
            target_counts = np.zeros(num_grids)
            
            with open(label_path, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
            
            # 标签数据格式：列表，每个元素包含 time 和 score
            # 假设格式为 [{"time": 10.5, "score": 0.8}, ...]
            if isinstance(label_data, dict) and "data" in label_data:
                label_data = label_data["data"]
            
            for item in label_data:
                time = item.get('time')
                if time is None:
                    time = item.get('timestamp')
                score = item.get('score', 0)
                
                if time is None or time >= MAX_DURATION:
                    continue
                
                grid_idx = int(time // WINDOW_SIZE)
                target_matrix[grid_idx] += score
                target_counts[grid_idx] += 1
            
            # 4. 构建样本
            video_sample_count = 0
            for i in range(num_grids):
                feature_vec = features_matrix[i]
                # 跳过特征全为0的窗口（除非有标签）
                if np.all(feature_vec == 0) and target_counts[i] == 0:
                    continue
                
                label_score = target_matrix[i] / target_counts[i] if target_counts[i] > 0 else 0.0
                
                self.samples.append({
                    "features": torch.tensor(feature_vec, dtype=torch.float32),
                    "label": torch.tensor([label_score], dtype=torch.float32),
                    "video_name": video_name,
                    "window_idx": i
                })
                video_sample_count += 1
            
            print(f"  提取 {video_sample_count} 个有效窗口")
        
        print(f"\n数据集构建完成！共提取 {len(self.samples)} 个有效时间窗口样本。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]["features"], self.samples[idx]["label"]


# ==========================================
# 训练函数
# ==========================================
def train_model():
    # ==========================================
    # 配置参数
    # ==========================================
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 1000
    
    # 路径配置
    BASE_DIR = "/home/featurize/work/LessonNet/Lesson_Net/data/processed"
    VISUAL_DIR = os.path.join(BASE_DIR, "visual", "masks")
    AUDIO_DIR = os.path.join(BASE_DIR, "audio", "transcribed")
    LABEL_DIR = "/home/featurize/work/LessonNet/Lesson_Net/data/raw/training_data/training_data"

    # 检测硬件
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备进行训练: {device}")

    # 准备数据
    dataset = LessonDataset(LABEL_DIR, VISUAL_DIR, AUDIO_DIR)
    if len(dataset) == 0:
        print("未找到有效数据，请检查路径。")
        return

    # 划分训练集和验证集 (80% / 20%)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = ImportanceScorer(input_dim=5).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # TensorBoard
    writer = SummaryWriter("./runs/lessonnet_training")

    best_val_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())

    print("\n开始训练...")
    for epoch in range(EPOCHS):
        # --- 训练阶段 ---
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * features.size(0)
            
        train_loss = train_loss / len(train_dataset)

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * features.size(0)
                
        val_loss = val_loss / len(val_dataset)
        
        # TensorBoard 记录
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)

        # 打印日志
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            
    writer.close()

    # 保存训练好的模型权重
    save_path = os.path.join(BASE_DIR, "best_importance_scorer.pth")
    torch.save(best_model_weights, save_path)
    print(f"\n训练结束！最优模型已保存至: {save_path}")
    print(f"最优验证集 Loss: {best_val_loss:.4f}")


# ==========================================
# 入口
# ==========================================
if __name__ == "__main__":
    train_model()