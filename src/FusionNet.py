import json
import torch
import torch.nn as nn
import numpy as np
import os



# ==========================================
#  模型定义
# ==========================================

class ImportanceScorer(nn.Module):
    """
    输入维度: 5 
    (Vis_Emo, Vis_Ges, Aud_Emo, Aud_Feat, OCR)
    """
    def __init__(self, input_dim=5, hidden_dim=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        logits = self.net(x)
        score = torch.sigmoid(logits)
        return score


class TemporalImportanceScorer(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, dropout=0.2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(hidden_dim, 1, kernel_size=1)
        )

    def forward(self, x):
        # x: [B, T, C]
        x = x.transpose(1, 2)      # [B, C, T]
        logits = self.net(x)       # [B, 1, T]
        scores = torch.sigmoid(logits)
        return scores.transpose(1, 2)  # [B, T, 1]
    

# ==========================================
# 3. 主执行流程（推理示例，可删除或保留）
# ==========================================

def main():
    # 此函数仅用于快速测试融合，实际训练请使用 train1.py
    # 配置示例路径（请替换为实际路径）
    VIS_BASE_DIR = "/home/featurize/work/LessonNet/Lesson_Net/data/processed/visual/masks/1_raw_video_00"
    VIS_EMO_PATH = os.path.join(VIS_BASE_DIR, "emotion.json")
    VIS_GES_PATH = os.path.join(VIS_BASE_DIR, "gesture.json")
    OCR_PATH = os.path.join(VIS_BASE_DIR, "ocr.json")
    AUDIO_PATH = "/home/featurize/work/LessonNet/Lesson_Net/data/processed/audio/transcribed/1_raw_video_00_transcript_audio_features.json"

    # 1. 融合特征
    features_matrix = fuse_multimodal_features(VIS_EMO_PATH, VIS_GES_PATH, AUDIO_PATH, OCR_PATH)
    print("特征矩阵示例（前5行）:")
    print(features_matrix[:5])

    # 2. 初始化模型并测试推理
    model = ImportanceScorer(input_dim=5)
    model.eval()
    with torch.no_grad():
        sample = torch.tensor(features_matrix[:5], dtype=torch.float32)
        scores = model(sample)
        print("示例分数:", scores.squeeze().tolist())


if __name__ == "__main__":
    main()