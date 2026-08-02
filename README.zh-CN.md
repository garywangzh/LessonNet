```text
    __                               _   __     __ 
   / /   ___  ______________  ____  / | / /__  / /_
  / /   / _ \/ ___/ ___/ __ \/ __ \/  |/ / _ \/ __/
 / /___/  __(__  |__  ) /_/ / / / / /|  /  __/ /_  
/_____/\___/____/____/\____/_/ /_/_/ |_/\___/\__/  
```

基于多模态对齐与轻量化处理的智能课堂笔记生成系统。

LessonNet 面向课堂视频自动生成结构化知识笔记：先用多模态感知器（语音、声学、表情、姿态、板书）提取细粒度教学特征，再用时序 Transformer 融合网络对每个时间片段做重要性打分，最后由大语言模型（DeepSeek V3）把高光片段归纳为结构化课堂笔记。

## 主要特性

- **四层流水线**：预处理 → 多模态感知 → 数据整合（FusionNet）→ 大模型输出
- **Video Perceptor**：SAM2.1 目标分割 + MediaPipe 姿态（手臂抬起等）+ DeepFace 情绪唤醒度 + PP-OCR 板书识别
- **Audio Perceptor**：Whisper 语音转写 + OpenSMILE eGeMAPS 声学情感 + Librosa 强调片段检测
- **FusionNet**：时间窗口对齐 + 位置编码 + 中心 Query/上下文 KV 的时序交叉注意力评分 + 动态阈值（均值＋标准差）筛选高光片段
- **轻量化处理**：目标区域局部推理、独立子进程隔离（避免显存竞争）、不均匀采样
- **数据标注**：六级重要性标准 + 自研 VideoAnnotator 3（播放同步打分）
- **远程部署**：Featurize 云 GPU 服务器（RTX 3060 12GB），SSH + 网页端交互，同名视频本地匹配降低传输开销

## 目录结构

```text
Lesson_Net/
├── src/                  # 核心代码（感知器、融合网络、训练脚本、标注工具）
│   ├── LessonNet.py      # 系统总控（一键执行完整流水线）
│   ├── FusionNet.py      # 特征融合与重要性评分网络
│   ├── FusionNet_temporal_transformer.py  # 时序 Transformer 评分器
│   ├── audio_percepter.py
│   ├── visual_percepter.py
│   ├── video_processor.py
│   ├── sam2_processor_new.py
│   ├── video_annotator3.py
│   └── train1.py / train2.py / train_temporal.py   # 三种递进式训练策略
├── config/default.yaml   # 全局配置
├── model/sam2/           # SAM2 源码（权重见 checkpoints，需另行下载）
├── client_app.py         # 远程框选提示词客户端
├── server_api.py         # 服务端接口
├── requirements.txt      # 依赖清单
├── requirements_featurize.txt  # Featurize 环境 pip freeze 快照
├── zihao.sh              # 环境安装脚本
└── activate.sh           # 环境激活脚本
```

## 环境安装

```bash
# 系统依赖（Ubuntu/Debian）
sudo apt-get update
sudo apt-get install -y ffmpeg build-essential cmake libasound2-dev

# Python 环境（推荐虚拟环境，PyTorch 2.4+ / CUDA 12.4 示例）
python3 -m venv zihao_torch
source zihao_torch/bin/activate
pip install --upgrade pip wheel setuptools
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# 本地安装 SAM2
cd model/sam2 && pip install -e . && cd ../..
```

> 完整安装步骤见 `zihao.sh`；服务器端依赖快照见 `requirements_featurize.txt`。

## 模型权重

以下权重不随仓库提供，请自行下载后放入对应目录：

- SAM2.1 checkpoints（项目使用 `sam2.1_hiera_large.pt`）→ `model/sam2/checkpoints/`
- MediaPipe Pose Landmarker（`pose_landmarker_heavy.task`）→ `checkpoints/mediapipe/`
- DeepFace 预训练权重 → `checkpoints/DeepFace/`

## 快速开始

```bash
export DEEPSEEK_API_KEY="your-key"   # 必须：大模型密钥通过环境变量提供
source activate.sh

# 一键运行完整流水线（音频感知 → 视觉感知 → 融合评分 → 报告生成）
python -m src.LessonNet
```

说明：

- `LessonNet.forward()` 一键执行完整流水线；各阶段主动释放 GPU 显存避免资源冲突。
- 视觉提示词通过远程 Gradio/网页端框选教师面部、身体与黑板区域，作为 SAM2 提示词。
- 生成报告为 Markdown（核心知识点总结 + 重点时间戳 + 关键画面）。

## 数据集与标注

- 数据来源：B 站公开课程（教师试讲、物理竞赛、语文精品课、生物复习、英语写作），共 15 节课、约 6.2 小时，训练 12 / 测试 3。
- 标注标准：六级重要性（1.0 核心考点 → 0 无效噪音），每级对应明确的多模态线索。
- 标注工具：`src/video_annotator3.py`，播放同步打分（Q/W/E/A/S/D 对应 1.0/0.8/0.6/0.4/0.2/0），支持变速播放、实时评分曲线、断点续标、JSON 导出。

## 训练

三种递进式训练策略（详见论文与 `src/train_*.py`）：

| 策略 | 模型 | 窗口 | 结果 |
| --- | --- | --- | --- |
| 策略一 | MLP | 10 s | 训练损失 0.056→0.048，不收敛 |
| 策略二 | 时序模型 | 1 s + 17 窗口上下文 | 0.058→0.052，不收敛 |
| 策略三 | 时序 Transformer | 1 s + 17 窗口上下文 | 0.058→0.022（验证约 0.020），收敛 |

说明：实验采用窗口级 8:2 划分，同一节课的不同窗口可能同时进入训练集与验证集，存在一定的信息泄露风险；严格评估建议改用视频级划分。

## 数据合规声明

课堂视频均来自公开渠道（B 站），来源链接已记录在数据清单中，仅用于非商业教学研究，符合合理使用原则；研究过程不采集、不展示可识别的个人敏感信息。本仓库不包含原始视频与个人数据。

## License

MIT License，详见 [LICENSE](LICENSE)。
