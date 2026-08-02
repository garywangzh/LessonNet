```text
    __                               _   __     __ 
   / /   ___  ______________  ____  / | / /__  / /_
  / /   / _ \/ ___/ ___/ __ \/ __ \/  |/ / _ \/ __/
 / /___/  __(__  |__  ) /_/ / / / / /|  /  __/ /_  
/_____/\___/____/____/\____/_/ /_/_/ |_/\___/\__/  
```

An intelligent classroom note-generation system based on multimodal alignment and lightweight processing.

LessonNet automatically generates structured knowledge notes from classroom videos: it first extracts fine-grained teaching features with multimodal perceptors (speech, acoustics, facial expression, pose, and blackboard content), then scores the importance of every time segment with a temporal Transformer fusion network, and finally uses a large language model (DeepSeek V3) to distill the highlight segments into structured classroom notes.

## Key Features

- **Four-layer pipeline**: preprocessing → multimodal perception → integration (FusionNet) → LLM output
- **Visual Perceptor**: SAM2.1 target segmentation + MediaPipe pose (arm raise, etc.) + DeepFace emotion arousal + PP-OCR blackboard recognition
- **Audio Perceptor**: Whisper speech transcription + OpenSMILE eGeMAPS acoustic emotion + Librosa emphasis-region detection
- **FusionNet**: time-window alignment + positional encoding + temporal cross-attention (center window as query, context as keys/values) for importance scoring, with a dynamic mean-plus-standard-deviation threshold for highlight selection
- **Lightweight processing**: region-wise local inference, isolated subprocesses to avoid GPU-memory contention, and non-uniform sampling
- **Data annotation**: six-level importance standard + self-developed VideoAnnotator 3 (playback-synchronized labeling)
- **Remote deployment**: Featurize cloud GPU server (RTX 3060 12 GB), SSH + web UI, same-name local matching to reduce transfer overhead

## Repository Layout

```text
Lesson_Net/
├── src/                  # Core code (perceptors, fusion network, training scripts, annotation tool)
│   ├── LessonNet.py      # System controller (one-click full pipeline)
│   ├── FusionNet.py      # Feature fusion and importance scoring
│   ├── FusionNet_temporal_transformer.py  # Temporal Transformer scorer
│   ├── audio_percepter.py
│   ├── visual_percepter.py
│   ├── video_processor.py
│   ├── sam2_processor_new.py
│   ├── video_annotator3.py
│   └── train1.py / train2.py / train_temporal.py   # Three progressive training strategies
├── config/default.yaml   # Global configuration
├── model/sam2/           # SAM2 source (weights are downloaded separately, see below)
├── client_app.py         # Remote bounding-box prompt client
├── server_api.py         # Server-side API
├── requirements.txt      # Dependency list
├── requirements_featurize.txt  # pip freeze snapshot of the Featurize environment
├── zihao.sh              # Environment setup script
└── activate.sh           # Environment activation script
```

## Installation

```bash
# System dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y ffmpeg build-essential cmake libasound2-dev

# Python environment (virtualenv recommended; PyTorch 2.4+ / CUDA 12.4 example)
python3 -m venv zihao_torch
source zihao_torch/bin/activate
pip install --upgrade pip wheel setuptools
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Install SAM2 locally
cd model/sam2 && pip install -e . && cd ../..
```

> See `zihao.sh` for the full setup script and `requirements_featurize.txt` for the server-side dependency snapshot.

## Model Weights

Model weights are not committed to this repository. Download them and place them as follows:

- SAM2.1 checkpoints (the project uses `sam2.1_hiera_large.pt`) → `model/sam2/checkpoints/`
- MediaPipe Pose Landmarker (`pose_landmarker_heavy.task`) → `checkpoints/mediapipe/`
- DeepFace pretrained weights → `checkpoints/DeepFace/`

## Quick Start

```bash
export DEEPSEEK_API_KEY="your-key"   # required: the LLM key is read from the environment
source activate.sh

# Run the full pipeline in one click (audio → visual → fusion/scoring → report)
python -m src.LessonNet
```

Notes:

- `LessonNet.forward()` runs the complete pipeline; GPU memory is released between stages to avoid resource conflicts.
- Visual prompts are drawn by the user through a remote Gradio/web UI (teacher face, body, and blackboard regions) and used as SAM2 prompts.
- Output is a Markdown report (core knowledge summary + highlight timestamps + key screenshots).

## Dataset and Annotation

- Data source: publicly available courses on Bilibili (teacher demo lessons, physics olympiad, Chinese high-quality lessons, biology review, English writing), 15 lessons / about 6.2 hours in total, train 12 / test 3.
- Annotation standard: six levels (1.0 core knowledge points → 0 invalid noise), each with explicit multimodal cues.
- Annotation tool: `src/video_annotator3.py` — playback-synchronized scoring (Q/W/E/A/S/D map to 1.0/0.8/0.6/0.4/0.2/0), variable playback speed, real-time score curve, resumable labeling, and JSON export.

## Training

Three progressive training strategies (see the paper and `src/train_*.py`):

| Strategy | Model | Window | Result |
| --- | --- | --- | --- |
| 1 | MLP | 10 s | Training loss 0.056→0.048, no convergence |
| 2 | Temporal model | 1 s + 17-window context | 0.058→0.052, no convergence |
| 3 | Temporal Transformer | 1 s + 17-window context | 0.058→0.022 (validation ≈ 0.020), converged |

Note: the experiments use a window-level 8:2 split, so different windows of the same lesson may appear in both the training and validation sets, which carries a risk of information leakage; a video-level split is recommended for stricter evaluation.

## Data Compliance

All classroom videos come from publicly accessible sources (Bilibili), with source links recorded in the data manifest, and are used only for non-commercial teaching and research, consistent with the fair-use principle. No identifiable personal sensitive information is collected or displayed. This repository does not contain raw videos or personal data.

## License

MIT License — see [LICENSE](LICENSE).

## README Languages

- English: this file
- 简体中文: [README.zh-CN.md](README.zh-CN.md)
