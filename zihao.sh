# Ubuntu/Debian 系统命令
sudo apt-get update
sudo apt-get install -y ffmpeg build-essential cmake libasound2-dev

# 安装A环境
# 1. 创建并激活环境
python3 -m venv zihao_torch
source zihao_torch/bin/activate

# 2. 升级基础构建工具
pip install --upgrade pip wheel setuptools

# 3. 安装核心深度学习框架 (PyTorch 2.4+ / CUDA 12.4 示例)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. 安装高效视频处理与基础视觉库
# decord 极其适合端到端的视频对象分割和运动提取任务，能显著降低 I/O 延迟
pip install decord opencv-python matplotlib scipy

# 5. 安装 YOLOv11 (用于黑板和人物的快速定位)
pip install ultralytics

# 6. 安装 SAM2 基础依赖 (如果作为 YOLO 的精细化辅助)
pip install hydra-core omegaconf iopath fvcore pycocotools

# 7. 安装音频与语音特征提取 (SenseVoice & Whisper 保留)
pip install funasr modelscope openai-whisper librosa

# 8. 安装 LLM 与 VLM 核心库 (Qwen3-VL, DeepSeek-V3)
pip install transformers accelerate sentencepiece tiktoken vercel-ai vllm

# 9.安装本地SAM2
cd ./model/sam2
pip install -e .

# 4. 安装 Mediapipe (针对 CUDA 的构建通常需要特定版本)
pip install mediapipe

# 5. 安装 Deepface
pip install deepface

# 6. 安装必要的辅助包 (固定 Numpy 版本如1.26.4 以防冲突)
pip install "numpy<2.0" "opencv-python==4.9.0.80" matplotlib scipy seaborn

pip install tf-keras

pip install opensmile

pip install "paddleocr"

pip install "paddleocr[all]"

pip install "PyYAML==6.0.2"

pip install --upgrade "packaging>=20.9" "requests>=2.25"

pip install "numpy<2.0" "opencv-python-headless>4.11.0" --no-deps --force-reinstall

#安装可视化界面
pip install fastapi uvicorn python-multipart


#optional 
# 1. 更新系统软件源
sudo apt-get update

# 2. 一次性安装 MediaPipe 和 OpenCV 在服务器端常缺的底层依赖
sudo apt-get install -y libgles2-mesa libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6