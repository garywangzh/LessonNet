import os
import cv2
import subprocess
import json
import logging
from pathlib import Path
import yaml

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, config_path='/home/featurize/work/LessonNet/Lesson_Net/config/default.yaml'):
        # 1. 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_dir = Path(self.config['data']['raw_dir'])
        self.processed_dir = Path(self.config['data']['processed_dir'])
        self.fps_extract = self.config['processing'].get('extract_fps', 1) # 默认每秒1帧
        
        # 创建必要的输出目录
        self._init_directories()

    def _init_directories(self):
        """初始化所有子文件夹，确保路径存在"""
        dirs = [
            self.processed_dir / 'visual' ,
            self.processed_dir / 'audio',
            self.processed_dir / 'text',
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        logger.info("目录结构初始化完成。")

    def process_video(self, video_filename):
        """处理单个视频文件"""
        video_path = self.raw_dir / video_filename
        if not video_path.exists():
            logger.error(f"视频文件不存在: {video_path}")
            return

        logger.info(f"开始处理视频: {video_filename}")
        base_name = video_path.stem
        
        # --- 步骤 1: 提取关键帧 (Visual) ---
        # 目标: data/processed/visual/frames/{video_name}/frame_0001.jpg
        frame_output_dir = self.processed_dir / 'visual' / 'frames' / base_name
        frame_output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("无法打开视频文件")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        frame_interval = int(fps / self.fps_extract)
        
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame_path = frame_output_dir / f"{saved_count:05d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                saved_count += 1
            frame_count += 1
            
        cap.release()
        logger.info(f"视觉处理完成: 提取了 {saved_count} 帧 (总时长 {duration:.2f}s)")

        # --- 步骤 2: 分离音频 (Audio) ---
        # 目标: data/processed/audio/{video_name}.wav
        audio_output_path = self.processed_dir / 'audio' / f"{base_name}.wav"
        
        # 使用 ffmpeg 分离音频 (假设系统已安装 ffmpeg)
        # -vn: 无视频, -acodec pcm_s16le: 编码格式, -ar 16000: 采样率 (Whisper 推荐)
        cmd = [
            'ffmpeg', '-y',
            '-i', str(video_path),
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            str(audio_output_path)
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info(f"音频分离完成: {audio_output_path.name}")
        except subprocess.CalledProcessError:
            logger.error("FFmpeg 处理失败，请检查是否安装了 FFmpeg 并添加到环境变量。")
            return

        # --- 步骤 3: 生成基础清单 (Manifest) ---
        manifest = {
            "video_name": base_name,
            "source_file": str(video_path),
            "duration_seconds": duration,
            "original_fps": fps,
            "extracted_fps": self.fps_extract,
            "total_frames_extracted": saved_count,
            "paths": {
                "frames": str(frame_output_dir),
                "audio": str(audio_output_path)
            },
            "status": "ready_for_modality_extraction"
        }
        
        manifest_path = self.processed_dir / f"{base_name}_manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=4, ensure_ascii=False)
            
        logger.info(f"清单已保存: {manifest_path}")
        return manifest

if __name__ == "__main__":
    # 简单测试入口
    # 请确保 config/default.yaml 已创建并指向正确的 raw 目录
    processor = VideoProcessor()
    
    # 替换为你放在 data/raw 下的真实视频文件名
    test_video = "lecture_sample.mp4" 
    
    if os.path.exists(Path(processor.raw_dir) / test_video):
        processor.process_video(test_video)
    else:
        print(f"请在 {processor.raw_dir} 目录下放入名为 {test_video} 的视频文件后再运行。")