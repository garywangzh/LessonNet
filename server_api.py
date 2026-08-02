# server_api.py
import os
import json
import uuid
import threading
import traceback
import time
from pathlib import Path
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
import uvicorn

# 导入 LessonNet
from src.LessonNet import LessonNet
from src.video_processor import VideoProcessor

app = FastAPI(title="LessonNet 核心分析引擎")

# 全局任务状态存储
task_status = {}

# 路径配置（根据您的实际目录调整）
VIDEO_ROOT = "/home/featurize/work/LessonNet/Lesson_Net/data/raw/training_data/training_data"
PROCESSED_ROOT = "/home/featurize/work/LessonNet/Lesson_Net/data/processed"

# 模型路径
SAM2_CHECKPOINT = "./model/sam2/checkpoints/sam2.1_hiera_large.pt"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
VISUAL_INPUT_DIR = "./data/processed/visual/frames"
VISUAL_OUTPUT_DIR = "./data/processed/visual/masks"
MEDIAPIPE_MODEL = "./checkpoints/mediapipe/pose_landmarker_heavy.task"
CONFIG_PATH = "/home/featurize/work/LessonNet/Lesson_Net/config/default.yaml"


def process_video_background(video_name, prompts, task_id, sam2_bboxes, frame_time, base_name, use_original_frame_for_ocr):
    """
    后台线程：处理视频并实时更新进度
    """
    try:
        # ---- 阶段 1: 视频预处理 ----
        task_status[task_id] = {"progress": 5, "status": "processing", "message": "开始视频预处理..."}
        
        vp_processor = VideoProcessor(config_path=CONFIG_PATH)
        
        task_status[task_id] = {"progress": 10, "status": "processing", "message": "提取关键帧和音频..."}
        
        manifest = vp_processor.process_video(video_name)
        
        if manifest is None:
            task_status[task_id] = {"progress": 0, "status": "failed", "message": "视频预处理失败"}
            return
        
        task_status[task_id] = {"progress": 35, "status": "processing", "message": f"预处理完成，提取 {manifest['total_frames_extracted']} 帧"}
        
        # ---- 阶段 2: 多模态分析 ----
        task_status[task_id] = {"progress": 40, "status": "processing", "message": "初始化 LessonNet 模型..."}
        
        ln = LessonNet(
            sam2_checkpoint=SAM2_CHECKPOINT,
            sam2_model_cfg=SAM2_CFG,
            visual_input_dir=VISUAL_INPUT_DIR,
            visual_output_dir=VISUAL_OUTPUT_DIR,
            model_path=MEDIAPIPE_MODEL,
            audio_model_size="base",
            audio_input_dir="./data/processed/audio",
            audio_output_dir="./data/processed/audio/transcribed",
            video_name=base_name,
            device=None
        )
        
        task_status[task_id] = {"progress": 50, "status": "processing", "message": "开始多模态分析（视觉、音频、融合）..."}
        
        # 执行 forward（视觉 + 音频），传递 use_original_frame_for_ocr
        ln.forward(
            video_name=base_name,
            audio_filename=f"{base_name}.wav",
            sam2_bboxes=sam2_bboxes,
            annotation_frame_time=frame_time,
            use_original_frame_for_ocr=use_original_frame_for_ocr
        )
        
        task_status[task_id] = {"progress": 80, "status": "processing", "message": "多模态分析完成，生成报告..."}
        
        # ---- 阶段 3: 生成报告 ----
        video_output_dir = os.path.join(PROCESSED_ROOT, base_name)
        os.makedirs(video_output_dir, exist_ok=True)
        report_path = os.path.join(video_output_dir, "final_report.md")
        
        ln.output(report_save_path=report_path)
        
        # ---- 阶段 4: 读取报告 ----
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                report_content = f.read()
        else:
            report_content = f"# 课堂分析报告 - {base_name}\n\n报告文件未生成，请检查处理日志。"
        
        # 完成
        task_status[task_id] = {
            "progress": 100,
            "status": "completed",
            "message": "✅ 处理完成！",
            "report": report_content
        }
        
        print(f"✅ 任务 {task_id} 处理完成")
        
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"❌ 任务 {task_id} 处理失败:")
        print(error_msg)
        task_status[task_id] = {
            "progress": 0,
            "status": "failed",
            "message": f"处理失败: {str(e)}"
        }


@app.post("/analyze/")
async def analyze_video(
    video_name: str = Form(...),
    prompts: str = Form(...)
):
    """
    接收分析请求，立即返回任务ID，后台处理视频
    """
    print(f"\n📥 收到请求: {video_name}")
    print(f"📦 prompts: {prompts[:200]}...")
    
    # 1. 解析 prompts
    try:
        data = json.loads(prompts)
        use_original_frame_for_ocr = data.get("use_original_frame_for_ocr", False)
        sam2_bboxes = {k: v for k, v in data.items() if k not in ["frame_time", "use_original_frame_for_ocr"]}
        frame_time = data.get("frame_time", 0)
        print(f"🎯 BBox: {sam2_bboxes}")
        print(f"🕐 frame_time: {frame_time}")
        print(f"📷 use_original_frame_for_ocr: {use_original_frame_for_ocr}")
        
    except Exception as e:
        print(f"❌ 解析失败: {e}")
        return {"status": "error", "message": f"解析 prompts 失败: {str(e)}"}
    
    # 2. 检查视频是否存在
    video_path = os.path.join(VIDEO_ROOT, video_name)
    if not os.path.exists(video_path):
        print(f"❌ 视频不存在: {video_path}")
        return {"status": "error", "message": f"视频不存在: {video_path}"}
    
    # 3. 生成任务ID
    task_id = str(uuid.uuid4())
    base_name = os.path.splitext(video_name)[0]
    
    # 4. 初始化任务状态
    task_status[task_id] = {
        "progress": 0,
        "status": "processing",
        "message": "任务已接收，准备开始..."
    }
    
    # 5. 启动后台线程处理（传递 use_original_frame_for_ocr）
    thread = threading.Thread(
        target=process_video_background,
        args=(video_name, prompts, task_id, sam2_bboxes, frame_time, base_name, use_original_frame_for_ocr)
    )
    thread.daemon = True
    thread.start()
    
    print(f"🚀 任务 {task_id} 已启动（后台处理）")
    
    # 6. 立即返回任务ID
    return {
        "status": "accepted",
        "task_id": task_id,
        "message": "任务已接收，正在后台处理..."
    }


@app.get("/progress/{task_id}")
async def get_progress(task_id: str):
    """
    客户端轮询进度接口
    """
    status = task_status.get(task_id)
    
    if status is None:
        return {"progress": 0, "status": "unknown", "message": "任务不存在"}
    
    if status.get("status") == "completed" and "report" in status:
        return {
            "progress": status.get("progress", 100),
            "status": status.get("status", "completed"),
            "message": status.get("message", "处理完成"),
            "report": status.get("report", "")
        }
    
    return {
        "progress": status.get("progress", 0),
        "status": status.get("status", "processing"),
        "message": status.get("message", "处理中...")
    }


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "ok"}


@app.get("/tasks")
async def list_tasks():
    """列出所有任务状态（调试用）"""
    return {
        "total": len(task_status),
        "tasks": {
            tid: {
                "progress": info.get("progress", 0),
                "status": info.get("status", "unknown"),
                "message": info.get("message", "")
            }
            for tid, info in task_status.items()
        }
    }


if __name__ == "__main__":
    print("🚀 启动 LessonNet 服务器")
    print(f"📁 VIDEO_ROOT: {VIDEO_ROOT}")
    print(f"📁 PROCESSED_ROOT: {PROCESSED_ROOT}")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8001)