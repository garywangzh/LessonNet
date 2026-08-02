from paddleocr import PaddleOCR
import os
import cv2
import json
import base64
import torch
from pathlib import Path
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# 第三方模型库导入
from openai import OpenAI
from deepface import DeepFace
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from .sam2_processor_new import sam2_processor
import sys
import json
import subprocess
from pathlib import Path
from tqdm import tqdm

os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # 顺便屏蔽 TF 烦人的底层警告信息
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
os.environ["DEEPFACE_HOME"] = "./checkpoints/DeepFace"

# 获取当前文件的绝对路径，再往上推一层找到 Lesson_Net 根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 强行将根目录插入到环境变量的最前面
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 注意：这里假设 sam2_processor 是你自己在其他文件中定义的类
# 如果在同一目录下，请使用类似: from your_sam2_module import sam2_processor
# 为了代码不报错，请确保该类存在。

class visual_percepter():
    def __init__(self, sam2_checkpoint, sam2_model_cfg, input_dir, output_dir,
                 model_path, fps=30, prompt=None, device=None):
        # 设备设置
        self.device = device if device else self._setup_device()
        self.fps = fps

        # 保存参数
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_model_cfg = sam2_model_cfg
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model_path = model_path
        self.prompt = prompt if prompt else "请提取黑板上的文本内容。"

        # 初始化 MediaPipe 姿态检测器
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mediapipe_detector = vision.PoseLandmarker.create_from_options(options)

        
        # 初始化 SAM2 处理器 (确保 sam2_processor 已正确导入)
        self.sam2_processor = sam2_processor(
            sam2_checkpoint, sam2_model_cfg, input_dir, output_dir, device=self.device, fps=fps
        )
        
        # # 初始化 PaddleOCR
        # print("正在初始化 PaddleOCR (强行使用 CPU 模式以防止与 SAM2 发生显存冲突)...")
        # # use_angle_cls=True 允许识别倾斜文字，use_gpu=False 物理隔离显存争夺
        # self.paddle_ocr = PaddleOCR(ocr_version="PP-OCRv4",
        #     use_doc_orientation_classify=False, 
        #     use_doc_unwarping=False, 
        #     use_textline_orientation=False,
        #     lang="ch")
        # print("PaddleOCR 初始化完成。")

    def _setup_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # 开启 TF32 加速 (针对 RTX 30/40/A100 等新架构显卡)
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            
        print(f"视觉感知器使用设备: {device}")
        return device
    
    # 图片转 Base64 辅助函数 (供 Qwen3-VL 使用)
    def _encode_image_to_base64(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def forward(self, video_name, sam2_bboxes, use_original_frame_for_ocr=False):
        self.video_dir = os.path.join(self.input_dir, video_name)
        self.video_save_dir = os.path.join(self.output_dir, video_name)
        
        # 调用 SAM2 处理（返回三个目录）
        if use_original_frame_for_ocr:
            # 模式：使用原帧做黑板OCR，但依然需要分割人脸和身体
            # 从 sam2_bboxes 中移除 blackboard，避免 SAM2 尝试分割黑板
            bboxes_for_sam2 = {k: v for k, v in sam2_bboxes.items() if k != "blackboard"}
            face_save_dir, body_save_dir, _ = self.sam2_processor.sam2_process(video_name, bboxes_for_sam2)
            # 黑板使用原始帧目录
            blackboard_save_dir = self.video_dir
        else:
            # 正常模式：分割所有目标
            face_save_dir, body_save_dir, blackboard_save_dir = self.sam2_processor.sam2_process(video_name, sam2_bboxes)
        
        torch.cuda.empty_cache()
        
        # 如果分割返回 None，使用默认路径
        if face_save_dir is None:
            face_save_dir = os.path.join(self.video_save_dir, "face")
        if body_save_dir is None:
            body_save_dir = os.path.join(self.video_save_dir, "body")
        if blackboard_save_dir is None:
            blackboard_save_dir = os.path.join(self.video_save_dir, "blackboard")
        
        # --- 特征提取 ---
        # 姿态识别 (使用身体分割)
        gesture_frames = self.gesture_recognition(body_save_dir)
        if gesture_frames:
            self._save_json(gesture_frames, os.path.join(self.video_save_dir, 'gesture.json'))
        
        # 情绪识别 (使用人脸分割)
        emotion_frames = self.emotion_recognition(face_save_dir)
        if emotion_frames:
            self._save_json(emotion_frames, os.path.join(self.video_save_dir, 'emotion.json'))
        
        # OCR 识别
        if use_original_frame_for_ocr:
            # 采样帧列表：每5帧取1帧
            all_frames = sorted([f for f in os.listdir(self.video_dir) if f.endswith(('.jpg', '.png'))])
            sampled_frames = all_frames[::40]  # 每5帧取1帧
            print(f"📌 [OCR模式] 原始帧总数: {len(all_frames)}, 采样后: {len(sampled_frames)} 帧")
            ocr_frames = self.ocr_recognition(blackboard_save_dir, frame_list=sampled_frames)
        else:
            ocr_frames = self.ocr_recognition(blackboard_save_dir)
        
        if ocr_frames:
            self._save_json(ocr_frames, os.path.join(self.video_save_dir, 'ocr.json'))
        
        print(f"视频 {video_name} 的视觉处理流程全部完成。")
    def gesture_recognition(self, body_save_dir):
        if not body_save_dir or not os.path.exists(body_save_dir):
            print(f"跳过姿态识别：目录 {body_save_dir} 不存在")
            return []

        print(f"开始姿态识别：{body_save_dir}")
        image_paths = sorted(Path(body_save_dir).glob("*.png"))
        results = []
        
        for img_path in tqdm(image_paths,desc="姿态识别", unit="帧"):
            frame_idx = int(img_path.stem)
            timestamp = frame_idx / self.fps

            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            detection_result = self.mediapipe_detector.detect(mp_image)

            arm_raise = None
            if detection_result.pose_landmarks:
                landmarks = detection_result.pose_landmarks[0]
                left_shoulder = landmarks[11]
                left_wrist = landmarks[15]
                # Y轴向下，如果手腕的Y值小于肩膀的Y值，说明手举过了肩膀
                arm_raise = left_shoulder.y - left_wrist.y 

            results.append({
                'frame_idx': frame_idx,
                'timestamp': timestamp,
                'arm_raise': arm_raise
            })
        
        valid_arm_raises = [res['arm_raise'] for res in results if res['arm_raise'] is not None]

        if valid_arm_raises:
            min_val = min(valid_arm_raises)
            max_val = max(valid_arm_raises)
            value_range = max_val - min_val

            # 2. 遍历结果进行缩放
            for res in results:
                if res['arm_raise'] is not None:
                    if value_range > 0:
                        # 归一化核心公式：(x - min) / (max - min)
                        # 保留4位小数，防止 JSON 保存时浮点数过长
                        normalized_value = (res['arm_raise'] - min_val) / value_range
                        res['arm_raise'] = round(normalized_value, 4)
                    else:
                        # 保护机制：如果所有帧的数值都一模一样（极少见），防止除以 0 报错
                        res['arm_raise'] = 0.0
        

        print(f"姿态识别完成，处理 {len(results)} 帧")
        return results
    
    def ocr_recognition(self, blackboard_save_dir, frame_list=None):
        """
        唤醒独立子进程进行 OCR 识别
        frame_list: 可选，指定要处理的帧文件名列表
        """
        if not blackboard_save_dir or not os.path.exists(blackboard_save_dir):
            print(f"跳过 OCR：目录 {blackboard_save_dir} 不存在")
            return []

        print(f"准备唤醒 OCR 独立进程...")
        
        final_output_json = os.path.join(os.path.dirname(blackboard_save_dir), "ocr.json")
        worker_script = os.path.join(os.path.dirname(__file__), "paddle_worker.py")
        
        # 构造命令
        cmd = [
            sys.executable,
            worker_script,
            str(blackboard_save_dir),
            str(self.fps),
            str(final_output_json)
        ]
        
        # 如果有帧列表，作为 JSON 字符串传递
        if frame_list is not None:
            cmd.append(json.dumps(frame_list))
            print(f"📌 [OCR] 传递帧列表，数量: {len(frame_list)}")
        
        try:
            subprocess.run(cmd, check=True)
            if os.path.exists(final_output_json):
                with open(final_output_json, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                os.remove(final_output_json)  # 清理临时文件
                return results
            return []
        except subprocess.CalledProcessError as e:
            print(f"❌ OCR 子进程运行失败: {e}")
            return []
    

    def emotion_recognition(self, face_save_dir):
        if not face_save_dir or not os.path.exists(face_save_dir):
            print(f"跳过情绪识别：目录 {face_save_dir} 不存在")
            return []
        
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        print(f"开始情绪识别：{face_save_dir}")
        image_paths = sorted(Path(face_save_dir).glob("*.png"))
        results = []

        # ==========================================
        # 🔥 策略一：定义课堂环境下的“情绪唤醒度”权重
        # ==========================================
        arousal_weights = {
            'surprise': 1.0,  # 重点强调 / 发现新知
            'happy': 0.9,     # 互动活跃 / 氛围好
            'angry': 0.8,     # 情绪激动 / 维持纪律
            'fear': 0.6,
            'disgust': 0.4,
            'sad': 0.3,
            'neutral': 0.1    # 平静讲授 (基线)
        }

        # 翻译字典，用于供给 LLM 更好的中文 Prompt
        en_to_zh = {
            'angry': '严肃/生气', 'disgust': '厌恶', 'fear': '担忧/恐惧',
            'happy': '高兴/喜悦', 'sad': '低落/悲伤', 'surprise': '惊讶/强调', 
            'neutral': '平静/中性'
        }

        for img_path in tqdm(image_paths, desc="情绪提取与多维分析", unit="帧"):
            frame_idx = int(img_path.stem)
            timestamp = frame_idx / self.fps

            arousal_score = 0.1 # 默认兜底分数
            top_emotions_text = "未知"
            
            try:
                objs = DeepFace.analyze(
                    img_path=str(img_path),
                    actions=['emotion'],
                    enforce_detection=False
                )
                
                # 提取识别结果
                face_data = objs[0] if isinstance(objs, list) and len(objs) > 0 else (objs if isinstance(objs, dict) else None)
                
                if face_data and 'emotion' in face_data:
                    emotion_dict = face_data['emotion'] # 这是一个包含 7 个 key 的字典，总和约为 100
                    
                    # 1. 计算加权活跃度分数 (Arousal Score)
                    weighted_sum = sum(emotion_dict[emo] * arousal_weights.get(emo, 0.1) for emo in emotion_dict)
                    # DeepFace的值是百分制(0-100)，除以100将其缩放到 0-1 之间
                    arousal_score = weighted_sum / 100.0 
                    
                    # 2. 提取 Top-2 情绪及其百分比 (用于生成详尽的 LLM Prompt)
                    # 按得分从高到低排序
                    sorted_emotions = sorted(emotion_dict.items(), key=lambda item: item[1], reverse=True)
                    
                    top1_name, top1_score = sorted_emotions[0]
                    top2_name, top2_score = sorted_emotions[1]
                    
                    # 如果第二大情绪占比超过 15%，我们认为这是一个“复合情绪”，否则只看 Top1
                    if top2_score > 15.0:
                        top_emotions_text = f"主要{en_to_zh[top1_name]}({top1_score:.0f}%)伴随{en_to_zh[top2_name]}({top2_score:.0f}%)"
                    else:
                        top_emotions_text = f"显著{en_to_zh[top1_name]}"

            except Exception as e:
                pass # 保持日志干净
            
            # print({
            #     'frame_idx': frame_idx,
            #     'timestamp': timestamp,
            #     'arousal_score': round(arousal_score, 4), # 活跃度数值，可用于画曲线或算方差
            #     'emotion_desc': top_emotions_text }        # 复合文本描述，直接喂给 LLM
            # )
            results.append({
                'frame_idx': frame_idx,
                'timestamp': timestamp,
                'arousal_score': round(arousal_score, 4), # 活跃度数值，可用于画曲线或算方差
                'emotion_desc': top_emotions_text         # 复合文本描述，直接喂给 LLM
            })

        # ==========================================
        # 🔥 全局 Min-Max 归一化 (针对 arousal_score)
        # ==========================================
        valid_scores = [res['arousal_score'] for res in results if res['arousal_score'] is not None]

        if valid_scores:
            min_val = min(valid_scores)
            max_val = max(valid_scores)
            value_range = max_val - min_val

            for res in results:
                if value_range > 0:
                    norm_val = (res['arousal_score'] - min_val) / value_range
                    res['arousal_score'] = round(norm_val, 4)
                else:
                    res['arousal_score'] = 0.5 # 零方差保护

        print(f"多维情绪分析完成，处理 {len(results)} 帧")
        return results
    
    # def emotion_recognition(self, face_save_dir):
        if not face_save_dir or not os.path.exists(face_save_dir):
            print(f"跳过情绪识别：目录 {face_save_dir} 不存在")
            return []
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')

        print(f"开始情绪识别：{face_save_dir}")
        image_paths = sorted(Path(face_save_dir).glob("*.png"))
        results = []

        for img_path in tqdm(image_paths,desc="情绪识别", unit="帧"):
            frame_idx = int(img_path.stem)
            timestamp = frame_idx / self.fps

            try:
                objs = DeepFace.analyze(
                    img_path=str(img_path),
                    actions=['emotion'],
                    enforce_detection=False
                )
                # DeepFace 可能返回列表（多张脸）或字典（单张脸），统一处理
                if isinstance(objs, list) and len(objs) > 0:
                    dominant = objs[0]['dominant_emotion']
                elif isinstance(objs, dict):
                    dominant = objs.get('dominant_emotion')
                else:
                    dominant = None
            except Exception as e:
                print(f"情绪识别出错 {img_path.name}: {e}")
                dominant = None

            results.append({
                'frame_idx': frame_idx,
                'timestamp': timestamp,
                'dominant_emotion': dominant
            })

        print(f"情绪识别完成，处理 {len(results)} 帧")
        return results
    
    def cleanup(self):
        """显式释放底层 C++ 资源，防止析构报错"""
        if hasattr(self, 'mediapipe_detector') and self.mediapipe_detector is not None:
            try:
                self.mediapipe_detector.close()
                print("✅ MediaPipe 姿态检测器已安全释放。")
            except Exception:
                pass

    def _save_json(self, frames, output_path):
        """
        通用 JSON 保存函数（全字段、高精度时间戳模式）。
        
        参数:
            frames: 包含帧信息的字典列表。每个字典将被直接保存，仅处理时间戳格式。
            output_path: JSON 保存路径
        """
        if not frames:
            print(f"跳过保存：未接收到有效帧数据。")
            return

        # 1. 格式化时间戳并清理无效帧
        # 确保按帧序号顺序处理
        frames.sort(key=lambda x: x['frame_idx'])
        
        formatted_frames = []
        for frame in frames:
            # 深拷贝字典，避免修改原对象
            new_frame = frame.copy()
            
            # 检查是否有有效的时间戳
            if 'timestamp' not in new_frame:
                continue
                
            # 2. 统一时间戳格式（保留4位小数，对应毫秒级精度）
            ts = new_frame['timestamp']
            new_frame['start'] = round(ts, 4)
            # 如果存在 'end' 字段且与 start 相同，则删除 end，或者也设为相同值
            # 为了配合你的手势数据格式，我们强制让 end = start
            new_frame['end'] = round(ts, 4)
            
            # 3. 移除 Python 特有的非 JSON 兼容类型（如有）
            # 这里可以添加对 numpy 类型的转换，防止保存报错
            for k, v in new_frame.items():
                if isinstance(v, (int, float, str, bool, type(None))):
                    continue
                elif isinstance(v, (list, tuple, dict)):
                    # 递归处理或保持原样
                    pass
                else:
                    # 转换为字符串或浮点数
                    try:
                        new_frame[k] = float(v)
                    except:
                        new_frame[k] = str(v)
                        
            formatted_frames.append(new_frame)

        # 4. 落盘保存
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_frames, f, ensure_ascii=False, indent=2)

        print(f"已保存 JSON：{output_path}，共保存 {len(formatted_frames)} 个时间点数据")


def main():
    # 配置参数（请根据实际路径修改）
    sam2_checkpoint = "./model/sam2/checkpoints/sam2.1_hiera_large.pt"
    sam2_model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    input_dir = "./data/processed/visual/frames/"
    output_dir = "./data/processed/visual/masks"
    
    # 请确保 MediaPipe 的 task 文件确实存在于此路径
    MODEL_PATH = "./checkpoints/mediapipe/pose_landmarker_heavy.task" 
    
    video_name = "lecture_sample"
    fps = 0.2  
    prompt = "请提取黑板上的文字内容，仅输出文字即可。"

    # 实例化并运行
    vp = visual_percepter(
        sam2_checkpoint=sam2_checkpoint,
        sam2_model_cfg=sam2_model_cfg,
        input_dir=input_dir,
        output_dir=output_dir,
        model_path=MODEL_PATH,
        fps=fps,
        prompt=prompt
    )
    
    vp.forward(video_name)

if __name__ == "__main__":
    main()