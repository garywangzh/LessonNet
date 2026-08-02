import os
import json
from .visual_percepter import visual_percepter   # 确保相对导入正确
from .audio_percepter import audio_percepter     # 确保相对导入正确
import torch
from pathlib import Path
from openai import OpenAI
import base64
import re
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class LessonNet:
    def __init__(self,
                 # 视觉基础配置
                 video_name=None,
                 sam2_checkpoint="./model/sam2/checkpoints/sam2.1_hiera_large.pt",
                 sam2_model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
                 visual_input_dir="./data/processed/visual/frames/",
                 visual_output_dir="./data/processed/visual/masks",
                 model_path="./checkpoints/mediapipe/pose_landmarker_heavy.task",
                 visual_prompt="请提取黑板上的文字内容，仅输出文字即可。",
                 # 音频基础配置
                 audio_model_size="base",
                 audio_input_dir="./data/processed/audio/",
                 audio_output_dir="./data/processed/audio/transcribed/",
                 # 大模型配置 (默认连接 DeepSeek API；密钥通过环境变量 DEEPSEEK_API_KEY 提供)
                 llm_base_url="https://api.deepseek.com",
                 llm_api_key=None,
                 llm_model_name="deepseek-chat", #deepseek-reasoner
                 device=None):
        """
        初始化 LessonNet，统筹视觉、音频与大模型融合模块。
        """
        if llm_api_key is None:
            llm_api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        self.visual_input_dir = visual_input_dir
        self.visual_output_dir = visual_output_dir
        self.audio_input_dir = audio_input_dir
        self.audio_output_dir = audio_output_dir
        
        self.audio_model_size = audio_model_size
        # 动态实例化 AP，确保路径正确
        self.ap = audio_percepter(
            model_size=self.audio_model_size,
            input_path=self.audio_input_dir,
            output_path=self.audio_output_dir
        )
        
        
        # 提前实例化大模型客户端
        self.llm_client = OpenAI(base_url=llm_base_url, api_key=llm_api_key, timeout=3600)
        self.llm_model_name = llm_model_name

        # 初始化视觉感知器 (常驻内存)
        print(">>> 正在初始化视觉感知器 (Visual Perceptor)...")
        self.vp = visual_percepter(
            sam2_checkpoint=sam2_checkpoint,
            sam2_model_cfg=sam2_model_cfg,
            input_dir=self.visual_input_dir,
            output_dir=self.visual_output_dir,
            model_path=model_path,
            prompt=visual_prompt,
            device=device
        )
        
        # 存放处理结果的数据结构
        self.visual_save_dir = None
        self.audio_transcripts = None
        self.audio_emphasis_regions = None
        self.video_name = video_name
        self.base_processed_dir = Path("./data/processed")
        self.processed_root = self.base_processed_dir
        self.visual_masks_dir = self.base_processed_dir / "visual" / "masks" / video_name if video_name else None
        self.audio_transcribed_dir = self.base_processed_dir / "audio" / "transcribed"
        self.visual_root_dir = self.base_processed_dir / "visual"

    def VisualPerceptor(self, video_name, sam2_bboxes=None, use_original_frame_for_ocr=False):
        print(f"\n{'='*40}\n[第一阶段] 启动视觉分析: {video_name}\n{'='*40}")
        self.vp.forward(
            video_name, 
            sam2_bboxes=sam2_bboxes,
            use_original_frame_for_ocr=use_original_frame_for_ocr
        )
        self.visual_save_dir = os.path.join(self.visual_output_dir, video_name)

    def AudioPerceptor(self, audio_filename):
        """调用音频处理模块"""
        print(f"\n{'='*40}\n[第二阶段] 启动音频分析: {audio_filename}\n{'='*40}")
        # audio_path = os.path.join(self.audio_input_dir, audio_filename)
        # base_name = os.path.splitext(audio_filename)[0]
        # audio_output_file = os.path.join(self.audio_output_dir, f"{base_name}_result.json")

        # 接收音频特征
        transcripts, _, emphasis_regions, _ = self.ap.forward(audio_filename)
        self.audio_transcripts = transcripts
        self.audio_emphasis_regions = emphasis_regions

    def _load_json(self, filepath):
        """辅助函数：安全加载 JSON 文件"""
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def ai_fusion(self):
        # 构建路径
        video_visual_dir = self.visual_masks_dir          # data/processed/visual/masks/{video_name}
        gesture_path = video_visual_dir / "gesture.json"
        emotion_path = video_visual_dir / "emotion.json"
        ocr_path = video_visual_dir / "ocr.json"
        
        # 音频文件路径（按新命名规则）
        transcript_path = self.audio_transcribed_dir / f"{self.video_name}_transcript.json"
        audio_feat_path = self.audio_transcribed_dir / f"{self.video_name}_transcript_audio_features.json"
        
        # 读取数据
        gesture_data = self._load_json(gesture_path)
        emotion_data = self._load_json(emotion_path)
        ocr_data = self._load_json(ocr_path)
        
        # 读取音频转录（也可以直接从 self.audio_transcripts 取，但保险起见读文件）
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcripts = json.load(f)
        
        # 读取音频特征（获取 emphasis_regions 等）
        with open(audio_feat_path, 'r', encoding='utf-8') as f:
            audio_feat = json.load(f)
            emphasis_regions = audio_feat.get("debug_info", {}).get("emphasis_regions", [])

        # 2. 组装 Prompt
        prompt = "你是一个专业的高中课堂AI助教。以下是我提取的一段课堂教学的多模态特征时间线数据。\n\n"
        prompt += "### 任务要求\n"
        prompt += "请根据这些数据，为学生生成一份详细的【课后复习问答】报告。\n\n"
        prompt += "### 报告结构\n"
        prompt += "1. **核心知识点问答**：基于板书和语音内容，提出3-5个关键问题，并在报告末尾统一给出答案。\n"
        prompt += "2. **课堂内容分段精讲**：将课堂划分为若干逻辑片段，对每个片段中的知识点做详细阐释。\n\n"
        prompt += "### 特殊标记规则（重要！）\n"
        prompt += "当你在报告中引用板书内容时，请使用以下格式标记时间点，后台将根据这些标记自动截取对应板书图片：\n"
        prompt += "`[板书截图: HH:MM:SS]` 例如：`[板书截图: 00:02:45]`\n\n"
        prompt += "**注意**：\n"
        prompt += "- 时间点精确到秒即可，后台会从该时间点前后2秒范围内寻找最佳板书帧\n"
        prompt += "- 只有你明确指出板书内容和对应时间点时，后台才会截取并插入图片\n"
        prompt += "- 如果没有合适的板书画面，可以不使用该标记\n\n"
        prompt += "### 输出示例\n"
        prompt += "```\n"
        prompt += "#### 片段一：函数定义与图像（00:02:30 - 00:06:00）\n"
        prompt += "本节课首先介绍了二次函数的标准形式，老师重点强调了系数a的作用。\n"
        prompt += "[板书截图: 00:03:15] 此时老师画出了开口向上的抛物线示意图。\n"
        prompt += "老师强调：'当a>0时，抛物线开口向上，函数有最小值。'\n"
        prompt += "```\n\n"
        prompt += "请严格按照上述格式输出报告，确保板书标记清晰可解析。"
        prompt += "【多模态时间线数据如下】：\n"
        
        # 写入板书信息 (OCR)
        prompt += "\n--- 视觉板书 (OCR) --- \n"
        for item in ocr_data:
            if item.get('text'):
                prompt += f"[{item['start']:.1f}秒 - {item['end']:.1f}秒] 板书内容更新: {item['text']}\n"
                
        # 写入教师情绪与手势
        prompt += "\n--- 教师状态 (手势与情绪) --- \n"
        for item in gesture_data:
            prompt += f"[{item['start']:.1f}秒 - {item['end']:.1f}秒] 教师动作: {item['arm_raise']}\n"
        for item in emotion_data:
            prompt += f"[{item['start']:.1f}秒 - {item['end']:.1f}秒] 教师情绪: {item['emotion_desc']}\n"

        # 写入语音文本，并标注重音
        prompt += "\n--- 教师语音内容 --- \n"
        for seg in self.audio_transcripts:
            t_start = seg['start']
            t_end = seg['end']
            text = seg['text']
            
            # 检查该句话是否落在“重音/强调”区间内
            is_emphasis = False
            for region in self.audio_emphasis_regions:
                # 安全处理：确保能正确解析 region
                if isinstance(region, (list, tuple)):
                    if len(region) >= 2:
                        r_start = float(region[0])
                        r_end = float(region[1])
                    elif len(region) == 1:
                        r_start = float(region[0])
                        r_end = r_start
                    else:
                        continue
                elif isinstance(region, dict):
                    r_start = float(region.get('start', region.get('timestamp', 0)))
                    r_end = float(region.get('end', r_start))
                else:
                    continue
                
                # 只要有一半的时间重叠，就认为是重点语气
                overlap = max(0, min(t_end, r_end) - max(t_start, r_start))
                if overlap > (t_end - t_start) * 0.3:
                    is_emphasis = True
                    break
                    
            prefix = "【重点语气！】" if is_emphasis else ""
            prompt += f"[{t_start:.1f}秒 - {t_end:.1f}秒] {prefix} {text}\n"

        self.fused_prompt = prompt
        print("特征融合完毕，Prompt 已生成。")
        
        # 确定保存路径 (你可以根据你实际的目录结构修改这里的基础路径)
        # 尝试获取类中已有的 output_dir，如果没有则默认保存在当前目录下的 data 文件夹
        save_dir = getattr(self, 'output_dir', './data/processed')
        os.makedirs(save_dir, exist_ok=True)
        fusion_json_path = os.path.join(save_dir, "fused_prompt.json")

        # 构造 JSON 数据结构
        # 封装成字典，方便以后如果想加入 "视频长度"、"时间戳" 等元数据时随时扩展
        fusion_data = {
            "task_name": "LessonNet_Multimodal_Fusion",
            "fused_prompt": self.fused_prompt
        }
        
        try:
            with open(fusion_json_path, 'w', encoding='utf-8') as f:
                # indent=2 保证 JSON 格式有缩进，人类肉眼可读
                json.dump(fusion_data, f, ensure_ascii=False, indent=2)
            print(f"✅ 融合结果快照已成功保存至：{fusion_json_path}")
        except Exception as e:
            print(f"❌ 保存融合结果 JSON 时出错: {e}")
            
        return self.fused_prompt

    
    def nn_fusion(self, vis_emo_path, vis_ges_path, audio_feat_path, ocr_path, transcripts_data=None, output_dir=None):
        """
        融合网络推理逻辑，并将生成的最终 Prompt 保存到本地。
        
        参数:
            vis_emo_path: emotion.json 路径
            vis_ges_path: gesture.json 路径
            audio_feat_path: audio_features.json 路径（按视频命名）
            ocr_path: ocr.json 路径
            transcripts_data: 可选的语音转录文本
            output_dir: 保存 prompt 快照的目录，默认为 vis_emo_path 所在目录的父目录的父目录（即 visual 根目录）
        """
        print(f"\n{'='*40}\n[第三阶段] 特征融合与 Prompt 生成\n{'='*40}")
        
        # 1. 运行 FusionNet 得到特征矩阵
        features_matrix = fuse_multimodal_features(vis_emo_path, vis_ges_path, audio_feat_path, ocr_path)
        
        # 2. 神经网络评分
        model = ImportanceScorer(input_dim=5)
        model.eval()

        all_scores = []
        high_value_segments = []
        window_size = 10 

        with torch.no_grad():
            for i in range(features_matrix.shape[0]):
                feature_vec = features_matrix[i]
                if np.all(feature_vec == 0): continue

                score = model(torch.tensor([feature_vec], dtype=torch.float32)).item()
                start_time = i * window_size
                time_str = f"{int(start_time//60):02d}:{int(start_time%60):02d}-{int((start_time+window_size)//60):02d}:{int((start_time+window_size)%60):02d}"
                
                all_scores.append(score)
                high_value_segments.append({
                    "time": time_str,
                    "score": score,
                    "features": feature_vec.tolist()
                })

        # 3. 动态阈值过滤
        threshold = np.mean(all_scores) + np.std(all_scores) if all_scores else 0.5
        selected_text = ""
        filtered_segments = []

        for seg in high_value_segments:
            if seg['score'] >= threshold:
                filtered_segments.append(seg)
                v_emo, v_ges, a_emo, a_feat, ocr_val = seg['features']
                desc = []
                if v_emo > 0.6: desc.append("情绪波动显著")
                if v_ges > 0.5: desc.append("肢体动作丰富")
                if ocr_val > 0.5: desc.append("板书信息增量大")
                desc_str = "，".join(desc) if desc else "多模态综合指标高"
                selected_text += f"- **[{seg['time']}]** (评分: {seg['score']:.2f}): {desc_str}\n"

        # 4. 组装终极 Prompt
        prompt = f"""你是一个资深的教育AI专家。以下是通过神经网络分析得出的课堂高光片段。
                    请根据这些时间点及其特征，结合附录中的语音文本，生成一份深度教学评估报告。

                    ### 1. 神经网络识别的高光时刻
                    {selected_text}

                    ### 2. 报告要求
                    - 分析教师在这些时刻的教学表现。
                    - 总结课堂的知识重点分布。
                    - 给出针对性的改进建议。
                    """
        if transcripts_data:
            prompt += f"\n### 附录：语音转录内容\n{transcripts_data}\n"

        # 5. 持久化保存
        self.fused_prompt = prompt
        
        # 确定保存目录
        if output_dir is None:
            # 默认：取 vis_emo_path 的父目录的父目录（即 visual 根目录）
            save_dir = os.path.dirname(os.path.dirname(vis_emo_path))
        else:
            save_dir = output_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 从 vis_emo_path 或 audio_feat_path 提取视频名（用于文件名）
        # 假设 audio_feat_path 包含视频名，如 "1_raw_video_00_transcript_audio_features.json"
        base_name = os.path.basename(audio_feat_path).replace("_transcript_audio_features.json", "")
        if not base_name:
            base_name = "unknown"
        
        prompt_snapshot_path = os.path.join(save_dir, f"{base_name}_fused_prompt_debug.txt")
        data_snapshot_path = os.path.join(save_dir, f"{base_name}_fusion_data.json")
        
        try:
            with open(prompt_snapshot_path, 'w', encoding='utf-8') as f:
                f.write(self.fused_prompt)
            with open(data_snapshot_path, 'w', encoding='utf-8') as f:
                json.dump({"threshold": threshold, "segments": filtered_segments}, f, ensure_ascii=False, indent=2)
            print(f"✅ Prompt 快照已保存至: {prompt_snapshot_path}")
        except Exception as e:
            print(f"⚠️ Prompt 保存失败（但不影响推理）: {e}")

        return self.fused_prompt
    
    def output(self, report_save_path=None):
        """
        生成并保存报告，将 [板书截图: xxx] 标记替换为 Base64 内嵌图片
        """
        if report_save_path is None and self.video_name:
            report_save_path = os.path.join(self.processed_root, self.video_name, "final_report.md")
        
        if not self.fused_prompt:
            print("⚠️ 没有生成 prompt，请先执行 ai_fusion 或 nn_fusion")
            return
        
        # 调用大模型生成报告
        messages = [
            {"role": "system", "content": "你是资深的教育AI专家，擅长将多模态课堂数据归纳为结构化报告。"},
            {"role": "user", "content": self.fused_prompt},
        ]
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model_name,
                messages=messages,
                max_tokens=4096,
                temperature=0.3
            )
            raw_report = response.choices[0].message.content
        except Exception as e:
            print(f"❌ 大模型生成失败: {e}")
            return
        
        # ---- 后处理：解析 [板书截图: xxx] 标记 ----
        plate_dir = os.path.join(os.path.dirname(report_save_path), "plate_images")
        os.makedirs(plate_dir, exist_ok=True)
        
        def image_to_base64(image_path):
            """将图片转换为 Base64 编码"""
            try:
                with open(image_path, "rb") as f:
                    return base64.b64encode(f.read()).decode('utf-8')
            except Exception as e:
                print(f"⚠️ 图片转Base64失败: {e}")
                return None
        
        def replace_plate_marker(match):
            time_str = match.group(1)
            abs_path = self._extract_plate_image(time_str, self.video_name, plate_dir)
            if abs_path and os.path.exists(abs_path):
                b64 = image_to_base64(abs_path)
                if b64:
                    return f'<img src="data:image/jpeg;base64,{b64}" alt="板书截图 {time_str}" style="max-width:100%; border:1px solid #ddd; border-radius:4px; margin:8px 0;">'
            # 截取失败，保留原标记（红色显示方便调试）
            return f'<span style="color:red;">[板书截图失败: {time_str}]</span>'
        
        # 正则替换
        pattern = r"\[板书截图:\s*(\d{2}:\d{2}:\d{2})\s*\]"
        final_report = re.sub(pattern, replace_plate_marker, raw_report)
        
        # 保存报告
        os.makedirs(os.path.dirname(report_save_path), exist_ok=True)
        with open(report_save_path, 'w', encoding='utf-8') as f:
            f.write(final_report)
        
        print(f"✅ 报告已保存至: {report_save_path}")        
    
    def _extract_plate_image(self, time_str, video_name, output_dir):
        """
        根据时间戳从原始帧中截取板书图片
        
        优先级：
        1. 先从 OCR 结果中找对应帧（如果有）
        2. 如果没有 OCR 匹配，直接从帧目录找最接近的帧
        
        参数:
            time_str: 时间字符串，如 "00:02:45"
            video_name: 视频名称
            output_dir: 图片保存目录
        
        返回:
            图片绝对路径，如果失败返回 None
        """
      
        # ==========================================
        # 1. 解析时间戳
        # ==========================================
        match = re.match(r"(\d{2}):(\d{2}):(\d{2})", time_str)
        if not match:
            print(f"⚠️ 时间格式错误: {time_str}")
            return None
        
        h, m, s = int(match[1]), int(match[2]), int(match[3])
        target_sec = h * 3600 + m * 60 + s
        print(f"📌 目标时间: {time_str} = {target_sec} 秒")
        
        # ==========================================
        # 2. 帧目录路径
        # ==========================================
        frame_dir = os.path.join(self.visual_input_dir, video_name)
        if not os.path.exists(frame_dir):
            print(f"⚠️ 帧目录不存在: {frame_dir}")
            return None
        
        # 获取所有帧
        frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(('.jpg', '.png'))])
        if not frame_files:
            print(f"⚠️ 帧目录为空: {frame_dir}")
            return None
        
        print(f"📌 帧目录: {frame_dir}, 共 {len(frame_files)} 帧")
        
        # ==========================================
        # 3. 从 manifest 获取实际 fps
        # ==========================================
        fps = 1
        manifest_path = os.path.join(self.base_processed_dir, f"{video_name}_manifest.json")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                    fps = manifest.get('extracted_fps', 1)
                    print(f"📌 从 manifest 读取 fps: {fps}")
            except Exception as e:
                print(f"⚠️ 读取 manifest 失败: {e}")
        
        # ==========================================
        # 4. 优先：从 OCR 结果中找对应帧
        # ==========================================
        ocr_json_path = os.path.join(self.visual_masks_dir, "ocr.json")
        closest_ocr_frame = None
        closest_ocr_diff = float('inf')
        
        if os.path.exists(ocr_json_path):
            try:
                with open(ocr_json_path, 'r', encoding='utf-8') as f:
                    ocr_data = json.load(f)
                print(f"📌 加载 OCR 数据，共 {len(ocr_data)} 条")
                
                for item in ocr_data:
                    # 获取 OCR 时间
                    ocr_time = item.get('timestamp')
                    if ocr_time is None:
                        ocr_time = item.get('start')
                    if ocr_time is None:
                        continue
                    
                    diff = abs(ocr_time - target_sec)
                    if diff < closest_ocr_diff:
                        closest_ocr_diff = diff
                        closest_ocr_frame = item
                
                # 如果 OCR 匹配成功（5秒内）
                if closest_ocr_frame is not None and closest_ocr_diff <= 5:
                    frame_idx = closest_ocr_frame.get('frame_idx')
                    print(f"📌 找到 OCR 匹配: 时间 {closest_ocr_frame.get('timestamp')} 秒, "
                        f"差 {closest_ocr_diff:.2f} 秒, frame_idx={frame_idx}")
                    
                    if frame_idx is not None:
                        # frame_idx 可能是 "00000.jpg" 或 0
                        if isinstance(frame_idx, str):
                            frame_idx = int(os.path.splitext(frame_idx)[0])
                        
                        # 构造帧路径
                        frame_path = os.path.join(frame_dir, f"{frame_idx:05d}.jpg")
                        if not os.path.exists(frame_path):
                            # 尝试 .png
                            frame_path = os.path.join(frame_dir, f"{frame_idx:05d}.png")
                        
                        if os.path.exists(frame_path):
                            # 保存图片
                            os.makedirs(output_dir, exist_ok=True)
                            output_filename = f"{video_name}_plate_{target_sec:06d}.jpg"
                            output_path = os.path.join(output_dir, output_filename)
                            shutil.copy2(frame_path, output_path)
                            print(f"✅ [OCR匹配] 截取板书图片: {output_filename}")
                            return output_path
                        else:
                            print(f"⚠️ OCR 帧不存在: {frame_path}")
            except Exception as e:
                print(f"⚠️ 读取 OCR 数据失败: {e}")
        
        # ==========================================
        # 5. 备选：直接从帧目录找最接近的帧
        # ==========================================
        print("📌 未找到 OCR 匹配，直接从帧目录查找...")
        
        # 计算目标帧索引
        target_frame_idx = int(target_sec * fps)
        print(f"📌 目标帧索引: {target_frame_idx}")
        
        # 找最接近的帧
        closest_frame = None
        closest_diff = float('inf')
        for f in frame_files:
            try:
                idx = int(os.path.splitext(f)[0])
                diff = abs(idx - target_frame_idx)
                if diff < closest_diff:
                    closest_diff = diff
                    closest_frame = f
            except ValueError:
                continue
        
        if closest_frame is None:
            print(f"⚠️ 未找到接近的帧")
            return None
        
        frame_path = os.path.join(frame_dir, closest_frame)
        if not os.path.exists(frame_path):
            return None
        
        print(f"📌 找到帧: {closest_frame}, 差 {closest_diff} 帧")
        
        # 保存图片
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"{video_name}_plate_{target_sec:06d}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        shutil.copy2(frame_path, output_path)
        
        print(f"✅ [帧匹配] 截取板书图片: {output_filename} (帧: {closest_frame})")
        return output_path

    def forward(self, video_name, audio_filename, report_save_path="data/processed/final_report.md", sam2_bboxes=None, annotation_frame_time=0, use_original_frame_for_ocr=False):
        print(">>> 启动 LessonNet 端到端流水线 <<<")
        self.AudioPerceptor(audio_filename=audio_filename)
        
        self.VisualPerceptor(
            video_name=video_name, 
            sam2_bboxes=sam2_bboxes,
            use_original_frame_for_ocr=use_original_frame_for_ocr
        )
        
        # 建议在跑完视觉后清理显存，防止随后的大模型推理 OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.ai_fusion()
        self.output(report_save_path=report_save_path)
        
        if hasattr(self, 'vp'):
            self.vp.cleanup()


# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    # 配置你的输入文件
    target_video = "1_raw_video_00"
    target_audio = "1_raw_video_00.wav"
    
    # 实例化并一键运行
    ln = LessonNet()
    ln.forward(
        video_name=target_video, 
        audio_filename=target_audio,
        report_save_path=f"data/processed/{target_video}_final_report.md"
    )
