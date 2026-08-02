import os
import json
import numpy as np
from pathlib import Path
from scipy.ndimage import label
import sys

# 第三方音频与模型库
import whisper
import librosa
import opensmile
import pandas as pd
import subprocess

class audio_percepter():
    def __init__(self, model_size="base", input_path=None, output_path=None):
        print(f"正在加载 Whisper {model_size} 模型...")
        self.model = whisper.load_model(model_size)
        self.input_path = input_path          # 单个音频文件路径
        self.output_path = output_path        # 输出 JSON 文件路径

    def whisper_transcribe(self, input_audio_path=None, output_file_path=None):
        print(f"开始语音转文本：{input_audio_path}")
        
        import torch
        with torch.autocast("cuda", enabled=False):
            result = self.model.transcribe(
                input_audio_path, 
                word_timestamps=True, 
                verbose=False,
                fp16=False
            )

        segments = []
        for segment in result['segments']:
            segments.append({
                "start": segment['start'],
                "end": segment['end'],
                "text": segment['text'].strip()
            })

        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)

        print(f"转录完成，共 {len(segments)} 句，已保存至 {output_file_path}")
        return segments
    
    def _aggregate_opensmile(self, df_features,window_size = 2.0):
        """
        【修改版】返回逐帧的音频情感分数列表
        """
        if df_features is None or df_features.empty:
            return []

        # 1. 选取关键特征
        target_cols = ['loudness_sma3', 'F0semitoneFrom27.5Hz_sma3', 'alphaRatio_sma3']
        available_cols = [col for col in target_cols if col in df_features.columns]
        
        if not available_cols:
            return []

        emotion_data = df_features[available_cols]
        
        # 2. 全局归一化 (基于整段音频的最大最小值，保证分数在 0-1 之间)
        min_vals = emotion_data.min()
        max_vals = emotion_data.max()
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1 
        
        normalized_data = (emotion_data - min_vals) / range_vals
        
        # 3. 加权计算每一帧的分数
        weights = np.ones(len(available_cols)) / len(available_cols)
        if 'loudness_sma3' in available_cols:
            weights[available_cols.index('loudness_sma3')] = 0.5
        weights = weights / weights.sum()
        
        # 计算每一行的加权和 -> 得到时间序列分数
        frame_scores = (normalized_data * weights).sum(axis=1)
        
         # --- 【新增】窗口合并逻辑 ---
        result_list = []
        frame_duration = 0.01 # 假设帧率 100Hz
        
        # 每隔 window_size 秒取一个平均值
        # 比如 window_size=2.0，则每 200 帧合并为 1 条数据
        step_frames = int(window_size / frame_duration)
        
        for i in range(0, len(frame_scores), step_frames):
            # 截取当前窗口的分数
            window_scores = frame_scores[i : i + step_frames]
            if len(window_scores) == 0: continue
            
            # 计算当前窗口的平均分
            avg_score = window_scores.mean()
            
            # 计算当前窗口的时间
            t_start = i * frame_duration
            t_end = min((i + step_frames) * frame_duration, len(frame_scores) * frame_duration)
            
            result_list.append({
                "start": round(t_start, 3),
                "end": round(t_end, 3),
                "audio_emotion": float(np.clip(avg_score, 0, 1))
            })
            
        return result_list

    def _aggregate_librosa(self, input_audio_path, emphasis_regions, window_size = 2.0):
        """
        【修改版】返回逐帧的音频特征分数列表
        """
        try:
            y, sr = librosa.load(input_audio_path, sr=16000)
        except Exception as e:
            print(f"音频读取失败: {e}")
            return []

        # 1. 计算 RMS (能量)
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        times = librosa.times_like(rms, sr=sr, hop_length=hop_length)

        # 2. 归一化 (基于整段音频)
        rms_min, rms_max = np.min(rms), np.max(rms)
        if rms_max - rms_min > 1e-9:
            rms_norm = (rms - rms_min) / (rms_max - rms_min)
        else:
            rms_norm = np.zeros_like(rms)

         # --- 【新增】窗口合并逻辑 ---
        result_list = []
        
        # 这里的 hop_length/sr 是每帧的时间长度
        frame_duration = hop_length / sr
        step_frames = int(window_size / frame_duration)
        
        for i in range(0, len(times), step_frames):
            # 防止越界
            if i >= len(rms_norm): break
            
            # 获取当前窗口的能量平均值
            # 注意：这里简单取当前帧的值，或者也可以像上面那样取窗口平均
            current_score = rms_norm[i] 
            
            t_start = times[i]
            t_end = t_start + window_size
            
            result_list.append({
                "start": round(t_start, 3),
                "end": round(t_end, 3),
                "audio_feature": float(np.clip(current_score, 0, 1))
            })
            
        return result_list

    # def opensmile_emotion_recognition(self, input_audio_path=None):
    #     print("开始提取 OpenSMILE eGeMAPS 声学情感特征...")
    #     try:
    #         y, sr = librosa.load(input_audio_path, sr=16000, mono=True)
    #     except Exception as e:
    #         print(f"音频读取失败: {e}")
    #         return None

    #     y = np.ascontiguousarray(y, dtype=np.float32)

    #     smile = opensmile.Smile(
    #         feature_set=opensmile.FeatureSet.eGeMAPSv02,
    #         feature_level=opensmile.FeatureLevel.LowLevelDescriptors 
    #     )
        
    #     df_features = smile.process_signal(y, sr)
    #     print(f"OpenSMILE 特征提取完成，共提取 {len(df_features)} 帧特征。")
    #     return df_features
    
    def opensmile_emotion_recognition(self, input_audio_path=None):
        """
        通过唤醒独立子进程来提取 OpenSMILE 特征，彻底规避内存冲突。
        """
        if not input_audio_path or not os.path.exists(input_audio_path):
            print(f"❌ 错误：音频文件 {input_audio_path} 不存在")
            return None

        print("开始唤醒 OpenSMILE 独立进程...")

        # 1. 自动生成输出 JSON 路径 (存放在音频同级目录下)
        output_json = input_audio_path.replace(".wav", "_smile_features.json").replace(".mp3", "_smile_features.json")
        
        # 2. 定位 worker 脚本路径
        worker_script = os.path.join(os.path.dirname(__file__), "opensmile_worker.py")
        
        # 3. 构造子进程命令
        cmd = [
            sys.executable,     # 使用当前环境的 Python
            worker_script,
            str(input_audio_path),
            str(output_json)
        ]

        try:
            # 4. 执行并等待
            # 使用 check=True，如果子进程 Segfault，这里会抛出异常
            subprocess.run(cmd, check=True)

            # 5. 读取生成的 JSON 数据并转回 DataFrame (保持与你原有逻辑一致)
            if os.path.exists(output_json):
                with open(output_json, 'r', encoding='utf-8') as f:
                    data_list = json.load(f)
                
                # 转回 DataFrame，方便后续计算（如归一化、滑窗等）
                df_features = pd.DataFrame(data_list)
                
                # 清理临时 JSON 文件（如果需要保留调试，可以注释掉这行）
                # os.remove(output_json) 
                
                print(f"✅ OpenSMILE 特征提取成功：获得 {len(df_features)} 帧数据")
                return df_features
            else:
                return None

        except subprocess.CalledProcessError as e:
            print(f"❌ OpenSMILE 子进程运行失败：{e}")
            return None
        except Exception as e:
            print(f"❌ 模块读取/转换失败：{e}")
            return None
    
    def feature_extraction(self, input_audio_path):
        print("开始提取基础声学特征 (能量与基频)...")
        y, sr = librosa.load(input_audio_path, sr=16000)
        
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C3'),
            fmax=librosa.note_to_hz('C6'),
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length
        )

        times_rms = librosa.times_like(rms, sr=sr, hop_length=hop_length)
        
        rms_norm = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-9)
        f0_clean = np.nan_to_num(f0)
        f0_norm = (f0_clean - np.min(f0_clean)) / (np.max(f0_clean) - np.min(f0_clean) + 1e-9)

        emphasis_score = 0.7 * rms_norm + 0.3 * f0_norm

        threshold = np.percentile(emphasis_score, 90)  
        emphasis_mask = emphasis_score > threshold

        labeled, n_regions = label(emphasis_mask)
        regions = []
        for i in range(1, n_regions + 1):
            idxs = np.where(labeled == i)[0]
            if len(idxs) > 0:
                t_start = times_rms[idxs[0]]
                t_end = times_rms[min(idxs[-1] + 1, len(times_rms) - 1)]
                if t_end - t_start > 0.2:
                    regions.append((t_start, t_end))
                    
        print(f"基础声学特征提取完成，共发现 {len(regions)} 处高能强调片段。")
        return regions

    def forward(self, audio_file_name=None, output_file_name=None):
        """
        处理单个音频文件，生成转录文本和音频特征 JSON。
        
        参数:
            audio_file_name: 音频文件名，如 "1_raw_video_00.wav"
            output_file_name: 已废弃，保留仅为兼容，实际会根据 audio_file_name 自动生成
        返回:
            transcripts: 转录结果列表
            df_features: OpenSMILE 特征 DataFrame
            emphasis_regions: 高能强调片段列表
            audio_features_data: 音频特征字典（包含 emotion 和 feature 分数）
        """
        if audio_file_name is None:
            raise ValueError("请提供 audio_file_name")

        # 1. 解析视频基础名（去掉扩展名）
        video_name = os.path.splitext(audio_file_name)[0]   # 例如 "1_raw_video_00"

        # 2. 构建输入音频完整路径
        input_audio_path = os.path.join(self.input_path, audio_file_name)
        if not os.path.exists(input_audio_path):
            raise FileNotFoundError(f"音频文件不存在: {input_audio_path}")

        # 3. 输出目录：self.output_path 通常是 data/processed/audio/transcribed
        os.makedirs(self.output_path, exist_ok=True)

        # 4. 构建两个输出文件路径
        transcript_json_path = os.path.join(self.output_path, f"{video_name}_transcript.json")
        audio_features_json_path = os.path.join(self.output_path, f"{video_name}_transcript_audio_features_.json")

        # --- 步骤1: 语音转文字转录 ---
        transcripts = self.whisper_transcribe(input_audio_path, transcript_json_path)

        # --- 步骤2: 提取声学特征（能量、基频）和 OpenSMILE 情感特征 ---
        emphasis_regions = self.feature_extraction(input_audio_path)
        df_features = self.opensmile_emotion_recognition(input_audio_path)

        # --- 步骤3: 将 OpenSMILE 和 Librosa 特征聚合成逐段分数列表 ---
        audio_emotion_list = self._aggregate_opensmile(df_features)
        audio_feature_list = self._aggregate_librosa(input_audio_path, emphasis_regions)

        # --- 步骤4: 保存音频特征 JSON（使用新命名）---
        audio_features_data = {
            "meta": {
                "audio_file": audio_file_name,
                "source_audio_path": input_audio_path,
                "video_name": video_name
            },
            "scores": {
                "audio_emotion": audio_emotion_list,
                "audio_feature": audio_feature_list
            },
            "debug_info": {
                "emphasis_regions": emphasis_regions
            }
        }

        with open(audio_features_json_path, 'w', encoding='utf-8') as f:
            json.dump(audio_features_data, f, ensure_ascii=False, indent=2)

        print(f"✅ 转录文件已保存: {transcript_json_path}")
        print(f"✅ 音频特征已保存: {audio_features_json_path}")

        return transcripts, df_features, emphasis_regions, audio_features_data
    

def main():
    input_path = "/home/featurize/work/LessonNet/Lesson_Net/data/processed/audio/"
    output_path = "/home/featurize/work/LessonNet/Lesson_Net/data/processed/audio/transcribed/"
    audio_file = "1_raw_video_00.wav" 
    output_file = "whisper_result.json" # 这个文件名现在只用于文本输出
        
    if not os.path.exists(input_path):
        print(f"未找到音频文件: {input_path}")
        return
    
    ap = audio_percepter(model_size="base", input_path=input_path, output_path=output_path)
    
    # 执行全流程
    # 注意：这里 output_file 仅作为文本文件的输出名
    transcripts, df_features, emphasis_regions, audio_features = ap.forward(audio_file, output_file)
    
    # 打印结果验证
    print("\n--- 处理结果摘要 ---")
    print(f"文本文件保存为: {os.path.join(output_path, output_file)}")
    print(f"特征文件保存为: {os.path.join(output_path, 'audio_features.json')}")

if __name__ == "__main__":
    main()