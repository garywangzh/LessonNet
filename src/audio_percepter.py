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

class audio_percepter:
    def __init__(self, model_size="base", input_path=None, output_path=None):
        print(f"正在加载 Whisper {model_size} 模型...")
        self.model = whisper.load_model(model_size)
        self.input_path = input_path
        self.output_path = output_path

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
        for segment in result.get("segments", []):
            segments.append({
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": segment["text"].strip()
            })

        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)

        print(f"转录完成，共 {len(segments)} 句，已保存至 {output_file_path}")
        return segments

    def opensmile_emotion_recognition(self, input_audio_path=None):
        """
        通过独立子进程提取 OpenSMILE 特征，避免主进程内存冲突。
        要求同目录下存在 opensmile_worker.py。
        """
        if not input_audio_path or not os.path.exists(input_audio_path):
            print(f"❌ 错误：音频文件 {input_audio_path} 不存在")
            return None

        print("开始唤醒 OpenSMILE 独立进程...")

        base, _ = os.path.splitext(input_audio_path)
        output_json = f"{base}_smile_features.json"

        worker_script = os.path.join(os.path.dirname(__file__), "opensmile_worker.py")

        cmd = [
            sys.executable,
            worker_script,
            str(input_audio_path),
            str(output_json)
        ]

        try:
            subprocess.run(cmd, check=True)

            if not os.path.exists(output_json):
                print("❌ OpenSMILE worker 未生成输出 JSON")
                return None

            with open(output_json, "r", encoding="utf-8") as f:
                data_list = json.load(f)

            df_features = pd.DataFrame(data_list)

            print(f"✅ OpenSMILE 特征提取成功：获得 {len(df_features)} 帧数据")
            return df_features

        except subprocess.CalledProcessError as e:
            print(f"❌ OpenSMILE 子进程运行失败：{e}")
            return None
        except Exception as e:
            print(f"❌ OpenSMILE 结果读取失败：{e}")
            return None

    def extract_librosa_features(self, input_audio_path):
        """
        一次性读取音频并提取 RMS、F0、emphasis score。
        避免后续重复读取和重复归一化。
        """
        print("开始提取 Librosa 基础声学特征...")

        y, sr = librosa.load(input_audio_path, sr=16000, mono=True)

        frame_length = 2048
        hop_length = 512

        rms = librosa.feature.rms(
            y=y,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]

        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C3"),
            fmax=librosa.note_to_hz("C6"),
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length
        )

        times = librosa.times_like(rms, sr=sr, hop_length=hop_length)

        rms_norm = self._minmax_norm(rms)

        f0_clean = np.nan_to_num(f0, nan=0.0)
        f0_norm = self._minmax_norm(f0_clean)

        emphasis_score = 0.7 * rms_norm + 0.3 * f0_norm

        print("Librosa 基础声学特征提取完成。")

        return {
            "sr": sr,
            "hop_length": hop_length,
            "times": times,
            "rms_norm": rms_norm,
            "f0_norm": f0_norm,
            "emphasis_score": emphasis_score
        }

    @staticmethod
    def _minmax_norm(x):
        x = np.asarray(x, dtype=np.float32)
        x_min = np.min(x)
        x_max = np.max(x)
        denom = x_max - x_min

        if denom < 1e-9:
            return np.zeros_like(x, dtype=np.float32)

        return (x - x_min) / denom

    def detect_emphasis_regions(self, librosa_data, percentile=90, min_duration=0.2):
        """
        根据统一提取出的 emphasis_score 检测高能强调区域。
        """
        times = librosa_data["times"]
        emphasis_score = librosa_data["emphasis_score"]

        threshold = np.percentile(emphasis_score, percentile)
        emphasis_mask = emphasis_score > threshold

        labeled, n_regions = label(emphasis_mask)

        regions = []
        for i in range(1, n_regions + 1):
            idxs = np.where(labeled == i)[0]
            if len(idxs) == 0:
                continue

            t_start = times[idxs[0]]

            last_idx = min(idxs[-1] + 1, len(times) - 1)
            t_end = times[last_idx]

            if t_end - t_start >= min_duration:
                regions.append({
                    "start": round(float(t_start), 3),
                    "end": round(float(t_end), 3),
                    "score": round(float(np.mean(emphasis_score[idxs])), 6)
                })

        print(f"共发现 {len(regions)} 处高能强调片段。")
        return regions

    def aggregate_opensmile_to_windows(self, df_features, duration, window_size=2.0):
        """
        将 OpenSMILE LLD 特征聚合到统一时间窗。
        输出每个窗口的 audio_emotion 分数。
        """
        if df_features is None or df_features.empty:
            return {}

        target_cols = [
            "loudness_sma3",
            "F0semitoneFrom27.5Hz_sma3",
            "alphaRatio_sma3"
        ]

        available_cols = [col for col in target_cols if col in df_features.columns]

        if not available_cols:
            print("⚠️ OpenSMILE 中未找到目标情感特征列。")
            return {}

        emotion_data = df_features[available_cols].astype(float)

        normalized_data = emotion_data.copy()
        for col in available_cols:
            normalized_data[col] = self._minmax_norm(emotion_data[col].values)

        weights = np.ones(len(available_cols), dtype=np.float32)

        if "loudness_sma3" in available_cols:
            weights[available_cols.index("loudness_sma3")] = 0.5

        weights = weights / weights.sum()

        frame_scores = (normalized_data.values * weights).sum(axis=1)

        # 如果 OpenSMILE worker 没有输出真实时间戳，这里只能根据总时长均匀映射。
        frame_times = np.linspace(0, duration, num=len(frame_scores), endpoint=False)

        window_scores = {}

        n_windows = int(np.ceil(duration / window_size))

        for w in range(n_windows):
            start = w * window_size
            end = min((w + 1) * window_size, duration)

            mask = (frame_times >= start) & (frame_times < end)

            if np.any(mask):
                score = float(np.mean(frame_scores[mask]))
            else:
                score = 0.0

            key = (round(start, 3), round(end, 3))
            window_scores[key] = float(np.clip(score, 0.0, 1.0))

        return window_scores

    def aggregate_librosa_to_windows(self, librosa_data, duration, window_size=2.0):
        """
        将 Librosa RMS / emphasis score 聚合到统一时间窗。
        输出每个窗口的 audio_feature 分数。
        """
        times = librosa_data["times"]
        rms_norm = librosa_data["rms_norm"]
        emphasis_score = librosa_data["emphasis_score"]

        window_scores = {}

        n_windows = int(np.ceil(duration / window_size))

        for w in range(n_windows):
            start = w * window_size
            end = min((w + 1) * window_size, duration)

            mask = (times >= start) & (times < end)

            if np.any(mask):
                rms_score = float(np.mean(rms_norm[mask]))
                emphasis = float(np.mean(emphasis_score[mask]))
            else:
                rms_score = 0.0
                emphasis = 0.0

            # audio_feature 使用 RMS 与 emphasis 的综合分数。
            # 如果你只想保存能量强度，可直接设为 rms_score。
            audio_feature = 0.6 * rms_score + 0.4 * emphasis

            key = (round(start, 3), round(end, 3))
            window_scores[key] = float(np.clip(audio_feature, 0.0, 1.0))

        return window_scores

    def merge_audio_scores(self, emotion_scores, feature_scores, duration, window_size=2.0):
        """
        合并 OpenSMILE 情感分数和 Librosa 音频特征分数。
        最终保存为按时间段对齐的 segments。
        """
        segments = []

        n_windows = int(np.ceil(duration / window_size))

        for w in range(n_windows):
            start = round(w * window_size, 3)
            end = round(min((w + 1) * window_size, duration), 3)
            key = (start, end)

            segments.append({
                "start": start,
                "end": end,
                "audio_emotion": float(emotion_scores.get(key, 0.0)),
                "audio_feature": float(feature_scores.get(key, 0.0))
            })

        return segments

    def forward(self, audio_file_name=None, output_file_name=None, window_size=2.0):
        """
        处理单个音频文件，生成：
        1. 转录文本 JSON
        2. 统一时间窗音频特征 JSON
        """
        if audio_file_name is None:
            raise ValueError("请提供 audio_file_name")

        video_name = os.path.splitext(audio_file_name)[0]

        input_audio_path = os.path.join(self.input_path, audio_file_name)
        if not os.path.exists(input_audio_path):
            raise FileNotFoundError(f"音频文件不存在: {input_audio_path}")

        os.makedirs(self.output_path, exist_ok=True)

        transcript_json_path = os.path.join(
            self.output_path,
            f"{video_name}_transcript.json"
        )

        audio_features_json_path = os.path.join(
            self.output_path,
            f"{video_name}_audio_features.json"
        )

        transcripts = self.whisper_transcribe(
            input_audio_path=input_audio_path,
            output_file_path=transcript_json_path
        )

        librosa_data = self.extract_librosa_features(input_audio_path)
        duration = float(librosa.get_duration(path=input_audio_path))

        emphasis_regions = self.detect_emphasis_regions(librosa_data)

        df_features = self.opensmile_emotion_recognition(input_audio_path)

        emotion_scores = self.aggregate_opensmile_to_windows(
            df_features=df_features,
            duration=duration,
            window_size=window_size
        )

        feature_scores = self.aggregate_librosa_to_windows(
            librosa_data=librosa_data,
            duration=duration,
            window_size=window_size
        )

        audio_segments = self.merge_audio_scores(
            emotion_scores=emotion_scores,
            feature_scores=feature_scores,
            duration=duration,
            window_size=window_size
        )

        audio_features_data = {
            "meta": {
                "audio_file": audio_file_name,
                "source_audio_path": input_audio_path,
                "video_name": video_name,
                "duration": round(duration, 3),
                "window_size": window_size
            },
            "segments": audio_segments,
            "debug_info": {
                "emphasis_regions": emphasis_regions,
                "opensmile_available": df_features is not None and not df_features.empty
            }
        }

        with open(audio_features_json_path, "w", encoding="utf-8") as f:
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