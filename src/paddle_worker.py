# src/paddle_worker.py
import os
import sys
import json
from pathlib import Path
from tqdm import tqdm
import paddle
paddle.set_device('cpu')

from paddleocr import PaddleOCR

def run_ocr(blackboard_save_dir, fps, output_json_path, frame_list=None):
    print(f"[OCR Worker] 启动，目录: {blackboard_save_dir}")
    
    # 初始化 OCR
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        ocr_version="PP-OCRv5",
        lang='ch',
        enable_mkldnn=False
    )
    
    # 获取图片列表
    if frame_list is not None:
        # 只处理指定的帧（frame_list 是文件名列表）
        image_paths = [Path(blackboard_save_dir) / f for f in frame_list if (Path(blackboard_save_dir) / f).exists()]
        print(f"[OCR Worker] 使用指定的 {len(image_paths)} 个帧")
    else:
        # 处理目录下所有 PNG/JPG
        image_paths = sorted(Path(blackboard_save_dir).glob("*.png"))
        if not image_paths:
            image_paths = sorted(Path(blackboard_save_dir).glob("*.jpg"))
        print(f"[OCR Worker] 使用目录下所有 {len(image_paths)} 个图片")
    
    if not image_paths:
        print("[OCR Worker] 没有找到图片，退出")
        return
    
    results = []
    for img_path in tqdm(image_paths, desc="提取板书 OCR", unit="帧"):
        try:
            # 从文件名提取时间戳（帧序号 / fps）
            stem = img_path.stem
            try:
                frame_idx = int(stem)
                timestamp = frame_idx / float(fps)
            except ValueError:
                timestamp = 0.0
            
            ocr_res = ocr.predict(str(img_path))
            frame_text = ""
            if ocr_res and ocr_res[0] is not None:
                # 提取识别出的文本
                text_lines = [line for line in ocr_res[0]['rec_texts']]
                frame_text = " ".join(text_lines)
            
            if frame_text.strip():
                results.append({
                    "timestamp": timestamp,
                    "text": frame_text.strip(),
                    "frame_idx": str(img_path.name),
                    "text_len": len(frame_text.strip()),
                })
        except Exception as e:
            print(f"处理 {img_path} 时出错: {e}")
    
    # 归一化文本长度
    if results:
        lengths = [r['text_len'] for r in results]
        min_len, max_len = min(lengths), max(lengths)
        range_len = max_len - min_len if max_len > min_len else 1
        for r in results:
            r['norm_text_len'] = (r['text_len'] - min_len) / range_len
    
    # 保存结果
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"[OCR Worker] 完成，提取 {len(results)} 帧板书数据")

if __name__ == "__main__":
    # 参数: python paddle_worker.py <dir> <fps> <output> [frame_list_json]
    if len(sys.argv) >= 4:
        blackboard_dir = sys.argv[1]
        fps = float(sys.argv[2])
        output_path = sys.argv[3]
        frame_list = None
        if len(sys.argv) > 4:
            try:
                frame_list = json.loads(sys.argv[4])
            except:
                frame_list = None
        run_ocr(blackboard_dir, fps, output_path, frame_list)
    else:
        print("用法: python paddle_worker.py <dir> <fps> <output> [frame_list_json]")
        sys.exit(1)