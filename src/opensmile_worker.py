# src/opensmile.py
import os
import sys
import json
import opensmile
import pandas as pd
import numpy as np

def run_opensmile_extraction(input_audio_path, output_json_path):
    """
    独立进程：提取 OpenSMILE eGeMAPS 特征并保存为 JSON
    兼容 opensmile 2.5.0
    """
    print(f"[OpenSMILE Worker] 启动。处理音频: {input_audio_path}")
    
    try:
        # 2.5.0 使用 Smile 类
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors
        )
        print("[OpenSMILE Worker] 使用 OpenSMILE 2.5.0 API (Smile)")
        
        # 提取特征
        df = smile.process_file(input_audio_path)
        
        if df is None or df.empty:
            print("[OpenSMILE Worker] 特征提取结果为空")
            return False
        
        # 重置索引，将多级索引展平
        df = df.reset_index()
        
        # 转换为字典列表
        results = df.to_dict(orient='records')
        
        # 处理时间列（Timedelta -> 秒）及 numpy 类型
        for item in results:
            if 'start' in item and hasattr(item['start'], 'total_seconds'):
                item['start'] = item['start'].total_seconds()
            if 'end' in item and hasattr(item['end'], 'total_seconds'):
                item['end'] = item['end'].total_seconds()
            # 确保所有值可 JSON 序列化
            for key, value in list(item.items()):
                if isinstance(value, (np.float32, np.float64)):
                    item[key] = float(value)
                elif isinstance(value, (np.int32, np.int64)):
                    item[key] = int(value)

        # 写入 JSON
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        print(f"[OpenSMILE Worker] 提取完成，共 {len(results)} 帧。结果保存至: {output_json_path}")
        return True

    except Exception as e:
        print(f"[OpenSMILE Worker] 运行崩溃: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python opensmile.py <input_audio> <output_json>")
        sys.exit(1)
    
    success = run_opensmile_extraction(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)