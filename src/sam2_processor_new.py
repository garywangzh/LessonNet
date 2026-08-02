import os
import numpy as np
import time
import torch
from PIL import Image
from model.sam2.sam2.build_sam import build_sam2_video_predictor
import cv2
from pathlib import Path
import json


class sam2_processor():
    def __init__(self, sam2_checkpoint, sam2_model_cfg, input_dir, output_dir, device, fps=30):
        # 假设 build_sam2_video_predictor 已经在外部导入
        self.sam2_predictor = build_sam2_video_predictor(sam2_model_cfg, sam2_checkpoint, device=device)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.device = device
        self.fps = fps

    def sam2_process(self, video_name, sam2_bboxes=None):
        """
        核心处理函数：一次性添加所有目标，然后统一传播
        """
        self.video_dir = os.path.join(self.input_dir, video_name)
        self.video_save_dir = os.path.join(self.output_dir, video_name)
        os.makedirs(self.video_save_dir, exist_ok=True)

        self.frame_names = [
            p for p in os.listdir(self.video_dir)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
        ]
        self.frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        if not self.frame_names:
            raise ValueError(f"视频目录 {self.video_dir} 中没有图片帧")

        inference_state = self.sam2_predictor.init_state(video_path=self.video_dir)
        
        if sam2_bboxes is None:
            sam2_bboxes = {}

        # 定义目标和对应的 obj_id
        targets = [
            ("blackboard", sam2_bboxes.get("blackboard"), 1),
            ("face", sam2_bboxes.get("face"), 2),
            ("body", sam2_bboxes.get("body"), 3),
        ]
        
        # 第一步：添加所有目标的框
        for target_name, bbox, obj_id in targets:
            if not bbox or len(bbox) != 4:
                print(f"⚠️ 警告：未提供 {target_name} BBox，跳过")
                continue
            
            print(f"📌 添加 {target_name} (obj_id={obj_id}): {bbox}")
            self.sam2_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=obj_id,
                box=np.array(bbox, dtype=np.float32)
            )
        
        # 第二步：一次性传播所有目标
        print("🔄 开始传播分割...")
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        
        # 第三步：分别保存每个目标
        result_dirs = {}
        for target_name, bbox, obj_id in targets:
            if not bbox or len(bbox) != 4:
                result_dirs[target_name] = None
                continue
            
            save_dir = os.path.join(self.video_save_dir, target_name)
            self.sam2_save(video_segments, save_dir, ann_obj_id=obj_id)
            result_dirs[target_name] = save_dir
            print(f"✅ {target_name} 保存完成")
        
        return result_dirs.get("face"), result_dirs.get("body"), result_dirs.get("blackboard")

    def sam2_segment(self, inference_state, box):
        """
        通用的 SAM2 分割调用核心
        :param box: 单个目标的边界框 [x_min, y_min, x_max, y_max]
        """
        ann_frame_idx = 0
        ann_obj_id = 1

        # 🔥 核心修改：使用 box 参数替代 points 和 labels
        _, out_obj_ids, out_mask_logits = self.sam2_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            box=np.array(box, dtype=np.float32)
        )

        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        return video_segments
    
    def sam2_segment_and_save(self, inference_state, prompt, target_name, output_base_dir):
        """为单个目标执行分割并保存"""
        ann_frame_idx = 0
        ann_obj_id = 1
        
        # 根据 prompt 类型添加标注
        if prompt.get('box'):
            # 使用框标注
            x_min, y_min, x_max, y_max = prompt['box']
            self.sam2_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                box=np.array([x_min, y_min, x_max, y_max]),
                points=np.array([]),
                labels=np.array([])
            )
        elif prompt.get('points') and len(prompt['points']) > 0:
            # 使用点标注
            self.sam2_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=np.array(prompt['points']),
                labels=np.array(prompt['labels'])
            )
        else:
            return None
        
        # 传播分割
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        
        # 保存结果
        save_dir = os.path.join(output_base_dir, target_name)
        self.sam2_save(video_segments, save_dir, ann_obj_id=1)
        return save_dir

    def sam2_blackboard(self, inference_state, bbox):
        if not bbox or len(bbox) != 4:
            print("⚠️ 警告：前端未提供黑板 BBox，跳过黑板分割")
            return None
            
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            video_segments = self.sam2_segment(inference_state, box=bbox)
            
        blackboard_save_dir = os.path.join(self.video_save_dir, 'blackboard')
        self.sam2_save(video_segments, blackboard_save_dir, ann_obj_id=1)
        return blackboard_save_dir

    def sam2_face(self, inference_state, bbox):
        if not bbox or len(bbox) != 4:
            print("⚠️ 警告：前端未提供人脸 BBox，跳过人脸分割")
            return None
            
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            video_segments = self.sam2_segment(inference_state, box=bbox)
            
        face_save_dir = os.path.join(self.video_save_dir, 'face')
        self.sam2_save(video_segments, face_save_dir, ann_obj_id=1)
        return face_save_dir

    def sam2_body(self, inference_state, bbox):
        if not bbox or len(bbox) != 4:
            print("⚠️ 警告：前端未提供身体 BBox，跳过身体分割")
            return None
            
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            video_segments = self.sam2_segment(inference_state, box=bbox)
            
        body_save_dir = os.path.join(self.video_save_dir, 'body')
        self.sam2_save(video_segments, body_save_dir, ann_obj_id=1)
        return body_save_dir

    def sam2_save(self, video_segments, output_dir, ann_obj_id=1):
        # 此部分与你的原始代码完全一致，负责裁剪和保存
        os.makedirs(output_dir, exist_ok=True)

        ref_img_path = os.path.join(self.video_dir, self.frame_names[0])
        ref_img = Image.open(ref_img_path)
        ref_w, ref_h = ref_img.size

        saved_count = 0
        skip_count = 0

        sorted_frames = sorted(video_segments.keys())
        for frame_idx in sorted_frames:
            frame_data = video_segments[frame_idx]
            if ann_obj_id not in frame_data:
                skip_count += 1
                continue

            mask = frame_data[ann_obj_id]
            if isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = mask
                
            if len(mask_np.shape) == 3:
                if mask_np.shape[0] == 1:
                    mask_np = mask_np[0]
                elif mask_np.shape[-1] == 1:
                    mask_np = mask_np[:, :, 0]
                else:
                    continue
            if len(mask_np.shape) != 2:
                continue

            mask_binary = mask_np > 0.5
            rows = np.any(mask_binary, axis=1)
            cols = np.any(mask_binary, axis=0)
            if not np.any(rows) or not np.any(cols):
                skip_count += 1
                continue

            y_indices = np.where(rows)[0]
            x_indices = np.where(cols)[0]
            y_min, y_max = y_indices[0], y_indices[-1]
            x_min, x_max = x_indices[0], x_indices[-1]

            if (y_max - y_min) < 2 or (x_max - x_min) < 2:
                skip_count += 1
                continue

            padding = 10
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(ref_w, x_max + padding)
            y_max = min(ref_h, y_max + padding)

            img_path = os.path.join(self.video_dir, self.frame_names[frame_idx])
            try:
                full_image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"无法打开 {img_path}: {e}")
                continue

            crop_box = (x_min, y_min, x_max + 1, y_max + 1)
            cropped_img = full_image.crop(crop_box)
            filename = f"{frame_idx:05d}.png"
            save_path = os.path.join(output_dir, filename)
            cropped_img.save(save_path)
            saved_count += 1

        print(f"{output_dir} 保存完成：成功 {saved_count} 张，跳过 {skip_count} 帧")
        return
    
    def get_interactive_prompt(self, target_name="目标", video_dir=None):
        """
        获取交互式标注点
        
        参数:
            target_name: 目标名称（用于提示）
            video_dir: 视频帧目录（如果不提供，使用 self.video_dir）
        """
        # 如果没有传入 video_dir，尝试使用 self.video_dir
        if video_dir is None:
            if not hasattr(self, 'video_dir') or self.video_dir is None:
                raise ValueError(f"请提供 video_dir 参数，或先调用 sam2_process 设置 self.video_dir")
            video_dir = self.video_dir
        
        # 读取第一帧
        first_frame_path = os.path.join(video_dir, self.frame_names[0])
        image = cv2.imread(first_frame_path)
        if image is None:
            raise ValueError(f"无法读取图像，请检查路径: {first_frame_path}")

        prompt = {'points': [], 'labels': [], 'box': None}
        window_name = f"Annotate: {target_name}"

        print(f"\n--- 正在为 [{target_name}] 提取第一帧 ---")
        print("请选择你的标注方式:")
        print("[1] 框选标注 (推荐，准确率高)")
        print("[2] 打点标注 (左键正样本，右键负样本)")
        choice = input("请输入 1 或 2 (默认1): ").strip()

        if choice == '2':
            # === 模式 2：打点标注 ===
            clone = image.copy()
            print("\n操作说明：")
            print(" - 鼠标左键：添加正样本点 (绿色，目标区域)")
            print(" - 鼠标右键：添加负样本点 (红色，背景区域)")
            print(" - 按 'c' 键：清空重新打点")
            print(" - 按 'q' 或 Enter 键：完成标注并继续")

            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    prompt['points'].append([x, y])
                    prompt['labels'].append(1)
                    cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)
                    cv2.imshow(window_name, clone)
                elif event == cv2.EVENT_RBUTTONDOWN:
                    prompt['points'].append([x, y])
                    prompt['labels'].append(0)
                    cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
                    cv2.imshow(window_name, clone)

            cv2.namedWindow(window_name)
            cv2.setMouseCallback(window_name, mouse_callback)
            cv2.imshow(window_name, clone)

            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 13:  # 13 是 Enter 键
                    break
                elif key == ord('c'):
                    prompt['points'].clear()
                    prompt['labels'].clear()
                    clone = image.copy()
                    cv2.imshow(window_name, clone)
            cv2.destroyWindow(window_name)

        else:
            # === 模式 1：框选标注 ===
            print("\n操作说明：")
            print(" - 按住鼠标左键拖拽画框")
            print(" - 按 SPACE 或 Enter 键：确认框选")
            print(" - 按 'c' 键：重新画框")
            
            bbox = cv2.selectROI(window_name, image, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow(window_name)
            
            if bbox[2] > 0 and bbox[3] > 0:
                x, y, w, h = bbox
                prompt['box'] = [x, y, x + w, y + h]
                print(f"已获取边界框: {prompt['box']}")
            else:
                print("未选择有效的边界框。")

        return prompt