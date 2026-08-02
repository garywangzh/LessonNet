# client_app.py
import gradio as gr
import cv2
import requests
import json
import os
import numpy as np

# ⚠️ 将这里替换为你 Ubuntu 服务器的实际 IP 地址
SERVER_URL = "http://127.0.0.1:8000/analyze/"

# 定义需要框选的目标顺序和颜色 (BGR 格式，用于 OpenCV 绘图)
TARGETS = [
    {"key": "face", "name": "人脸", "color": (255, 0, 0)},       # 蓝色
    {"key": "body", "name": "教师全身", "color": (0, 255, 0)},     # 绿色
    {"key": "blackboard", "name": "黑板或PPT", "color": (0, 0, 255)} # 红色
]

def extract_first_frame(video_path):
    """提取视频第一帧供用户画框"""
    if video_path is None:
        return None, get_instruction_text(0)
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        # 保存一张原始帧的拷贝到状态中，用于每次重绘
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return rgb_frame, get_instruction_text(0)
    return None, "❌ 无法读取视频帧"

def get_instruction_text(step):
    """根据当前步数生成醒目的 UI 引导语"""
    if step >= 6:
        return "✅ **所有目标已框选完毕！请点击下方蓝色的【提交至服务器分析】按钮。**"
    
    target_idx = step // 2
    is_first_point = (step % 2 == 0)
    target_name = TARGETS[target_idx]["name"]
    action = "左上角 ↖️" if is_first_point else "右下角 ↘️"
    
    return f"👉 **当前任务：请在右侧图片上点击【{target_name}】的 {action}** (进度: {target_idx+1}/3)"

def handle_click(evt: gr.SelectData, state, original_image):
    """处理用户在图片上的点击事件，实现两点画框逻辑"""
    if original_image is None:
        return state, None, "请先上传视频！"
    
    step = state["step"]
    if step >= 6:
        return state, draw_boxes(original_image, state), get_instruction_text(step)

    x, y = evt.index
    target_idx = step // 2
    target_key = TARGETS[target_idx]["key"]
    is_first_point = (step % 2 == 0)

    if is_first_point:
        # 记录左上角
        state["temp_point"] = (x, y)
    else:
        # 记录右下角，计算出真正的 [x_min, y_min, x_max, y_max]
        pt1 = state["temp_point"]
        pt2 = (x, y)
        x_min, x_max = min(pt1[0], pt2[0]), max(pt1[0], pt2[0])
        y_min, y_max = min(pt1[1], pt2[1]), max(pt1[1], pt2[1])
        state["bboxes"][target_key] = [x_min, y_min, x_max, y_max]
        state["temp_point"] = None # 清空临时点

    state["step"] += 1
    
    # 每次点击后重新绘制图像
    annotated_img = draw_boxes(original_image, state)
    instruction = get_instruction_text(state["step"])
    
    return state, annotated_img, instruction

def draw_boxes(img, state):
    """在图像上绘制已完成的框和当前点击的临时点"""
    img_copy = img.copy()
    
    # 1. 绘制已经确认的矩形框
    for i, target in enumerate(TARGETS):
        bbox = state["bboxes"].get(target["key"])
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), target["color"], 3)
            # 添加半透明背景文字
            cv2.putText(img_copy, target["name"], (x1, max(20, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, target["color"], 2)

    # 2. 绘制当前点击的第一个点（十字准星）
    if state.get("temp_point") is not None:
        px, py = state["temp_point"]
        current_color = TARGETS[state["step"] // 2]["color"]
        cv2.drawMarker(img_copy, (px, py), current_color, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        cv2.putText(img_copy, "Start", (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, current_color, 2)

    return img_copy

def reset_selection(original_image):
    """重置用户的框选状态"""
    new_state = {"step": 0, "bboxes": {}, "temp_point": None}
    return new_state, original_image, get_instruction_text(0)

def submit_to_server(video_path, state):
    """将视频和最终确定的 BBox 坐标发送给服务器"""
    if not video_path:
        return "❌ 请先上传视频！"
    
    if state["step"] < 6:
        return f"❌ 坐标未收集完整！请按照上方提示继续在图片上点击。"

    yield "⏳ 正在上传视频并请求 Ubuntu 服务器启动 LessonNet 分析引擎，请耐心等待..."

    try:
        with open(video_path, "rb") as f:
            files = {"video": (os.path.basename(video_path), f, "video/mp4")}
            # 发送结构如: {"face": [x1,y1,x2,y2], ...}
            data = {"prompts": json.dumps(state["bboxes"])}
            
            response = requests.post(SERVER_URL, files=files, data=data, timeout=3600)
            
            if response.status_code == 200:
                res_json = response.json()
                if res_json["status"] == "success":
                    yield f"🎉 分析完成！下面是生成的报告：\n\n---\n{res_json['report']}"
                else:
                    yield f"❌ 服务器处理报错：\n{res_json['message']}"
            else:
                yield f"❌ 网络请求失败，状态码：{response.status_code}"
    except Exception as e:
        yield f"❌ 连接服务器失败，请检查 IP 地址和网络连通性。\n详细错误: {e}"

# ==========================================
# 构建 Gradio 界面
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎓 LessonNet 教学视频多模态分析客户端")
    
    # 核心状态机字典
    state_var = gr.State({"step": 0, "bboxes": {}, "temp_point": None})
    original_frame_var = gr.State(None) # 纯净的第一帧图片
    
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="第一步：上传教学视频")
            instruction_box = gr.Markdown("👉 **当前任务：请先上传视频**")
            
            with gr.Row():
                reset_btn = gr.Button("🔄 重置框选", variant="secondary")
                submit_btn = gr.Button("🚀 提交至服务器分析", variant="primary")
            
            gr.Markdown("""
            ### 📝 操作指南
            1. 等待视频第一帧加载到右侧。
            2. 根据上方加粗文字的提示，在画面中点击**两次**。
               - 第一次点击框的左上角
               - 第二次点击框的右下角
            3. 三个目标全部画完后，点击蓝色按钮提交。
            """)
            
        with gr.Column(scale=2):
            frame_output = gr.Image(label="视频画面 (在此点击进行交互)", interactive=True)
            
    with gr.Row():
        report_output = gr.Markdown("此处将显示服务器生成的教学分析报告...")

    # 事件绑定
    # 视频上传完毕 -> 提取第一帧 -> 初始化纯净帧 -> 重置状态 -> 更新 UI 提示
    video_input.change(
        fn=extract_first_frame, 
        inputs=video_input, 
        outputs=[frame_output, instruction_box]
    ).then(
        fn=lambda img: (img, {"step": 0, "bboxes": {}, "temp_point": None}),
        inputs=frame_output,
        outputs=[original_frame_var, state_var]
    )
    
    # 图片点击 -> 处理逻辑画框 -> 更新图片和状态语
    frame_output.select(
        fn=handle_click, 
        inputs=[state_var, original_frame_var], 
        outputs=[state_var, frame_output, instruction_box]
    )
    
    # 重置按钮 -> 恢复为纯净帧和步数 0
    reset_btn.click(
        fn=reset_selection,
        inputs=[original_frame_var],
        outputs=[state_var, frame_output, instruction_box]
    )
    
    # 提交按钮
    submit_btn.click(
        fn=submit_to_server,
        inputs=[video_input, state_var],
        outputs=report_output
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)