"""
示例：speed_estimator.py

说明：
- 这是一个演示性脚本，展示如何使用 YOLOv8 检测网球、将像素坐标通过单应矩阵（homography）映射到场地平面（米），并基于跟踪的世界坐标计算速度。
- 你需要：
  - 训练好的 YOLOv8 权重（或用官方预训练权重做微调），路径填入 YOLO_MODEL。
  - 计算好的单应矩阵 H（像素 -> 世界平面（米））。可用 README 中的标定步骤计算。

注意：该脚本使用了最简单的最近邻跟踪，实际项目建议使用更稳健的跟踪器（ByteTrack、DeepSort、Norfair 等），并加入亚像素定位、滤波与插值。
"""

import cv2
import numpy as np
from ultralytics import YOLO
import math
import collections
import os

# ---------- 配置 ----------
VIDEO_IN = 'input.mp4'            # 输入视频
VIDEO_OUT = 'output_with_speed.mp4'  # 输出视频
YOLO_MODEL = 'runs/detect/exp/weights/best.pt'  # 训练好的权重路径（请修改）
FPS = None  # 如果为 None 则从视频读取
MIN_DETECT_CONF = 0.3
MAX_ASSIGN_DIST_M = 1.5  # 跟踪分配阈值（米）

# 单应矩阵 H（像素 -> 世界平面（米）），请在运行前设置。
# 例如通过 README 中的标定步骤计算，并在这里粘贴为 numpy 数组或保存为 .npy 并加载
H = None  # 示例：H = np.load('homography.npy')

# ---------- 简单跟踪器（基于最近邻） ----------
class Track:
    def __init__(self, id, pos_world, pos_px, t):
        self.id = id
        self.history_world = [(t, pos_world)]
        self.history_px = [(t, pos_px)]
        self.last_pos = pos_world
        self.last_px = pos_px
        self.last_t = t
        self.missed = 0

    def add(self, pos_world, pos_px, t):
        self.history_world.append((t, pos_world))
        self.history_px.append((t, pos_px))
        self.last_pos = pos_world
        self.last_px = pos_px
        self.last_t = t
        self.missed = 0

    def predict_speed(self, window=3):
        if len(self.history_world) < 2:
            return 0.0
        pts = list(self.history_world[-window:])
        d = 0.0
        dt = 0.0
        for i in range(1, len(pts)):
            (t0, p0), (t1, p1) = pts[i-1], pts[i]
            d += math.hypot(p1[0]-p0[0], p1[1]-p0[1])
            dt += (t1 - t0)
        return (d / dt) if dt>0 else 0.0

# ---------- 坐标转换 ----------
def pixel_to_world(px, H):
    # px: (x, y) 单个像素点；H: 单应矩阵，将像素映射到场地平面（单位为米）
    pts = np.array([[[px[0], px[1]]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(pts, H)  # 返回 [[[xw, yw]]]
    return float(dst[0,0,0]), float(dst[0,0,1])

# ---------- 主流程 ----------
def main():
    global H, FPS
    if not os.path.exists(VIDEO_IN):
        print(f"输入视频 {VIDEO_IN} 不存在，请修改 VIDEO_IN 路径或把视频放在仓库中。")
        return

    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        print("无法打开视频")
        return
    if FPS is None:
        FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt_frame = 1.0 / FPS

    # 检查 H
    if H is None:
        print("请先计算并设置单应矩阵 H（像素 -> 米），例如在 README 中的标定步骤。可以把 H 存为 homography.npy 然后在此处加载。")
        return

    model = YOLO(YOLO_MODEL)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(VIDEO_OUT, fourcc, FPS, (w, h))

    tracks = {}
    next_id = 1
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        t_now = frame_idx * dt_frame

        results = model.predict(frame, imgsz=1280, conf=MIN_DETECT_CONF, device=0)
        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, 'xyxy') else np.array([])
        confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, 'conf') else np.array([])
        clss = r.boxes.cls.cpu().numpy().astype(int) if hasattr(r.boxes, 'cls') else np.array([])

        detections = []
        for (x1,y1,x2,y2), conf, cls in zip(boxes, confs, clss):
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            detections.append((cx, cy, float(conf), int(cls)))

        # 过滤出球类（假设 class 0 为 ball）
        ball_dets = [d for d in detections if d[3] == 0]

        ball_world = []
        for cx, cy, conf, cls in ball_dets:
            try:
                wx, wy = pixel_to_world((cx,cy), H)
                ball_world.append(((cx,cy), (wx,wy), conf))
            except Exception as e:
                # 转换失败
                continue

        # 关联：简单最近邻（基于世界坐标）
        assigned_tracks = set()
        used_dets = set()

        for det_idx, (pix, world, conf) in enumerate(ball_world):
            best_id = None
            best_dist = float('inf')
            for tid, tr in tracks.items():
                dist = math.hypot(world[0]-tr.last_pos[0], world[1]-tr.last_pos[1])
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid
            if best_id is not None and best_dist < MAX_ASSIGN_DIST_M:
                tracks[best_id].add(world, pix, t_now)
                assigned_tracks.add(best_id)
                used_dets.add(det_idx)
            else:
                # 新 track
                tid = next_id
                next_id += 1
                tracks[tid] = Track(tid, world, pix, t_now)
                assigned_tracks.add(tid)
                used_dets.add(det_idx)

        # 增加 missed 计数，清理长时间未见的 track
        remove_ids = []
        for tid, tr in list(tracks.items()):
            if tid not in assigned_tracks:
                tr.missed += 1
            if tr.missed > int(FPS*1.0):  # 1秒内没见则删除
                remove_ids.append(tid)
        for tid in remove_ids:
            del tracks[tid]

        # 绘制当前帧：bbox/速度/轨迹
        for tid, tr in tracks.items():
            # 绘制最近像素位置
            px = tr.last_px
            cv2.circle(frame, (int(px[0]), int(px[1])), 6, (0,255,0), -1)
            # 绘制轨迹（像素历史）
            for i in range(1, len(tr.history_px)):
                _, p0 = tr.history_px[i-1]
                _, p1 = tr.history_px[i]
                cv2.line(frame, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (0,200,0), 2)
            speed_m_s = tr.predict_speed()
            speed_kmh = speed_m_s * 3.6
            text = f"ID{tid}: {speed_kmh:.1f} km/h"
            cv2.putText(frame, text, (int(px[0])+8, int(px[1])-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print("完成，输出：", VIDEO_OUT)

if __name__ == '__main__':
    main()
