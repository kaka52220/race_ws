# -*- coding: utf-8 -*-
"""
detect_node.py —— RK3588 NPU 上跑 YOLOv8n 挡板检测

架构位置：
    usb_cam ──┬──→ vision_node  ──→ /track        (循迹, 30Hz)
              │
              └──→ detect_node  ──→ /detections   (本节点, 15Hz)
                                        ↓
                                   brain_node ──→ /cmd_vel → base_node

为什么单独一个节点、不塞进 vision_node：
    NPU 推理 11.7ms + 循迹 13ms 串在一起 = 25ms，30Hz 的周期只剩 33ms，很勉强。
    分成两个进程，RK3588 多核并行，互不拖累。

实测（320x320 INT8）：
    推理 11.7 ms
    挡板召回 130/143 = 91%
    负样本单帧误检 19/207 = 9%
    → 所以必须做时间滤波：最近 5 帧里至少 3 帧检出才算真。
      误报概率 0.09³ ≈ 0.07%，实测够用。

模型：
    3 类 {0: 红绿灯, 1: 挡板, 2: 斑马线}，但只训练了挡板，
    另两类的输出通道没训练过，这里只取挡板（模型 idx=1 → 消息 CLASS_BARRIER=2）。
"""
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from custom_msgs.msg import Detection, Detections

try:
    from rknnlite.api import RKNNLite
except ImportError:                     # 没有 NPU 环境时给出清楚的报错
    RKNNLite = None


# ── 模型常量（和转换时一致，不能改）──
MODEL_PATH = "/home/orangepi/race_ws/src/vision_perception/models/yolov8n_barrier.rknn"
IMGSZ = 320
NC = 3
IDX_BARRIER = 1                          # 模型里挡板的索引
CLASS_BARRIER = Detection.CLASS_BARRIER  # 消息里挡板的编号（=2）

# ── 相机几何（和 lane_params 一致，用来把框底边反投成距离）──
CAM_H = 0.16            # 镜头离地高 [m]
CAM_F = 443.2           # 焦距 [px]（1280 宽 / 99.5° 水平视场）
HORIZON_ROW = 262       # 地平线所在行（720 高）

# 归一化框 -> 原图坐标时用，运行中会被真实图像尺寸覆盖
DEFAULT_W, DEFAULT_H = 1280, 720


class DetectNode(Node):
    def __init__(self):
        super().__init__("detect_node")

        self.declare_parameter("model_path", MODEL_PATH)
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("detect_hz", 15.0)        # 跑多快，不用跟 30Hz 相机
        self.declare_parameter("conf_thr", 0.25)         # INT8 后分数二值化（0 或 1.65），这个阈值实际是"有没有"
        self.declare_parameter("iou_thr", 0.45)
        self.declare_parameter("hit_frames", 3)          # 最近 window 帧里至少几帧检出才算真
        self.declare_parameter("window_frames", 5)
        self.declare_parameter("min_box_h", 0.04)        # 框高占画面比例下限，滤掉远处小噪点
        self.declare_parameter("publish_empty", True)   # 没检出也发空消息
        self.declare_parameter("publish_debug", False)   # 发布画了框的调试图
        self.declare_parameter("debug_hz", 5.0)
        # ★ 挡板真实高度（米）。距离 = f × H / 框像素高。
        #   实测：底边反投在 0.8m 外就废了（1.5m 报成 4.4m），因为靠近地平线后
        #   1 像素 = 9cm；同样 1.5m，框高度模型 1 像素只差 1.3cm，且不受地面反光影响。
        self.declare_parameter("barrier_h_m", 0.40)
        # 线性修正：真实距离 = dist_a × 上式 + dist_b（默认不修正，用卷尺标定后再填）
        self.declare_parameter("dist_a", 1.0)
        self.declare_parameter("dist_b", 0.0)

        if RKNNLite is None:
            self.get_logger().error("没有 rknnlite，装 rknn-toolkit-lite2")
            raise RuntimeError("rknnlite 缺失")

        self.rk = RKNNLite()
        mp = self.get_parameter("model_path").value
        if self.rk.load_rknn(mp) != 0:
            raise RuntimeError(f"load_rknn 失败: {mp}")
        if self.rk.init_runtime() != 0:
            raise RuntimeError("init_runtime 失败")
        self.get_logger().info(f"NPU 模型已加载: {mp}")

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Detections, "/detections", 10)
        self.pub_dbg = (self.create_publisher(Image, "/debug/detect_image", 2)
                        if self.get_parameter("publish_debug").value else None)
        self.last_dbg = 0.0
        self.sub = self.create_subscription(
            Image, self.get_parameter("image_topic").value, self.on_image, 2)

        self.hits = []                    # 最近若干帧的布尔（检出挡板？）
        self.last_run = 0.0
        self.period = 1.0 / max(float(self.get_parameter("detect_hz").value), 1.0)
        self.frame_cnt = 0
        self.t_start = time.time()
        self.n_run = 0

    # ─────────────────────────────────────────────────────────
    def _letterbox(self, img):
        h, w = img.shape[:2]
        r = min(IMGSZ / w, IMGSZ / h)
        nw, nh = int(round(w * r)), int(round(h * r))
        canvas = np.full((IMGSZ, IMGSZ, 3), 114, np.uint8)
        dx, dy = (IMGSZ - nw) // 2, (IMGSZ - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(img, (nw, nh),
                                                    interpolation=cv2.INTER_LINEAR)
        return canvas, r, dx, dy

    def _decode(self, out):
        """(1, 4+nc, N) -> [(cx,cy,w,h,score)]，坐标在 320 输入图上"""
        pred = np.asarray(out[0])
        if pred.ndim == 3:
            pred = pred[0]
        if pred.shape[0] != 4 + NC:
            pred = pred.T
        boxes = pred[:4].T                       # cx,cy,w,h
        scores = pred[4 + IDX_BARRIER]
        keep = scores >= float(self.get_parameter("conf_thr").value)
        if not keep.any():
            return []
        boxes, scores = boxes[keep], scores[keep]
        xywh = np.stack([boxes[:, 0] - boxes[:, 2] / 2,
                         boxes[:, 1] - boxes[:, 3] / 2,
                         boxes[:, 2], boxes[:, 3]], 1)
        idx = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(),
                               float(self.get_parameter("conf_thr").value),
                               float(self.get_parameter("iou_thr").value))
        if len(idx) == 0:
            return []
        res = []
        for i in np.array(idx).flatten():
            res.append((float(boxes[i, 0]), float(boxes[i, 1]),
                        float(boxes[i, 2]), float(boxes[i, 3]), float(scores[i])))
        return res

    def _ground_dist(self, v_bottom, img_h):
        """框底边所在行 -> 距相机的地面纵向距离 [m]

        针孔模型：row = f*h/d + horizon  =>  d = f*h / (row - horizon)
        近处（row 大）准，远处（接近地平线）误差大。
        """
        dv = v_bottom - HORIZON_ROW
        if dv <= 8:                       # 太接近地平线，反投无意义
            return -1.0
        d = CAM_F * CAM_H / dv
        return float(d) if 0.05 < d < 8.0 else -1.0

    # ─────────────────────────────────────────────────────────
    def on_image(self, msg):
        self.frame_cnt += 1
        now = self.get_clock().now().nanoseconds / 1e9
        # 限速：不跟 30Hz 相机，按 detect_hz 跑
        if now - self.last_run < self.period:
            return
        self.last_run = now
        self.n_run += 1

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        h_img, w_img = frame.shape[:2]

        lb, r, dx, dy = self._letterbox(frame)
        inp = np.expand_dims(cv2.cvtColor(lb, cv2.COLOR_BGR2RGB), 0)
        out = self.rk.inference(inputs=[inp])
        boxes = self._decode(out)

        # 映射回原图 + 尺寸过滤 + 换算距离
        dets = []
        for cx, cy, bw, bh, sc in boxes:
            x1, y1 = (cx - bw / 2 - dx) / r, (cy - bh / 2 - dy) / r
            x2, y2 = (cx + bw / 2 - dx) / r, (cy + bh / 2 - dy) / r
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(w_img), x2), min(float(h_img), y2)
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            if (y2 - y1) / h_img < float(self.get_parameter("min_box_h").value):
                continue
            dets.append((x1, y1, x2, y2, sc))

        # 时间滤波：最近 window 帧里至少 hit 帧有检出
        self.hits.append(len(dets) > 0)
        win = int(self.get_parameter("window_frames").value)
        if len(self.hits) > win:
            self.hits = self.hits[-win:]
        confirmed = sum(self.hits) >= int(self.get_parameter("hit_frames").value)

        m = Detections()
        m.header = Header()
        m.header.stamp = msg.header.stamp
        m.header.frame_id = "base_link"
        if confirmed:
            for x1, y1, x2, y2, sc in dets:
                d = Detection()
                d.class_id = CLASS_BARRIER
                d.confidence = min(1.0, max(0.0, sc))     # INT8 会给出 >1 的值，夹一下
                d.cx = (x1 + x2) / 2.0 / w_img
                d.cy = (y1 + y2) / 2.0 / h_img
                d.w = (x2 - x1) / w_img
                d.h = (y2 - y1) / h_img
                # ── 距离：两个模型分工，正好互补 ──
                #   框完整   → 框高度 d = f·H/h_px
                #              远处准；近距离挡板顶部会超出画面上沿，框被裁，就不准了
                #   顶部被裁 → 底边反投 d = f·h/(row-地平线)
                #              近距离准（0.4~0.7m 实测误差 <2%）；远处接近地平线后
                #              1 像素=9cm，1.5m 能报成 4.4m，所以远处绝不能用它
                h_px = y2 - y1
                clipped_top = y1 <= 1.0
                if clipped_top:
                    raw_d = self._ground_dist(y2, h_img)
                elif h_px > 12:
                    raw_d = CAM_F * float(self.get_parameter("barrier_h_m").value) / h_px
                else:
                    raw_d = -1.0
                d.dist_m = (raw_d * float(self.get_parameter("dist_a").value)
                            + float(self.get_parameter("dist_b").value)) if raw_d > 0 else -1.0
                d.bearing_rad = 0.0
                d.color_state = Detection.COLOR_UNKNOWN
                m.items.append(d)
        elif not self.get_parameter("publish_empty").value:
            return

        # 调试图：把框画出来，看框有没有被图像边界截断
        if self.pub_dbg is not None:
            now2 = self.get_clock().now().nanoseconds / 1e9
            if now2 - self.last_dbg > 1.0 / max(float(self.get_parameter("debug_hz").value), 0.5):
                self.last_dbg = now2
                vis = frame.copy()
                for x1, y1, x2, y2, sc in dets:
                    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                    cv2.putText(vis, f"{y2-y1:.0f}px", (int(x1), max(20, int(y1) - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(vis, f"n={len(dets)}", (10, 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(vis, "bgr8"))

        self.pub.publish(m)

    # ROS 关闭时释放 NPU
    def destroy_node(self):
        try:
            self.rk.release()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = None
    try:
        node = DetectNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
