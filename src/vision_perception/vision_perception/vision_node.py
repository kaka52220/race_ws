# -*- coding: utf-8 -*-
"""
vision_node.py —— ROS 2 薄壳：把 LaneTracker 包成一个节点

它只做四件事，算法一行都不在这里：
    1. 订阅 /camera/image_raw   （sensor_msgs/Image）
    2. 转成 numpy（cv_bridge）
    3. 调 LaneTracker.process()
    4. 发 /track（custom_msgs/TrackInfo）

另外可选发 /debug/track_image（调试图，10 Hz 就够，给 rqt_image_view 看）。

为什么要分成"节点"和"算法"两个文件：
    · 算法可以脱离 ROS 用图片离线调（offline_test.py）
    · 节点出问题（话题名错、时间戳错）时，不用怀疑算法
    · 两个人可以并行：一个人调算法，一个人整合
"""
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header, Float32, Float32MultiArray
from cv_bridge import CvBridge
import numpy as np

from custom_msgs.msg import TrackInfo
from .lane_tracker import LaneTracker
from .right_line import RightLine   # andy：避障区右黄线单独拟合
from .lane_params import DEFAULTS
from .crosswalk import crosswalk_score, red_light_box


def _latch_value(owner, key, v, tol, need=3):
    """值滞回：新值必须连续出现 need 帧才接受，否则保持当前值。

    为什么需要：A 点实测 e/theta/kappa 在「两套解」之间 50/50 交替，
    中值滤波（5 帧）压不住 —— 中值仍然是两者之一。
    滞回能锁住一套解，只有持续 6 帧（0.2s）的新值才切换。
    """
    st = getattr(owner, "_latch_st", None)
    if st is None:
        st = {}
        owner._latch_st = st
    cur, cnt = st.get(key, (v, 0))
    if abs(v - cur) <= tol:
        st[key] = (0.7 * cur + 0.3 * v, 0)     # 同侧 → 轻微跟随
    else:
        cnt += 1
        if cnt >= need:
            st[key] = (v, 0)                    # 持续够久 → 接受新值
        else:
            st[key] = (cur, cnt)                # 否则保持
    return st[key][0]


class VisionNode(Node):
    def __init__(self):
        super().__init__("vision_node")

        # ── 算法参数：默认值来自 lane_params.py（唯一来源）──
        # 想在命令行临时改：ros2 run ... --ros-args -p lane.pitch_deg:=11.0
        self.declare_parameter("debug_image_hz", 3.0)   # 3 Hz 够看，10 Hz 会让帧率掉到 17
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("publish_debug", True)
        # 斑马线 / 红灯（阈值法，见 crosswalk.py）
        self.declare_parameter("publish_crosswalk", True)
        self.declare_parameter("crosswalk_trigger", 8.0)
        # 把 lane_params 里的每个参数都登记成 ROS 参数（前缀 lane.）
        for k, v in DEFAULTS.items():
            self.declare_parameter(f"lane.{k}", v)

        overrides = {}
        for k in DEFAULTS:
            val = self.get_parameter(f"lane.{k}").value
            # 按默认值的类型还原，别一律转成 float ——
            # 比如 ransac_seed 必须是 int，传 float 给 numpy 的 default_rng 会报错
            if isinstance(DEFAULTS[k], bool):
                overrides[k] = bool(val)
            elif isinstance(DEFAULTS[k], int):
                overrides[k] = int(val)
            else:
                overrides[k] = float(val)
        self.tracker = LaneTracker(overrides)

        self.bridge = CvBridge()
        self.pub_track = self.create_publisher(TrackInfo, "/track", 10)
        self.rline = RightLine()
        self.pub_rline = self.create_publisher(Float32MultiArray, "/lane_right", 10)
        self.pub_cw = (self.create_publisher(Float32, "/crosswalk", 10)
                       if self.get_parameter("publish_crosswalk").value else None)
        self.pub_dbg = (self.create_publisher(Image, "/debug/track_image", 2)
                        if self.get_parameter("publish_debug").value else None)
        self.sub = self.create_subscription(
            Image, self.get_parameter("image_topic").value, self.on_image, 2)

        self.last_dbg = 0.0
        self.dbg_period = 1.0 / max(float(self.get_parameter("debug_image_hz").value), 0.1)
        self.frame_cnt = 0
        self.t_start = time.time()
        self.get_logger().info(
            f"vision_node 启动：几何 f={self.tracker.f:.0f}px "
            f"地平线第 {self.tracker.horizon_row:.0f} 行 "
            f"BEV {self.tracker.Wb}x{self.tracker.Hb}")

    def on_image(self, msg):
        t0 = time.time()
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge 失败: {e}", throttle_duration_sec=2.0)
            return
        t_conv = time.time()

        # 尺寸和参数不一致时，按参数缩放（免得 IPM 全错）
        h, w = frame.shape[:2]
        if (w, h) != (DEFAULTS["img_w"], DEFAULTS["img_h"]):
            import cv2
            frame = cv2.resize(frame, (DEFAULTS["img_w"], DEFAULTS["img_h"]))

        r = self.tracker.process(frame)
        t_proc = time.time()

        m = TrackInfo()
        m.header = Header()
        m.header.stamp = msg.header.stamp          # 用图像的时间戳，方便和控制对齐
        m.header.frame_id = "base_link"
        # ★ 2026-10-29 实测：tracker 会在两套拟合之间逐帧切换
        #   （e 在 -1.2/-4.9cm、θ 在 +1.5/-17.5°、κ 在 0.05/1.38 之间跳），
        #   舵角跟着从 5° 跳到 16°，就是"抽搐"。这里对输出做 5 帧中值滤波。
        if not hasattr(self, "_hist"):
            self._hist = []
        self._hist.append((float(r["e_m"]), float(r["theta_rad"]),
                           float(r["curvature"]), float(getattr(r, "get", lambda k, d=0: d)("quality", 0.0))))
        if len(self._hist) > 5:
            self._hist.pop(0)
        _e = sorted(h[0] for h in self._hist)[len(self._hist) // 2]
        _t = sorted(h[1] for h in self._hist)[len(self._hist) // 2]
        _k = sorted(h[2] for h in self._hist)[len(self._hist) // 2]

        m.lateral_error = float(_latch_value(self, "e", _e, 0.03))
        m.heading_error = float(_latch_value(self, "t", _t, 0.09))
        m.curvature = float(_latch_value(self, "k", _k, 0.25))
        # 中心线点列表（供控制端做纯追踪）。C 是 (d, x) 的 Nx2 数组
        _p = r.get("path") if hasattr(r, "get") else None
        if _p is not None and len(_p):
            m.path_d = [float(t[0]) for t in _p]
            m.path_x = [float(t[1]) for t in _p]
        m.lane_width = float(r["lane_width_m"])
        m.quality = float(r["quality"])
        m.is_lost = bool(r["is_lost"])
        m.mode = int(r["mode"])
        # 诊断字段：调参时靠这两个判断卡在哪一环
        m.n_points = int(r.get("n_pts", 0))
        m.reject_reason = str(r.get("reject_reason", ""))[:120]
        self.pub_track.publish(m)
        try:
            rl = self.rline.update(r["mask"], self.tracker.d_of_row, self.tracker.x_of_col)
            self.pub_rline.publish(Float32MultiArray(data=[float(rl[0]), float(rl[1]), float(rl[2])] if rl else [0.0, 0.0, 0.0]))
        except Exception as ex:
            self.get_logger().warn(f"右线拟合失败: {ex}", throttle_duration_sec=5.0)
        # ── 斑马线：地面极亮面积百分比，越大 = 离斑马线越近 ──
        #   阈值法，0 误检（实测 350 张负样本），不需要训练。
        #   >= crosswalk_trigger（默认 8%）= 约 0.7 米内，大脑据此停车。
        if self.pub_cw is not None:
            try:
                cw = crosswalk_score(frame)
                self.pub_cw.publish(Float32(data=float(cw)))
                self.cw_score = cw
            except Exception as e:
                self.get_logger().warn(f"斑马线检测失败: {e}",
                                       throttle_duration_sec=5.0)

        # 调试图（低频）
        now = time.time()
        if self.pub_dbg is not None and now - self.last_dbg > self.dbg_period:
            self.last_dbg = now
            try:
                dbg = self.tracker.visualize(frame, r)
                self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(dbg, "bgr8"))
            except Exception as e:
                self.get_logger().warn(f"调试图失败: {e}", throttle_duration_sec=5.0)

        # 每 5 秒报一次运行状况
        self.frame_cnt += 1
        el = now - self.t_start
        if el >= 5.0:
            fps = self.frame_cnt / el
            self.get_logger().info(
                f"{fps:4.1f} Hz | 总 {(time.time()-t0)*1000:5.1f} ms "
                f"[取图 {(t_conv-t0)*1000:5.1f} 算法 {(t_proc-t_conv)*1000:5.1f} "
                f"其余 {(time.time()-t_proc)*1000:5.1f}] ms | "
                f"e={m.lateral_error*100:+6.1f}cm "
                f"{'丢线' if m.is_lost else 'ok'} q={m.quality:.2f}")
            self.frame_cnt = 0
            self.t_start = now


def main():
    rclpy.init()
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
