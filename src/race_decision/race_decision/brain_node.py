# -*- coding: utf-8 -*-
"""
brain_node.py —— 决策 + 运动控制（v2，2026-10-24）

只做 ROS 胶水：话题进 → BrainCore → /cmd_vel 出。全部逻辑在 brain_core.py 及其三个模块里：
    lane_estimator.py  卡尔曼滤波：视觉 + 陀螺仪 + 车速 → 车道相对位姿 [e, θ, κ]
                       （延迟补偿、野值门限、丢线/路口时推算）
    controller.py      纯跟踪外环（自带曲率前馈）+ 偏航角速度内环 + 曲率限速
    race_fsm.py        WAIT / CRUISE / LOST_HOLD / LOST_STOP / JUNCTION / TIMEOUT

订阅：
    /track        custom_msgs/TrackInfo     视觉（header.stamp = 图像时间，用来补偿延迟）
    /imu          sensor_msgs/Imu           base_node 发，angular_velocity.z = 偏航角速度（左正）
    /wheel_speed  geometry_msgs/TwistStamped base_node 发，linear.x = 实测车速
    /detections   custom_msgs/Detections    （预留，YOLO 接入后用）
    /junction     std_msgs/Empty            （预留）谁认出路口谁发一下 → 进入路口直行
发布：
    /cmd_vel      geometry_msgs/Twist       linear.x = m/s，angular.z = 舵机角 rad（左正，约定同 v1）
    /brain/state  std_msgs/String           当前状态（调试）

没有 /imu 或 /wheel_speed 也能跑：自动退化（内环关、用指令速度推算），日志会提示。

参数全部可以 ros2 param set 现场改，立即生效。常用：
    max_speed      直道最高速 m/s           a_lat     弯道限速旋钮（越小弯里越慢）
    min_speed      弯道最低速               ld0/ld_k  预瞄距离（画龙就加大，切内线就减小）
    kr / ki_r      偏航角速度内环           steer_trim_deg  直线跑偏
    speed_scale    一键倍率                 dry_run   只算不发
"""
import math

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import Imu
from std_msgs.msg import Empty, String
from custom_msgs.msg import TrackInfo

try:
    from custom_msgs.msg import Detections
except ImportError:      # 老版本 custom_msgs 没有 Detections 也不影响巡线
    Detections = None

from .brain_core import BrainCore, all_defaults

NODE_ONLY = {
    "steer_sign": 1.0,          # 方向翻转。2026-10-23 实测 +1 正确
    "speed_scale": 1.0,         # 一键倍率：同时缩放 max_speed / min_speed / 各状态限速
    "dry_run": False,
    "control_hz": 50.0,
    "vision_latency_s": 0.06,   # 图像时间戳不可用时假设的视觉延迟
    "log_period_s": 2.0,
}


def _stamp_sec(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class BrainNode(Node):
    def __init__(self):
        super().__init__("brain_node")
        defaults = all_defaults()
        defaults.update(NODE_ONLY)
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.core = BrainCore({n: self.get_parameter(n).value for n in all_defaults()})
        self._apply_scale()
        self.add_on_set_parameters_callback(self._on_params)

        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pub_state = self.create_publisher(String, "/brain/state", 10)
        self.create_subscription(TrackInfo, "/track", self.on_track, 10)
        self.create_subscription(Imu, "/imu", self.on_imu, 20)
        self.create_subscription(TwistStamped, "/wheel_speed", self.on_speed, 20)
        self.create_subscription(Empty, "/junction", lambda _m: self.core.junction_hint(), 5)
        if Detections is not None:
            self.create_subscription(Detections, "/detections",
                                     lambda m: self.core.on_detections(m.items), 5)

        self.t_prev = self.now()
        self.t_log = self.t_prev
        self.last_state = None
        self.imu_warned = False
        hz = max(float(self.get_parameter("control_hz").value), 1.0)
        self.timer = self.create_timer(1.0 / hz, self.on_timer)
        self.get_logger().info(
            f"brain_node v2 启动：{hz:.0f} Hz，最高速 {self.get_parameter('max_speed').value} m/s，"
            f"dry_run={self.get_parameter('dry_run').value}")

    # ─────────────────────────────────────────────────────────
    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _apply_scale(self):
        s = float(self.get_parameter("speed_scale").value)
        for n in ("max_speed", "min_speed", "lost_speed", "junction_speed"):
            self.core.set_param(n, float(self.get_parameter(n).value) * s)

    def _on_params(self, params):
        for prm in params:
            if prm.name in self.core.p:
                self.core.set_param(prm.name, prm.value)
        # speed_scale / 速度类参数改了要重新乘倍率；回调里 get_parameter 还是旧值，所以手动算
        vals = {p.name: p.value for p in params}
        if any(n in vals for n in ("speed_scale", "max_speed", "min_speed",
                                   "lost_speed", "junction_speed")):
            s = float(vals.get("speed_scale", self.get_parameter("speed_scale").value))
            for n in ("max_speed", "min_speed", "lost_speed", "junction_speed"):
                base = float(vals.get(n, self.get_parameter(n).value))
                self.core.set_param(n, base * s)
        return SetParametersResult(successful=True)

    # ── 话题回调 ──
    def on_track(self, msg):
        now = self.now()
        stamp = _stamp_sec(msg.header.stamp)
        lat = now - stamp
        if stamp <= 0.0 or lat < 0.0 or lat > 0.5:
            # 时间戳不可用（相机没打戳 / 时钟不一致）→ 用固定延迟
            stamp = now - float(self.get_parameter("vision_latency_s").value)
        self.core.on_track(stamp=stamp, now=now, e=msg.lateral_error,
                           th=msg.heading_error, k=msg.curvature,
                           width=msg.lane_width, quality=msg.quality,
                           is_lost=bool(msg.is_lost))
        if msg.is_lost:
            self.get_logger().warn(f"丢线(黄点 {msg.n_points}): {msg.reject_reason}",
                                   throttle_duration_sec=2.0)

    def on_imu(self, msg):
        self.core.on_imu(self.now(), msg.angular_velocity.z)

    def on_speed(self, msg):
        self.core.on_speed(self.now(), msg.twist.linear.x)

    # ── 控制周期 ──
    def on_timer(self):
        now = self.now()
        dt = min(max(now - self.t_prev, 1e-3), 0.1)
        self.t_prev = now
        out = self.core.tick(now, dt)

        hold = float(self.get_parameter("max_speed").value) * \
            float(self.get_parameter("speed_scale").value) <= 1e-6   # max_speed=0：只算方向
        v = 0.0 if hold else out["v"]
        omega = out["steer"] * float(self.get_parameter("steer_sign").value)
        self.publish(v, omega)

        if self.core.imu_disabled and not self.imu_warned:
            self.imu_warned = True
            self.get_logger().error(
                f"IMU 自检不通过（r_imu/r_模型 = {self.core.imu_ratio:+.2f}，应接近 +1）："
                f"已停用 IMU 内环。多半是 base_node 的 gyro_sign 反了，跑 tools/check_imu.py 标定")

        if out["state"] != self.last_state:
            self.get_logger().info(f"状态 {self.last_state} → {out['state']}")
            self.last_state = out["state"]
            self.pub_state.publish(String(data=out["state"]))

        if now - self.t_log > float(self.get_parameter("log_period_s").value):
            self.t_log = now
            info = out.get("info", {})
            ratio = self.core.imu_ratio
            ratio_s = "-" if ratio is None else f"{ratio:.2f}"
            fb = math.degrees(info.get("delta_fb", 0.0))
            self.get_logger().info(
                f"[{out['state']}] e={out['e'] * 100:+5.1f}cm "
                f"θ={math.degrees(out['th']):+5.1f}° κ={out['k']:+.2f} "
                f"κcmd={info.get('kappa_cmd', 0.0):+.2f} "
                f"δ={math.degrees(out['steer']):+5.1f}°(fb {fb:+4.1f}) "
                f"v={v:.2f}/{out['v_meas']:.2f} r={out['r']:+.2f} "
                f"imu={'ok' if out['imu_ok'] else '无'} "
                f"| 帧 {self.core.n_frames} 野值 {self.core.n_rejected} | IMU比 {ratio_s}")
            self.core.n_frames = self.core.n_rejected = 0

    def publish(self, v, omega):
        if bool(self.get_parameter("dry_run").value):
            return
        t = Twist()
        t.linear.x = float(v)
        t.angular.z = float(omega)
        self.pub.publish(t)


def main():
    rclpy.init()
    node = BrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish(0.0, 0.0)          # 退出前务必发一次 0 速
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
