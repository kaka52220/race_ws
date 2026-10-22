# -*- coding: utf-8 -*-
"""
brain_node.py —— 决策 + 运动控制

它订阅 /track（视觉给的三个数），算出方向盘打多少、油门给多少，发 /cmd_vel。

这一版**只做一件事：跟着车道线走**。
斑马线停车、挡板避障、计圈这些，等 YOLO 出来再往这个骨架上加状态。

控制律：Stanley（比 PID 稳，参数有物理意义）

    δ = θ_e  −  atan(k·e / (v + v_eps))  +  atan(L·κ)
        └─航向─┘  └────横向纠偏────┘     └─曲率前馈─┘

    δ: 舵机角（左正）
    θ_e: 车道方向相对车头的角度（左正）—— 车道往左拐就左打
    e: 车离车道中心多远（左正）—— 车偏左就往右修，所以是减号
    κ: 车道曲率 —— 提前打角，别等偏了才修（这就是前馈）

    然后把 δ 换成角速度：ω = v·tan(δ)/L
    base_node 收到 /cmd_vel 后自己反算出舵机角，所以两边是一致的。

安全：
    · 视觉超过 track_timeout 没更新 → 立即停车
    · /track 报丢线 → 减速，持续丢线就停
    · dry_run=true 时只算不发（车架起来调参用）
"""
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from custom_msgs.msg import TrackInfo


class BrainNode(Node):
    def __init__(self):
        super().__init__("brain_node")

        # ── 全部可调参数（现场用 ros2 param set 改，立即生效）──
        # 车辆几何
        self.declare_parameter("wheelbase_m", 0.143)
        self.declare_parameter("steer_max_deg", 25.0)
        self.declare_parameter("steer_trim_deg", 0.0)     # 直线跑偏时调这个
        self.declare_parameter("steer_rate_deg_per_s", 300.0)  # 舵机变化率上限

        # 控制律
        self.declare_parameter("stanley_k", 2.0)          # 横向纠偏增益，越大纠得越猛
        self.declare_parameter("v_eps", 0.15)             # 防低速时纠偏项爆炸

        # 速度
        self.declare_parameter("max_speed", 0.30)         # 全局最高速 m/s
        self.declare_parameter("min_speed", 0.06)         # 转弯时最低速，别停
        self.declare_parameter("speed_scale", 1.0)        # 一键倍率：0.5 = 全部减半
        self.declare_parameter("a_lat", 0.8)              # 允许的侧向加速度 m/s²
        self.declare_parameter("accel", 0.8)              # 加速度上限
        self.declare_parameter("decel", 1.5)              # 减速度上限

        # 安全 / 调试
        self.declare_parameter("track_timeout_s", 0.5)    # 多久没视觉就停车
        self.declare_parameter("lost_slowdown", 0.5)      # 丢线时速度乘这个系数
        self.declare_parameter("dry_run", False)          # true = 只算不发
        self.declare_parameter("control_hz", 50.0)

        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.sub = self.create_subscription(TrackInfo, "/track", self.on_track, 10)

        self.last = None            # 最近一次的 TrackInfo
        self.last_time = 0.0
        self.v_cmd = 0.0            # 当前指令速度（做加减速斜坡用）
        self.steer_prev = 0.0       # 上一帧舵机角（做变化率限制用）
        self.state = "WAIT"
        self.n_ok = 0
        self.n_lost = 0
        self.t_last_log = self.get_clock().now().nanoseconds / 1e9

        period = 1.0 / max(float(self.get_parameter("control_hz").value), 1.0)
        self.timer = self.create_timer(period, self.on_timer)
        self.get_logger().info(
            f"brain_node 启动：{self.get_parameter('control_hz').value:.0f} Hz，"
            f"最高速 {self.get_parameter('max_speed').value} m/s，"
            f"dry_run={self.get_parameter('dry_run').value}")

    # ─────────────────────────────────────────────────────────
    def on_track(self, msg):
        self.last = msg
        self.last_time = self.get_clock().now().nanoseconds / 1e9
        if msg.is_lost:
            self.n_lost += 1
        else:
            self.n_ok += 1

    # ─────────────────────────────────────────────────────────
    def on_timer(self):
        g = self.get_parameter
        now = self.get_clock().now().nanoseconds / 1e9
        v_max = float(g("max_speed").value) * float(g("speed_scale").value)

        # ── 视觉超时：这是最后一道保险，必须最先判 ──
        if self.last is None or (now - self.last_time) > float(g("track_timeout_s").value):
            self.state = "WAIT_VISION" if self.last is None else "VISION_TIMEOUT"
            self.publish(0.0, 0.0)
            self.v_cmd = 0.0
            return

        msg = self.last

        # ── 丢线：不打死，减速慢慢找线 ──
        if msg.is_lost:
            self.state = "LOST"
            v = self.v_cmd * float(g("lost_slowdown").value)
            self.publish(max(v, 0.0), 0.0)      # 丢线时方向盘回正，别乱打
            self.v_cmd = max(v, 0.0)
            return

        self.state = "CRUISING"
        kappa = float(msg.curvature)
        e = float(msg.lateral_error)
        th = float(msg.heading_error)
        L = float(g("wheelbase_m").value)

        # ── 速度：按曲率限速（v = sqrt(a_lat / |κ|)）──
        if abs(kappa) > 1e-3:
            v_curve = math.sqrt(float(g("a_lat").value) / abs(kappa))
        else:
            v_curve = v_max
        v_target = min(v_max, v_curve)
        v_target = max(v_target, float(g("min_speed").value))

        # 加减速斜坡（别一脚油门一脚刹车）
        if v_target > self.v_cmd:
            v_cmd = min(v_target, self.v_cmd + float(g("accel").value) * 0.02)
        else:
            v_cmd = max(v_target, self.v_cmd - float(g("decel").value) * 0.02)
        self.v_cmd = v_cmd

        # ── Stanley 转向 ──
        steer_ff = math.atan(L * kappa)                      # 曲率前馈
        corr = math.atan(float(g("stanley_k").value) * e
                         / (v_cmd + float(g("v_eps").value)))
        delta = th - corr + steer_ff + math.radians(float(g("steer_trim_deg").value))

        # 限幅
        d_max = math.radians(float(g("steer_max_deg").value))
        delta = max(-d_max, min(d_max, delta))
        # 变化率限制（舵机从 0 打到大角度要 80~120 ms，直接阶跃会甩头）
        rate = math.radians(float(g("steer_rate_deg_per_s").value)) * 0.02
        delta = max(self.steer_prev - rate, min(self.steer_prev + rate, delta))
        self.steer_prev = delta

        # 舵机角 → 角速度（base_node 会反算回去）
        omega = v_cmd * math.tan(delta) / L
        self.publish(v_cmd, omega)

        # 每 2 秒打一行日志
        if now - self.t_last_log > 2.0:
            self.t_last_log = now
            self.get_logger().info(
                f"[{self.state}] e={e*100:+6.1f}cm θ={math.degrees(th):+5.2f}° "
                f"κ={kappa:+.3f} → δ={math.degrees(delta):+5.1f}° "
                f"v={v_cmd:.2f}m/s | ok={self.n_ok} lost={self.n_lost}")
            self.n_ok = self.n_lost = 0

    # ─────────────────────────────────────────────────────────
    def publish(self, v, omega):
        if bool(self.get_parameter("dry_run").value):
            return                      # 只算不发（车架起来时用）
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
        rclpy.shutdown()


if __name__ == "__main__":
    main()
