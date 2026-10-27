# -*- coding: utf-8 -*-
"""
brain_node.py —— 大脑：/track（视觉） → /cmd_vel（底盘）

结构：
    ControlCore   纯控制逻辑，不依赖 ROS。车上和离线仿真跑的是**同一份**
                  （tools/make_sim_viewer.py 就是 import 这个类算 viewer 数据）
    BrainNode     只管 ROS：订阅、视觉超时、发话题、打日志

ControlCore.step() 从上到下 5 步：
    ① 失线滞回 → 保持转向 + 保持速度
    ② 滤波     → 航向角/曲率低通
    ③ 速度     → 按曲率限速 + 斜坡
    ④ 转向     → e 纠偏 + 曲率前馈 + 航向角
    ⑤ 限幅     → 舵角上下限 + 变化率 + 低通

所有可调量都在 PARAMS 表里。调参顺序见 docs/调参顺序.md。
"""
import math
from collections import deque


# ═══════════════════════════════════════════════════════════
#  参数表：唯一定义处。节点声明、调试网页的顺序和中文名都取自这里
#    (默认值, 网页显示名)；值为 None 的是分组标题
# ═══════════════════════════════════════════════════════════
PARAMS = {
    "#① 转向律：先调 stanley_k，直道不画龙再加 kappa_gain": None,
    "stanley_k":            (0.60,  "e 增益　画龙↓ / 修不回↑"),
    "kappa_gain":           (0.35,  "曲率前馈　切内线↑"),
    "theta_gain":           (0.0,   "航向角权重　最后才加"),
    "e_clip_m":             (0.35,  "e 夹取 m"),
    "v_stanley_min":        (0.50,  "纠偏分母速度下限"),
    "#② 速度包络：定完转向再定这个": None,
    "max_speed":            (0.21,  "最高速 m/s"),
    "a_lat":                (0.8,   "弯道侧向加速度　小=弯道慢"),
    "min_speed":            (0.0,   "最低速 m/s（0=不限）"),
    "speed_scale":          (1.0,   "速度倍率"),
    "quality_speed_floor":  (0.35,  "视觉质量低于此值时开始降速"),
    "quality_speed_min_scale": (0.55, "低质量恢复帧的最低速度比例"),
    "accel":                (0.8,   "加速度 m/s²"),
    "decel":                (1.5,   "减速度 m/s²"),
    "#③ 舵机：画龙调不掉时再动": None,
    "steer_max_deg":        (30.0,  "最大舵角 度（硬件 40）"),
    "steer_rate_deg_per_s": (110.0, "舵角变化率 度/秒"),
    "steer_tau_s":          (0.16,  "舵角低通 s（0=关）"),
    "steer_trim_deg":       (0.0,   "直线微调 度"),
    "steer_sign":           (1.0,   "方向符号　上车前必须确认"),
    "wheelbase_m":          (0.143, "轴距 m"),
    "#④ 滤波": None,
    "heading_tau_s":        (0.20,  "航向/曲率低通 s"),
    "#⑤ 失线保护：跑通之后再管": None,
    "lost_speed":           (0.15,  "丢线速度上限　=max_speed 则保持速度"),
    "lost_hold_n":          (3,     "丢线参考帧数"),
    "lost_hold_min_n":      (1,     "最少历史帧（1=不回正）"),
    "lost_enter_frames":    (3,     "进入丢线帧数"),
    "lost_exit_frames":     (8,     "退出丢线帧数"),
    "track_timeout_s":      (0.5,   "视觉超时 s"),
    "#⑥ 运行": None,
    "control_hz":           (50.0,  "控制频率 Hz"),
    "dry_run":              (False, "只算不发"),
}

DEFAULTS = {n: s[0] for n, s in PARAMS.items() if s is not None}


def lowpass(prev, new, tau, dt):
    """一阶低通。tau<=0 = 关闭。"""
    if tau <= 1e-4:
        return new
    a = min(1.0, dt / (tau + dt))
    return a * new + (1.0 - a) * prev


# ═══════════════════════════════════════════════════════════
#  ControlCore：控制逻辑，不依赖 ROS
#  车上：BrainNode 每个控制周期调一次（50 Hz）
#  离线：make_sim_viewer.py 每张图跑若干个周期（模拟 250 ms 的图间隔）
# ═══════════════════════════════════════════════════════════
class ControlCore:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = "WAIT"
        self.v = 0.0                # 速度斜坡状态
        self.steer_prev = 0.0       # 变化率限制状态
        self.delta_f = 0.0          # 舵角低通状态
        self.th_s = None            # 航向角低通状态
        self.k_s = None             # 曲率低通状态
        self.lost_cnt = self.ok_cnt = 0
        self.steer_hist = deque(maxlen=15)
        self.delta_raw = 0.0

    def step(self, e, th, kappa, is_lost, p, dt, new_frame=True, quality=1.0):
        """喂一帧视觉量，返回 (速度 m/s, 舵机角 rad 左正)。

        new_frame=False 表示这一 tick 没有新的视觉消息（控制 50 Hz、视觉 30 Hz，
        大部分 tick 是重复用同一帧）。失线计数只在 new_frame=True 时累加，
        否则 lost_enter_frames 就变成按控制周期算，而不是按视觉帧算。
        """
        hold = float(p["max_speed"]) * float(p["speed_scale"]) <= 1e-6

        # ① 失线滞回：保持转向（最近几帧中位数）+ 保持速度（只减速不加速）
        if new_frame:
            if is_lost:
                self.lost_cnt += 1
                self.ok_cnt = 0
            else:
                self.ok_cnt += 1
                self.lost_cnt = 0
        if (self.lost_cnt >= p["lost_enter_frames"]
                or (self.state == "LOST" and self.ok_cnt < p["lost_exit_frames"])):
            self.state = "LOST"
            recent = list(self.steer_hist)[-int(p["lost_hold_n"]):]
            steer = (float(sorted(recent)[len(recent) // 2])
                     if len(recent) >= int(p["lost_hold_min_n"]) else 0.0)
            v = float(p["lost_speed"])
            if self.v > 1e-3:
                v = min(v, self.v)          # 只会减速，绝不会加速
            self.v = 0.0 if hold else v
            self.delta_raw = steer
            return self.v, steer * float(p["steer_sign"])

        # ② 滤波：航向角和曲率
        self.state = "CRUISING"
        self.th_s = lowpass(self.th_s or 0.0, th, float(p["heading_tau_s"]), dt)
        self.k_s = lowpass(self.k_s or 0.0, kappa, float(p["heading_tau_s"]), dt)
        th, kappa = self.th_s, self.k_s

        # ③ 速度：v = min(max_speed, sqrt(a_lat/|κ|))，再按最低速兜底
        v_max = float(p["max_speed"]) * float(p["speed_scale"])
        k_abs = abs(kappa)
        v_target = min(v_max, math.sqrt(float(p["a_lat"]) / k_abs)
                       if k_abs > 1e-3 else v_max)
        # 质量只负责平滑降速，不直接刹停；连续丢线仍由 LOST 状态机处理。
        q = max(0.0, min(1.0, float(quality)))
        q_floor = float(p.get("quality_speed_floor", 0.35))
        q_min = float(p.get("quality_speed_min_scale", 0.55))
        if q < q_floor:
            v_target *= q_min
        elif q < 1.0:
            v_target *= q_min + (1.0 - q_min) * ((q - q_floor) / max(1.0 - q_floor, 1e-6))
        if float(p["min_speed"]) > 0.0:
            v_target = max(v_target, float(p["min_speed"]))
        dv = (float(p["accel"]) if v_target > self.v else float(p["decel"])) * dt
        self.v = (min(v_target, self.v + dv) if v_target > self.v
                  else max(v_target, self.v - dv))
        v = self.v

        # ④ 转向：delta = theta_gain·θ − atan(k·e/v) + kappa_gain·atan(L·κ) + trim
        lim_e = float(p["e_clip_m"])                    # 路口 e 可能 1 米，不夹会打满舵
        e_c = max(-lim_e, min(lim_e, e))
        v_eff = max(v, float(p["v_stanley_min"]))       # 否则 v→0 时纠偏角冲到 90°
        delta = (float(p["theta_gain"]) * th
                 - math.atan(float(p["stanley_k"]) * e_c / v_eff)
                 + math.atan(float(p["wheelbase_m"]) * kappa) * float(p["kappa_gain"])
                 + math.radians(float(p["steer_trim_deg"])))
        self.delta_raw = delta

        # ⑤ 限幅 → 变化率 → 低通
        d_max = math.radians(float(p["steer_max_deg"]))
        delta = max(-d_max, min(d_max, delta))
        rate = math.radians(float(p["steer_rate_deg_per_s"])) * dt
        delta = max(self.steer_prev - rate, min(self.steer_prev + rate, delta))
        self.steer_prev = delta
        self.delta_f = lowpass(self.delta_f, delta, float(p["steer_tau_s"]), dt)
        self.steer_hist.append(self.delta_f)
        return v, self.delta_f * float(p["steer_sign"])


# ═══════════════════════════════════════════════════════════
#  BrainNode：只做 ROS 的事（延迟 import，让 ControlCore 能离线用）
# ═══════════════════════════════════════════════════════════
def build_node():
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from custom_msgs.msg import TrackInfo

    class BrainNode(Node):
        def __init__(self):
            super().__init__("brain_node")
            for name, spec in PARAMS.items():
                if spec is not None:
                    self.declare_parameter(name, spec[0])
            self.declare_parameter("ui_spec", ";".join(
                f"{n}|{s[1]}" if s is not None else f"#|{n[1:]}"
                for n, s in PARAMS.items()))

            self.core = ControlCore()
            self.last = None
            self.last_time = 0.0
            self.n_ok = self.n_lost = 0
            self.t_log = 0.0
            self.last_stamp = None

            self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
            self.sub = self.create_subscription(TrackInfo, "/track", self.on_track, 10)
            hz = max(float(self.get_parameter("control_hz").value), 1.0)
            self.timer = self.create_timer(1.0 / hz, self.on_timer)
            self.get_logger().info(
                f"brain_node 启动：{hz:.0f} Hz，"
                f"最高速 {self.get_parameter('max_speed').value} m/s")

        def params(self):
            return {n: self.get_parameter(n).value for n in DEFAULTS}

        def on_track(self, msg):
            self.last = msg
            self.last_time = self.get_clock().now().nanoseconds * 1e-9

        def on_timer(self):
            now = self.get_clock().now().nanoseconds * 1e-9
            p = self.params()
            dt = 1.0 / max(float(p["control_hz"]), 1.0)

            # 视觉超时 → 停车（最后一道保险，ROS 侧的事）
            if self.last is None or (now - self.last_time) > p["track_timeout_s"]:
                self.core.state = "WAIT_VISION" if self.last is None else "VISION_TIMEOUT"
                self.core.v = 0.0
                self.publish(0.0, 0.0)
                return

            msg = self.last
            stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            new_frame = stamp != self.last_stamp
            self.last_stamp = stamp
            v, delta = self.core.step(
                float(msg.lateral_error), float(msg.heading_error),
                float(msg.curvature), bool(msg.is_lost), p, dt, new_frame,
                quality=float(getattr(msg, "quality", 1.0)))
            if msg.is_lost:
                self.n_lost += 1
            else:
                self.n_ok += 1
            self.publish(v, delta)

            if self.core.state == "LOST":
                self.get_logger().warn(
                    f"丢线(黄点 {msg.n_points}): {msg.reject_reason}"
                    f" | 保持转向 {math.degrees(self.core.delta_raw):+.1f}°"
                    f" v={v:.2f}", throttle_duration_sec=2.0)
            elif now - self.t_log > 2.0:
                self.t_log = now
                th = self.core.th_s or 0.0
                k = self.core.k_s or 0.0
                self.get_logger().info(
                    f"[CRUISING] e={msg.lateral_error * 100:+6.1f}cm "
                    f"θ={math.degrees(th):+5.2f}° κ={k:+.3f} → "
                    f"δ原始={math.degrees(self.core.delta_raw):+6.1f}° "
                    f"δ={math.degrees(delta):+6.1f}° v={v:.3f}"
                    f" | ok={self.n_ok} lost={self.n_lost}")
                self.n_ok = self.n_lost = 0

        def publish(self, v, omega):
            if bool(self.get_parameter("dry_run").value):
                return
            t = Twist()
            t.linear.x = float(v)
            t.angular.z = float(omega)      # 底盘按「舵机角」直接收
            self.pub.publish(t)

    return BrainNode


def main():
    import rclpy
    rclpy.init()
    node = build_node()()
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
