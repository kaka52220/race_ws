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
        self.declare_parameter("steer_rate_deg_per_s", 200.0)  # 舵机变化率上限
        self.declare_parameter("steer_sign", 1.0)         # 方向翻转。★ 2026-10-23 静态实测：车偏左时前轮应指车道中心；设为 -1 时指外侧 → 正确的就是 +1
        self.declare_parameter("steer_tau_s", 0.08)       # 转向低通时间常数（秒）。画龙就调大，0=关闭

        # 控制律
        self.declare_parameter("stanley_k", 1.5)          # 横向纠偏增益，越大纠得越猛
        self.declare_parameter("v_eps", 0.0)              # 已由 v_stanley_min 提供固定分母，这里归零

        # 速度
        # ══════════════════════════════════════════════════════════
        # ★★ 速度单位说明（2026-10-23 编码器实测，别改错 ★★
        #   linear.x 在 base_node 的 run 模式下 = 归一化 0~1，
        #   内部换算成 set_car_run 的百分比（linear.x=0.01 → 1%）。
        #   这台车（亚博控制板 V3.0 / 固件 3.5 / 编码器 1267 脉冲每轮圈）实测：
        #       1% → 0.163 m/s   ← 唯一可用的慢速
        #       2% → 0.860 m/s   ← 直接跳 5 倍
        #       3%~6% → 1.2 m/s  ← 饱和
        #   所以 max_speed 只能取 0.01；取 0.02 就变 0.86 m/s，
        #   取 0.03 会被 min_speed 顶到 3% → 1.2 m/s 直接冲出赛道。
        # ══════════════════════════════════════════════════════════
        self.declare_parameter("max_speed", 0.10)         # 全局最高速 m/s
        self.declare_parameter("min_speed", 0.10)         # 转弯时最低速，别停
        self.declare_parameter("speed_scale", 1.0)        # 一键倍率：0.5 = 全部减半
        self.declare_parameter("a_lat", 0.8)              # 允许的侧向加速度 m/s²
        self.declare_parameter("accel", 0.8)              # 加速度上限
        self.declare_parameter("decel", 1.5)              # 减速度上限

        # 安全 / 调试
        self.declare_parameter("track_timeout_s", 0.5)    # 多久没视觉就停车
        self.declare_parameter("lost_speed", 0.0)        # 丢线时的固定爬行速度（不要用乘法，会指数衰减）
        self.declare_parameter("lost_enter_frames", 3)    # 连续丢几帧才进 LOST（滞回，防抖）
        self.declare_parameter("lost_exit_frames", 8)     # 连续好帧几帧才退出 LOST
        self.declare_parameter("v_stanley_min", 0.50)     # Stanley 分母的固定参考值（固件单位未知，改成常数=纯比例控制）
        self.declare_parameter("dry_run", False)          # true = 只算不发
        self.declare_parameter("control_hz", 50.0)

        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.sub = self.create_subscription(TrackInfo, "/track", self.on_track, 10)

        self.last = None            # 最近一次的 TrackInfo
        self.last_time = 0.0
        self.v_cmd = 0.0            # 当前指令速度（做加减速斜坡用）
        self.steer_prev = 0.0       # 上一帧舵机角（做变化率限制用）
        self.delta_f = 0.0          # 低通滤波后的舵机角
        self.state = "WAIT"
        self.lost_cnt = 0           # 连续丢线帧数（滞回用）
        self.ok_cnt = 0             # 连续正常帧数（滞回用）
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

        # ── 丢线处理：固定爬行速度 + 滞回，**不能指数衰减** ──
        #   老写法 v = v_cmd * lost_slowdown 每 20ms 执行一次，等于每 20ms 乘 0.5，
        #   5 个周期就归零 → 车走走停停、速度锯齿（2026-10-23 实测日志抓到的）。
        #   而且 CRUISING↔LOST 快速来回切，速度就在 0 和满速之间摆。
        if msg.is_lost:
            self.lost_cnt += 1
            self.ok_cnt = 0
        else:
            self.ok_cnt += 1
            self.lost_cnt = 0

        enter_lost = self.lost_cnt >= int(g("lost_enter_frames").value)
        still_lost = self.state == "LOST" and self.ok_cnt < int(g("lost_exit_frames").value)

        # ★ max_speed=0 表示「绝对停住，只算方向」——调试前轮用。
        #   否则 min_speed / lost_speed 会把速度抬起来，车一直转，没法看前轮。
        hold = float(g("max_speed").value) * float(g("speed_scale").value) <= 1e-6

        if enter_lost or still_lost:
            self.state = "LOST"
            # 用一个**固定**的爬行速度，而不是在当前速度上乘系数
            v = 0.0 if hold else float(g("lost_speed").value)
            self.publish(v, 0.0)                # 丢线时方向盘回正，别乱打
            self.v_cmd = v
            # 把视觉给的失败原因打出来，省得每次都去开调试图
            self.get_logger().warn(
                f"丢线(黄点 {msg.n_points}): {msg.reject_reason}",
                throttle_duration_sec=2.0)
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
        if v_max > 1e-6:                       # max_speed=0 时不要被 min_speed 抬起来
            v_target = max(v_target, float(g("min_speed").value))

        # 加减速斜坡（别一脚油门一脚刹车）
        if v_target > self.v_cmd:
            v_cmd = min(v_target, self.v_cmd + float(g("accel").value) * 0.02)
        else:
            v_cmd = max(v_target, self.v_cmd - float(g("decel").value) * 0.02)
        self.v_cmd = v_cmd

        # ── Stanley 转向 ──
        #   ⚠ 分母用 max(v_cmd, v_stanley_min)：车快停时会 v→0，分母趋近 v_eps，
        #     纠偏角会冲到 90°，实测出现过 δ原始=-92.8° 的爆表值。
        #     给分母一个速度下限，避免低速时纠偏项失控。
        steer_ff = math.atan(L * kappa)                      # 曲率前馈
        v_eff = max(v_cmd, float(g("v_stanley_min").value))
        corr = math.atan(float(g("stanley_k").value) * e
                         / (v_eff + float(g("v_eps").value)))
        delta = th - corr + steer_ff + math.radians(float(g("steer_trim_deg").value))
        delta_raw = delta          # 滤波/限幅前的原始转角，只用于日志诊断

        # 限幅
        d_max = math.radians(float(g("steer_max_deg").value))
        delta = max(-d_max, min(d_max, delta))
        # 变化率限制（舵机从 0 打到大角度要 80~120 ms，直接阶跃会甩头）
        rate = math.radians(float(g("steer_rate_deg_per_s").value)) * 0.02
        delta = max(self.steer_prev - rate, min(self.steer_prev + rate, delta))
        self.steer_prev = delta

        # ── 一阶低通阻尼（Stanley 本身没有阻尼项，不加会画龙）──
        #   为什么需要：纯比例纠偏 + 系统延迟（视觉 16ms + 舵机响应 + 底盘）
        #   = 相位滞后 = 正反馈 = 左右来回摆。这是控制里的常识。
        #   tau 越大越平滑但越迟钝；tau=0 等于关掉滤波。
        #     a = dt / (tau + dt)，dt 是控制周期（1/control_hz）
        tau = float(g("steer_tau_s").value)
        if tau > 1e-4:
            dt = 1.0 / max(float(g("control_hz").value), 1.0)
            a = dt / (tau + dt)
            self.delta_f = a * delta + (1.0 - a) * self.delta_f
            delta = self.delta_f
        else:
            self.delta_f = delta

        # ── 关键：base_driver 把 angular.z 当成「舵机角（弧度）」直接收，不做运动学换算 ──
        #   队友 base_node.py 里的写法：
        #       steer_angle = int(-v_z * 57.3)      # 等价于 steer_angle = -degrees(angular.z)
        #       steer_angle = max(-30, min(30, steer_angle))
        #
        #   ★ 2026-10-23 实测确认：发 angular.z=+0.5 → steer_angle=-28.6° → 前轮往【左】
        #     所以底盘的约定是「舵机角为负 = 左，为正 = 右」，和常规正好相反。
        #
        #   我们的 delta 是「左正」：delta>0 要往左打，就需要 steer_angle = -degrees(delta)。
        #   又因为 steer_angle = -degrees(omega)，所以：
        #       -degrees(omega) = -degrees(delta)  ⟹  omega = +delta
        #
        #   ⚠ 绝不能用 omega = v·tan(δ)/L —— 那会让符号反掉，且幅度随速度翻倍。
        omega = delta * float(g("steer_sign").value)
        self.publish(0.0 if hold else v_cmd, omega)

        # 每 2 秒打一行日志
        if now - self.t_last_log > 2.0:
            self.t_last_log = now
            self.get_logger().info(
                f"[{self.state}] e={e*100:+6.1f}cm θ={math.degrees(th):+5.2f}° "
                f"κ={kappa:+.3f} → δ原始={math.degrees(delta_raw):+6.1f}° "
                f"滤波后={math.degrees(delta):+6.1f}° "
                f"v={v_cmd:.3f} | ok={self.n_ok} lost={self.n_lost}")
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
