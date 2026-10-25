# -*- coding: utf-8 -*-
"""
brain_node.py —— 决策 + 运动控制

它订阅 /track（视觉给的三个数），算出方向盘打多少、油门给多少，发 /cmd_vel。

这一版**只做一件事：跟着车道线走**。
斑马线停车、挡板避障、计圈这些，等 YOLO 出来再往这个骨架上加状态。

控制律：Stanley（比 PID 稳，参数有物理意义）

    δ = θ_e  −  atan(k·e / v)  +  atan(L·κ)
        └─航向─┘  └──横向纠偏──┘   └─曲率前馈─┘
    （v 取 max(v_cmd, v_stanley_min)，防止低速时分母趋零把纠偏角放大到 90°）

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

from collections import deque
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
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
        self.declare_parameter("steer_tau_s", 0.05)
        # ★ 实测 0.08 时输出滞后明显，0.05 更好。再小方向会抖。       # 转向低通时间常数（秒）。画龙就调大，0=关闭

        # 控制律
        # 横向纠偏增益。★ 2026-10-24 陀螺仪实测标定过：
        #   set_akm_steering_angle(25) 实际只让前轮转 15.2°，转弯半径 0.53 m。
        #   k=1.5 时，e=10cm 就要走 R=0.81 m 的弯 —— 在 54cm 宽车道里等于打半舵。
        #   改 0.7 后：e=5cm → R=3.4m，e=10cm → R=1.7m，e=20cm → R=0.86m。
        #   觉得还不够柔就继续往 0.5 调；觉得跟弯慢就往 1.0 调。
        self.declare_parameter("stanley_k", 0.7)
        # 转向律的权重（0~1）。见下方 on_track 里的说明。
        #   实测：e 平滑可信、theta 乱跳，所以先只用 e。
        self.declare_parameter("theta_gain", 0.3)   # 航向角权重。
        # θ 先夹到 ±这个角度再乘 theta_gain。
        # 实测日志里 θ 会跳到 ±71°，0.3×71 = 21° 的野值。
        self.declare_parameter("theta_clip_deg", 30.0)
        #   ★ 2026-10-24 调通的值：0.3。进弯靠它提前打方向。
        #     设 0 会明显「进弯滞后」（少了 0.3×θ 这一项前馈）。
        self.declare_parameter("kappa_gain", 0.5)   # 曲率前馈权重。
        #   前馈角 = kappa_gain × atan(0.143 × κ)。
        #   κ=1.5 时：0.5→6.0°，1.0→12.1°。弯道转不过来就往上加。

        # 速度：linear.x 是归一化 0~1，base_node 乘 100 得到 set_car_run 的百分比。
        # 新固件实测（2026-10-24，落地编码器）：每 1% ≈ 0.0134 m/s，线性、带负载补偿。
        #   speed  8 → 0.106 m/s    speed 12 → 0.161 m/s
        #   speed 10 → 0.138 m/s    speed 15 → 0.208 m/s
        # 所以 max_speed=0.12 → 12% → 0.161 m/s。
        self.declare_parameter("max_speed", 0.60)         # 全局最高速（米/秒）
        self.declare_parameter("min_speed", 0.20)         # 转弯时最低速（米/秒），别停
        self.declare_parameter("speed_scale", 1.0)        # 一键倍率：0.5 = 全部减半
        self.declare_parameter("a_lat", 0.17)
        # ★ 0.15 才有弯道减速：v = √(a_lat/κ)。
        #   κ=1.5 时 a_lat=0.15 → v=0.32；若取 0.8 → v=0.73 相当于不减速，
        #   全速冲进弯会「转不过来」。这个参数是弯道能不能过的关键。              # 允许的侧向加速度 m/s²
        self.declare_parameter("accel", 0.8)              # 加速度上限
        self.declare_parameter("decel", 1.5)              # 减速度上限

        # 安全 / 调试
        self.declare_parameter("track_timeout_s", 0.5)    # 多久没视觉就停车
        self.declare_parameter("lost_speed", 0.40)        # 丢线时的速度上限。
        # ★ 实际速度 = min(lost_speed, 丢线前速度) —— 只会减速不会加速。
        #   设成和 max_speed 一样 = 完全保持原速度（配合保持转向 = 按原轨迹继续走）。
        #   想减速就调小；0.07 是"急刹爬行"，弯道里会看起来像停车。
        # ★ 丢线时「往哪转往哪转」：参考最近几帧的转向角，不回正。
        #   弯道里一丢线就回正，车会直冲出去（2026-10-24 推车过弯实测）。
        self.declare_parameter("lost_hold_n", 3)          # 参考最近几帧
        self.declare_parameter("lost_hold_min_n", 1)      # 至少几帧历史才沿用。1 = 只要跑过一帧就不回正
        self.declare_parameter("lost_enter_frames", 3)    # 连续丢几帧才进 LOST（滞回，防抖）
        self.declare_parameter("lost_exit_frames", 8)     # 连续好帧几帧才退出 LOST
        self.declare_parameter("v_stanley_min", 0.30)     # Stanley 分母的速度下限（米/秒）。
        # 真车才 0.21 m/s，这个 0.5 只是个「固定参考速度」，让纠偏项退化成纯比例控制，
        # 避免低速时 1/v 把纠偏角放大到 90°（实测出现过 -92.8°）。
        # e 进入控制律前先夹到这个范围（米）。宽区域（广场/路口）视觉给的 e 可能到 1 米，
        # 不夹的话纠偏项直接打满舵，等于闭着眼睛拐弯。0.35 m 已超过半个窄车道宽。
        self.declare_parameter("e_jump_max_m", 0.06)
        # ★ 单帧跳变保护：视觉 30Hz，一个周期（33ms）车最多走 1.3cm，
        #   e 一帧跳超过 6cm 必定是锁错线。实测出弯时出现过 -0.053 → -0.706
        #   → -0.005 → -0.412 这种乱跳，不挡的话方向盘会来回甩，把车甩出赛道。
        self.declare_parameter("e_jump_frames", 5)
        #   连续这么多帧都跳，说明是真位移（车真的偏了），就接受
        self.declare_parameter("e_clip_m", 0.20)
        # ★ 车道总宽才 0.54m，|e|>0.27 说明车已经在赛道外了。
        #   0.35 时纠偏角 = atan(0.7×0.35/0.5) = 26° → 打满舵。
        #   实测日志里出现过 e=+86cm 的野值，按 0.35 算直接甩出赛道。
        # 航向角/曲率的一阶低通时间常数（秒）。越大越平滑、响应越慢。
        # 实测 theta 单帧噪声中位 10°，不滤的话转向一直在抖。
        self.declare_parameter("heading_tau_s", 0.06)
        # ★ 实测 0.12 时弯道出口回正慢半拍，0.06 明显好很多。
        #   它同时滤 theta 和 kappa，再小会让曲率噪声进方向。

        self.th_s = None          # 航向角平滑值
        self.k_s = None           # 曲率平滑值
        self.declare_parameter("dry_run", False)          # true = 只算不发
        self.declare_parameter("control_hz", 50.0)
        # ── 斑马线停车（视觉阈值法发 /crosswalk，见 crosswalk.py）──
        #   分数是"地面极亮面积百分比"，越大 = 离斑马线越近。
        #   实测：>=8% 约 0.7 米内；负样本 207 张 / 挡板 143 张零误检。
        self.declare_parameter("crosswalk_trigger", 8.0)   # 超过这个分数就停车
        # ★ 先减速再停：高速（0.6）冲过来时刹车距离不够，会停在斑马线上。
        #   分数到 crosswalk_slow_trigger 就先把速度压到 crosswalk_slow_speed，
        #   等真到 crosswalk_trigger 再停 —— 这样停得准，又不用把停车阈值降太低。
        self.declare_parameter("crosswalk_slow_trigger", 3.0)
        self.declare_parameter("crosswalk_slow_speed", 0.25)
        self.declare_parameter("crosswalk_stop_s", 3.0)    # 停几秒（规则要求 3s）
        self.declare_parameter("crosswalk_rearm", 0.6)     # 分数掉到触发值的这个比例以下，才允许下次再停
        # 冷却：停完之后这么多秒内不再触发。防止分数在阈值附近抖动时
        # 反复触发把自己永久锁死（实测过：假值和真值抢话题就会这样）。
        self.declare_parameter("crosswalk_cooldown_s", 1.0)

        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.sub = self.create_subscription(TrackInfo, "/track", self.on_track, 10)
        self.create_subscription(Float32, "/crosswalk", self.on_crosswalk, 10)

        self.last = None            # 最近一次的 TrackInfo
        self.last_time = 0.0
        self.v_cmd = 0.0            # 当前指令速度（做加减速斜坡用）
        self.steer_prev = 0.0       # 上一帧舵机角（做变化率限制用）
        self.delta_f = 0.0          # 低通滤波后的舵机角
        self.state = "WAIT"
        self.lost_cnt = 0           # 连续丢线帧数（滞回用）
        self.cw = 0.0               # 最新斑马线分数
        self.cw_stop_until = 0.0    # 停车截止时刻
        self.cw_armed = True        # 是否允许触发（过一个斑马线只停一次）
        self.cw_done_at = 0.0       # 上次停车结束的时刻（冷却用）
        self.e_prev = None          # 上一帧采信的 e（跳变保护用）
        self.e_jump_cnt = 0         # 连续跳变帧数
        # 最近若干帧的转向角，丢线时沿用（「往哪转往哪转」，不回正）
        self.steer_hist = deque(maxlen=15)
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

    def on_crosswalk(self, msg):
        self.cw = float(msg.data)

    # ─────────────────────────────────────────────────────────
    def on_timer(self):
        g = self.get_parameter
        now = self.get_clock().now().nanoseconds / 1e9
        v_max = float(g("max_speed").value) * float(g("speed_scale").value)
        # 斑马线前预减速：分数够了就先压速，别等冲过头再刹
        if self.cw >= float(g("crosswalk_slow_trigger").value):
            v_max = min(v_max, float(g("crosswalk_slow_speed").value))

        # ── 视觉超时：这是最后一道保险，必须最先判 ──
        if self.last is None or (now - self.last_time) > float(g("track_timeout_s").value):
            self.state = "WAIT_VISION" if self.last is None else "VISION_TIMEOUT"
            self.publish(0.0, 0.0)
            self.v_cmd = 0.0
            return

        msg = self.last

        # ── 斑马线停车：分数超阈值就停 crosswalk_stop_s 秒，然后继续 ──
        #   滞回：分数要掉到触发值的 crosswalk_rearm 倍以下才重新武装，
        #   否则停完 3 秒会立刻再触发一次，车就卡在斑马线不动了。
        trig = float(g("crosswalk_trigger").value)
        cool = float(g("crosswalk_cooldown_s").value)
        if self.cw >= trig and self.cw_armed and now >= self.cw_done_at + cool:
            self.cw_stop_until = now + float(g("crosswalk_stop_s").value)
            self.cw_armed = False
            self.get_logger().warn(
                f"斑马线 {self.cw:.1f}% >= {trig:.1f}% → 停车 "
                f"{float(g('crosswalk_stop_s').value):.1f}s",
                throttle_duration_sec=2.0)
        elif self.cw < trig * float(g("crosswalk_rearm").value) \
                and now >= self.cw_done_at + cool:
            self.cw_armed = True
        if now < self.cw_stop_until:
            self.state = "STOP_CROSSWALK"
            self.v_cmd = 0.0
            self.publish(0.0, 0.0)
            return
        elif self.cw_stop_until > 0.0 and self.cw_done_at < self.cw_stop_until:
            self.cw_done_at = self.cw_stop_until      # 停完了，记下时刻，开始冷却

        # ── 丢线处理：固定爬行速度 + 滞回 ──
        #   不要写成「每帧把速度乘一个系数」：20ms 一帧乘 0.5，五个周期就归零，
        #   表现为走走停停、速度锯齿，而且 CRUISING↔LOST 来回切。
        #   给一个固定的 lost_speed，配合 enter/exit 帧数滞回，就平稳了。
        #   ★ 可信度判定（质量、跨度、e/theta/kappa 超限）全部在 vision 做完了，
        #     这里只看 msg.is_lost，不再重复判 —— 见 06_代码架构与职责边界.md
        if msg.is_lost:
            self.lost_cnt += 1
            self.ok_cnt = 0
        else:
            self.ok_cnt += 1
            self.lost_cnt = 0

        enter_lost = self.lost_cnt >= int(g("lost_enter_frames").value)
        still_lost = self.state == "LOST" and self.ok_cnt < int(g("lost_exit_frames").value)

        # ★ max_speed=0 表示「绝对停住，只算方向」——调试前轮用。
        hold = float(g("max_speed").value) * float(g("speed_scale").value) <= 1e-6

        if enter_lost or still_lost:
            self.state = "LOST"
            # 用一个**固定**的爬行速度，而不是在当前速度上乘系数
            v = 0.0 if hold else float(g("lost_speed").value)
            # ★ 丢线时不要回正，保持「刚才往哪转」。
            #   原来这里是 publish(v, 0.0) —— 方向盘瞬间回正。
            #   实测后果（2026-10-24 推车过弯）：弯道里一丢线就掰直，
            #   车直冲出去，等重新看见线时已经在赛道外了。
            #   取最近 lost_hold_n 帧真实转向角的**中位数**：
            #     · 中位数不怕偶发野值（实测混一个 -80° 进去，平均被带到 0°，中位数还是 20°）
            #     · 只取最近几帧，不用整段历史，弯道里不会滞后
            #   为什么用历史而不是外推预测：只需要「别停、别回正」，
            #   外推在弯道里容易打过头（舵机 25° 才转前轮 15.2°，本来就非线性）。
            n_hold = int(g("lost_hold_n").value)
            recent = list(self.steer_hist)[-n_hold:]
            if len(recent) >= int(g("lost_hold_min_n").value):
                steer_lost = float(np.median(recent))
            else:
                steer_lost = 0.0        # 只有刚上电、一帧都没跑过时才会走到这里
            # ⚠ 必须和正常帧一样乘 steer_sign，否则改了符号后丢线会往反方向打
            self.publish(v, steer_lost * float(g("steer_sign").value))
            self.v_cmd = v
            # 把视觉给的失败原因打出来，省得每次都去开调试图
            self.get_logger().warn(
                f"丢线(黄点 {msg.n_points}): {msg.reject_reason}"
                f" | 保持转向 {math.degrees(steer_lost):+.1f}°",
                throttle_duration_sec=2.0)
            return

        self.state = "CRUISING"
        kappa = float(msg.curvature)
        e = float(msg.lateral_error)
        th = float(msg.heading_error)

        # ── e 单帧跳变保护 ──
        #   33ms 内车最多走 1.3cm，e 跳超过 e_jump_max_m 说明视觉锁错线了。
        #   这一帧沿用上一帧的值；但连续 e_jump_frames 帧都在跳，说明是真的偏了，接受。
        if self.e_prev is not None and abs(e - self.e_prev) > float(g("e_jump_max_m").value):
            self.e_jump_cnt += 1
            if self.e_jump_cnt < int(g("e_jump_frames").value):
                e = self.e_prev
            else:
                self.e_jump_cnt = 0          # 持续跳 = 真位移，接受
        else:
            self.e_jump_cnt = 0
        self.e_prev = e
        L = float(g("wheelbase_m").value)

        # ── 航向角/曲率的时序低通 ──
        #   实测（2026-10-24）：同一帧图只改鸟瞰图深度范围，theta 就会差 10°（中位）、
        #   最坏 79°。车跑起来每帧都在变，不滤波的话转向一直在抖。
        #   一阶低通：alpha = dt / tau，dt = 控制周期 20ms。
        self.th_s = 0.0 if self.th_s is None else self.th_s
        self.k_s = 0.0 if self.k_s is None else self.k_s
        a = min(1.0, 0.02 / max(float(g("heading_tau_s").value), 1e-3))
        self.th_s += a * (th - self.th_s)
        self.k_s += a * (kappa - self.k_s)
        th = self.th_s
        kappa = self.k_s

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

        # ── 转向律 ──
        #   delta = theta_gain·θ  −  atan(k·e/v)  +  kappa_gain·atan(L·κ)
        #
        #   ★ 2026-10-24 静态扫描的结论：**e 可信，theta 不可信**。
        #     车摆在车道正中、车头顺着车道，沿弯道取 5 个位置：
        #         e     : -0.4  -2.4  +6.4  +18.9  +19.8 cm   ← 平滑、单调、合理
        #         theta : -1.3  +23.2 +41.7  -8.4  -29.1 deg   ← 乱跳、方向都会反
        #     用 theta 打方向就会「提前切内线、再冲出外侧」，而 theta 一整天没修好。
        #     所以把 theta 和 kappa 做成权重，默认先关掉，只用横向误差。
        #
        #   代价：没有前馈，弯道会滞后一点（等 e 变大了才修）。
        #   好处：不会因为一个错误的 40° 航向角打满舵。
        #   等 e 控制跑通之后，可以一点点把 theta_gain 加回来（0.2 / 0.4 ...）。
        #   分母给速度下限：车快停时 v→0，纠偏角会冲到 90°（实测出现过 -92.8°）。
        #   e 先夹到 ±e_clip_m：宽区域视觉给出的 e 可能到 1 米，不夹会直接打满舵。
        e_c = max(-float(g("e_clip_m").value), min(float(g("e_clip_m").value), e))
        # θ 也夹一下：视觉偶尔给出 ±70° 的野值，不限幅就是 0.3×70=21° 的方向突变
        th_max = math.radians(float(g("theta_clip_deg").value))
        th_c = max(-th_max, min(th_max, th))
        v_eff = max(v_cmd, float(g("v_stanley_min").value))
        corr = math.atan(float(g("stanley_k").value) * e_c / v_eff)
        steer_ff = math.atan(L * kappa) * float(g("kappa_gain").value)
        delta = (float(g("theta_gain").value) * th_c - corr + steer_ff
                 + math.radians(float(g("steer_trim_deg").value)))
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

        # 记下这一帧的转向角，丢线时用它「往哪转往哪转」
        self.steer_hist.append(float(delta))

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
