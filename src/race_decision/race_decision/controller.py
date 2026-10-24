# -*- coding: utf-8 -*-
"""
controller.py —— 双环控制器（纯 Python，不依赖 ROS）

外环（几何）：纯跟踪 pure pursuit
    在车前 Ld 处取车道中心（再加 e_ref 平移，避障用）上的目标点，车体坐标系左正：
        y(s) = −(e − e_ref) + s·tanθ + κ·s²/2
    要走的曲率 κ_cmd = 2·y / Ld²
    ★ 车在中心、方向对齐时 κ_cmd 恰好等于车道曲率 κ —— 曲率前馈是自带的，不用另外加。
    Ld = ld0 + ld_k·v（快了看远一点，更稳）

内环（偏航角速度，需要 IMU）：
    δ_舵机 = atan(L·κ_cmd)/steer_ratio  +  kr·(v·κ_cmd − r)  +  积分
    · 前一项是理想几何（舵机 25° 实测只转前轮 15.2° → steer_ratio = 0.608）
    · 后一项用陀螺仪实测 r 去追期望 r：舵机非线性、零偏、轮胎打滑都由它吃掉
    · 低速（v < yaw_loop_min_v）或没 IMU 时自动关闭内环，退化成纯几何

速度：v = min(max_speed, sqrt(a_lat / |κ|))，κ 取车道曲率和 κ_cmd 的较大者；
      v_cap 由状态机给（丢线/路口降速）；加减速斜坡。

输出 δ 是「舵机角，左正，弧度」，和 v1 的 /cmd_vel angular.z 约定一致。
"""
import math

DEFAULTS = {
    "wheelbase_m": 0.143,
    "steer_ratio": 0.608,          # 前轮角 / 舵机角（2026-10-24 陀螺仪实测 15.2/25）
    "steer_max_deg": 25.0,
    "steer_rate_deg_per_s": 300.0,
    "steer_trim_deg": 0.0,
    # 预瞄距离 = ld0 + ld_k·v，夹在 [ld_min, ld_max]。
    # 越长越稳但弯道越切内线（仿真扫参：0.22+0.5v 在 R0.55 弯里最大偏差最小）
    "ld0": 0.22,
    "ld_k": 0.5,
    "ld_min": 0.20,
    "ld_max": 0.60,
    "theta_clip_rad": 1.0,
    "kr": 0.3,                     # 偏航角速度环 P（舵机 rad / (rad/s)）
    "ki_r": 0.5,                   # 偏航角速度环 I
    "i_max_deg": 5.0,              # 积分限幅
    "yaw_loop_min_v": 0.08,
    "max_speed": 0.30,
    "min_speed": 0.15,
    "a_lat": 0.15,                 # 允许侧向加速度 m/s²（实际是在限「弯里多快」，调速主旋钮）
    "accel": 0.5,
    "decel": 1.2,
}


class Controller:
    def __init__(self, params=None):
        self.p = dict(DEFAULTS)
        if params:
            self.p.update(params)
        self.reset()

    def reset(self):
        self.v_cmd = 0.0
        self.delta = 0.0
        self.i_term = 0.0

    def lookahead(self, v):
        p = self.p
        return max(p["ld_min"], min(p["ld_max"], p["ld0"] + p["ld_k"] * v))

    def step(self, e, th, k, v_meas, r_meas, dt, e_ref=0.0, v_cap=None):
        """返回 (v_cmd [m/s], delta_servo [rad, 左正], info)。r_meas=None 表示没 IMU。"""
        p = self.p

        # ── 外环：纯跟踪 ──
        v_ref = max(self.v_cmd, 0.0)
        ld = self.lookahead(v_ref)
        thc = max(-p["theta_clip_rad"], min(p["theta_clip_rad"], th))
        y = -(e - e_ref) + ld * math.tan(thc) + 0.5 * k * ld * ld
        kappa_cmd = 2.0 * y / (ld * ld)

        # ── 速度 ──
        v_max = p["max_speed"]
        if v_cap is not None:
            v_max = min(v_max, v_cap)
        kp = max(abs(k), abs(kappa_cmd))
        v_target = min(v_max, math.sqrt(p["a_lat"] / kp)) if kp > 1e-3 else v_max
        v_target = max(v_target, min(p["min_speed"], v_max))
        if v_target > self.v_cmd:
            self.v_cmd = min(v_target, self.v_cmd + p["accel"] * dt)
        else:
            self.v_cmd = max(v_target, self.v_cmd - p["decel"] * dt)

        # ── 内环：几何前馈 + 偏航角速度反馈 ──
        L = p["wheelbase_m"]
        d_max = math.radians(p["steer_max_deg"])
        delta_ff = math.atan(L * kappa_cmd) / p["steer_ratio"]
        r_des = v_meas * kappa_cmd
        fb = 0.0
        if r_meas is not None and v_meas >= p["yaw_loop_min_v"] and p["kr"] > 0:
            err = r_des - r_meas
            i_max = math.radians(p["i_max_deg"])
            # 打满舵时不积分（抗饱和）
            if abs(self.delta) < d_max - 1e-3:
                self.i_term = max(-i_max, min(i_max, self.i_term + p["ki_r"] * err * dt))
            fb = p["kr"] * err + self.i_term
        else:
            self.i_term *= 0.98            # 没内环时积分慢慢放掉
        delta = delta_ff + fb + math.radians(p["steer_trim_deg"])
        delta_raw = delta

        delta = max(-d_max, min(d_max, delta))
        rate = math.radians(p["steer_rate_deg_per_s"]) * dt
        delta = max(self.delta - rate, min(self.delta + rate, delta))
        self.delta = delta

        info = {"kappa_cmd": kappa_cmd, "ld": ld, "r_des": r_des,
                "delta_ff": delta_ff, "delta_fb": fb, "delta_raw": delta_raw,
                "v_target": v_target}
        return self.v_cmd, delta, info
