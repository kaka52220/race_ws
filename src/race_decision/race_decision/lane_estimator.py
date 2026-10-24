# -*- coding: utf-8 -*-
"""
lane_estimator.py —— 车道相对位姿的卡尔曼滤波（纯 Python + numpy，不依赖 ROS）

状态 x = [e, θ, κ]，符号与 TrackInfo 一致（左正）：
    e  车体中心相对车道中心的横向偏差 [m]，车偏左为正
    θ  车道方向相对车头的夹角 [rad]，车道往左拐为正
    κ  车道曲率 [1/m]，左弯为正

运动模型（车速 v、偏航角速度 r，r 左转为正）：
    de/dt = −v·sinθ        车头偏向车道右侧（θ>0）时车往右漂
    dθ/dt =  v·κ − r       车道自己在转，车头也在转
    dκ/dt =  0 （随机游走）

为什么要它（相对 v1 的三道低通滤波）：
    · 两帧视觉之间用陀螺仪 + 车速预测，平滑但**不滞后**
    · 视觉帧是 50~80 ms 前拍的：按图像时间戳把测量推到「现在」再融合（延迟补偿）
    · 单帧野值（路口虚线上 e 跳 60 cm）用马氏距离门限直接丢掉
    · 丢线 / 路口时只预测不更新，等于用 IMU 推算继续走
"""
import math

import numpy as np


class LaneEstimator:
    def __init__(self, sigma_e=0.02, sigma_th=0.08, sigma_k=0.4,
                 q_e=0.002, q_th=0.05, q_k=2.0,
                 gate=16.0, reject_reset=8, relock_e_spread=0.06,
                 relock_th_spread=0.25, max_jump_e=0.15, max_jump_th=0.45,
                 max_latency_s=0.3):
        # 视觉测量噪声（quality=1 时）
        self.R0 = np.diag([sigma_e ** 2, sigma_th ** 2, sigma_k ** 2])
        # 过程噪声谱密度（每秒）
        self.Qd = np.diag([q_e, q_th, q_k])
        self.gate = gate                    # 3 自由度卡方门限（99% ≈ 11.3，这里放宽一点）
        # 连续拒绝 reject_reset 帧，且这些被拒的测量**彼此一致**（e 极差 < relock_e_spread，
        # θ 极差 < relock_th_spread）→ 认为是估计漂了、视觉是对的，重新锁定。
        # 路口虚线那种每帧乱跳的测量互相不一致，永远不会触发重锁。
        self.reject_reset = reject_reset
        self.relock_e_spread = relock_e_spread
        self.relock_th_spread = relock_th_spread
        self.max_latency_s = max_latency_s
        # 绝对跳变上限：不管 P 多大，单帧 e 跳超过 max_jump_e 或 θ 跳超过 max_jump_th 都不采纳
        # （长时间推算后 P 变大、门限变松，路口野值就可能混进来）
        self.max_jump_e = max_jump_e
        self.max_jump_th = max_jump_th
        self.reset()

    # ─────────────────────────────────────────────────────────
    def reset(self):
        self.x = np.zeros(3)
        self.P = np.diag([0.1 ** 2, 0.3 ** 2, 2.0 ** 2])
        self.t = None
        self.initialised = False
        self.n_reject = 0
        self.rejected = []      # 最近连续被拒的测量（已推到当时的「现在」）
        self.hist = []          # [(t_start, t_end, v, r)] 最近 max_latency_s 的输入

    def state(self):
        return float(self.x[0]), float(self.x[1]), float(self.x[2])

    def sigma_e(self):
        return math.sqrt(max(self.P[0, 0], 0.0))

    def force_straight(self):
        """路口：认定前方是直路，曲率清零（不确定度也压小，别被预测拉偏）。"""
        self.x[2] = 0.0
        self.P[2, :] = 0.0
        self.P[:, 2] = 0.0
        self.P[2, 2] = 0.05 ** 2

    # ─────────────────────────────────────────────────────────
    @staticmethod
    def _step(x, v, r, dt):
        e, th, k = x
        return np.array([e - v * math.sin(th) * dt, th + (v * k - r) * dt, k])

    def predict(self, t, v, r):
        """把状态推进到时刻 t（秒），用这段时间的车速 v [m/s] 和偏航角速度 r [rad/s]。"""
        if self.t is None:
            self.t = t
            return
        dt = t - self.t
        if dt <= 0.0:
            return
        self.hist.append((self.t, t, v, r))
        while self.hist and self.hist[0][1] < t - self.max_latency_s:
            self.hist.pop(0)
        self.t = t
        if not self.initialised:
            return
        dt = min(dt, 1.0)                       # 卡顿保护
        n = max(1, int(math.ceil(dt / 0.02)))   # 大步长拆成 ≤20 ms 的小步
        h = dt / n
        for _ in range(n):
            th = self.x[1]
            F = np.array([[1.0, -v * math.cos(th) * h, 0.0],
                          [0.0, 1.0, v * h],
                          [0.0, 0.0, 1.0]])
            self.x = self._step(self.x, v, r, h)
            self.P = F @ self.P @ F.T + self.Qd * h

    def _shift_to_now(self, z, t_meas):
        """测量是 t_meas 时刻的；用记录的 v、r 把它按模型推到 self.t。"""
        if self.t is None or t_meas >= self.t:
            return z
        t0 = max(t_meas, self.t - self.max_latency_s)
        for (ta, tb, v, r) in self.hist:
            if tb <= t0:
                continue
            dt = tb - max(ta, t0)
            if dt > 0:
                z = self._step(z, v, r, dt)
        return z

    def check(self, t_meas, e, th, k, quality=1.0):
        """只判断这帧能不能过门限，不改状态（路口里用来判断视觉是否恢复）。"""
        if not self.initialised:
            return True
        z = self._shift_to_now(np.array([e, th, k], dtype=float), t_meas)
        q = min(1.0, max(0.05, float(quality)))
        y = z - self.x
        if abs(y[0]) > self.max_jump_e or abs(y[1]) > self.max_jump_th:
            return False
        return float(y @ np.linalg.solve(self.P + self.R0 / (q * q), y)) <= self.gate

    def update(self, t_meas, e, th, k, quality=1.0):
        """融合一帧视觉测量。返回 True=采纳，False=当作野值丢掉。"""
        z = np.array([e, th, k], dtype=float)
        q = min(1.0, max(0.05, float(quality)))
        R = self.R0 / (q * q)
        if not self.initialised:
            self.x = self._shift_to_now(z, t_meas)
            self.P = R.copy()
            self.initialised = True
            self.n_reject = 0
            if self.t is None:
                self.t = t_meas
            return True

        z = self._shift_to_now(z, t_meas)
        y = z - self.x
        S = self.P + R
        d2 = float(y @ np.linalg.solve(S, y))
        if (d2 > self.gate or abs(y[0]) > self.max_jump_e
                or abs(y[1]) > self.max_jump_th):
            self.n_reject += 1
            self.rejected.append(z)
            self.rejected = self.rejected[-self.reject_reset:]
            if len(self.rejected) >= self.reject_reset:
                es = [m[0] for m in self.rejected]
                ts = [m[1] for m in self.rejected]
                if (max(es) - min(es) < self.relock_e_spread
                        and max(ts) - min(ts) < self.relock_th_spread):
                    self.x = z
                    self.P = R.copy()
                    self.n_reject = 0
                    self.rejected = []
                    return True
            return False
        self.n_reject = 0
        self.rejected = []
        K = self.P @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(3) - K) @ self.P
        return True
