# -*- coding: utf-8 -*-
"""
brain_core.py —— brain 的全部逻辑（不依赖 ROS）

brain_node.py 只负责把 ROS 话题喂进来、把结果发出去；
离线仿真 tools/sim_brain.py 和单元测试直接用这个类，保证「测的就是跑的」。

    on_track(stamp, now, ...)   一帧 /track（stamp = 图像时间，用来做延迟补偿）
    on_imu(now, r)              偏航角速度 rad/s，左正
    on_speed(now, v)            实测车速 m/s
    tick(now, dt) -> dict       每个控制周期调用一次，返回 v / steer / state / 诊断量
"""
import math

from .controller import Controller, DEFAULTS as CTL_DEFAULTS
from .lane_estimator import LaneEstimator
from .race_fsm import RaceFSM, DEFAULTS as FSM_DEFAULTS

EST_KEYS = ("sigma_e", "sigma_th", "sigma_k", "q_e", "q_th", "q_k", "gate",
            "reject_reset", "relock_e_spread", "relock_th_spread", "max_jump_e",
            "max_jump_th", "max_latency_s")

CORE_DEFAULTS = {
    "track_timeout_s": 0.5,
    "imu_timeout_s": 0.2,
    "speed_timeout_s": 0.3,
    "use_imu": True,
    "use_wheel_speed": True,
    "imu_check_n": 40,          # IMU 符号自检：转弯时累计这么多个样本评一次
    "imu_check_min_ratio": 0.3,  # 实测 r 与舵角模型 r 的回归系数低于它（反号/没信号）→ 停用 IMU
}


def all_defaults():
    d = {}
    d.update(CTL_DEFAULTS)
    d.update(FSM_DEFAULTS)
    d.update(CORE_DEFAULTS)
    return d


class BrainCore:
    def __init__(self, params=None):
        p = all_defaults()
        if params:
            p.update(params)
        self.p = p
        self.ctl = Controller({k: p[k] for k in CTL_DEFAULTS})
        self.fsm = RaceFSM({k: p[k] for k in FSM_DEFAULTS})
        self.est = LaneEstimator(**{k: p[k] for k in EST_KEYS if k in p})
        self.t_track = None
        self.t_imu = None
        self.r_imu = 0.0
        self.t_speed = None
        self.v_meas = 0.0
        self.last_meas = None
        self.n_frames = 0
        self.n_rejected = 0
        self.imu_disabled = False
        self.imu_ratio = None       # 最近一次自检：r_imu ≈ ratio × r_模型（≈1 说明 IMU 和舵机标定都对）
        self._chk = [0, 0.0, 0.0]

    def set_param(self, name, value):
        """运行时改参数（ros2 param set 走这里）。"""
        self.p[name] = value
        if name in self.ctl.p:
            self.ctl.p[name] = value
        if name in self.fsm.p:
            self.fsm.p[name] = value

    # ── 输入 ──
    def on_imu(self, now, r):
        self.t_imu = now
        self.r_imu = float(r)

    def on_speed(self, now, v):
        self.t_speed = now
        self.v_meas = float(v)

    def junction_hint(self):
        self.fsm.junction_hint()

    def on_detections(self, items):
        self.fsm.on_detections(items)

    def _inputs(self, now):
        imu_ok = (self.p["use_imu"] and not self.imu_disabled and self.t_imu is not None
                  and now - self.t_imu <= self.p["imu_timeout_s"])
        spd_ok = (self.p["use_wheel_speed"] and self.t_speed is not None
                  and now - self.t_speed <= self.p["speed_timeout_s"])
        v = self.v_meas if spd_ok else self.ctl.v_cmd
        if imu_ok:
            r = self.r_imu
        else:
            # 没 IMU：用当前舵角按自行车模型估计 r（不如实测准，内环关掉）
            wheel = self.ctl.delta * self.ctl.p["steer_ratio"]
            r = v * math.tan(wheel) / self.ctl.p["wheelbase_m"]
        return v, r, imu_ok

    def on_track(self, stamp, now, e, th, k, width, quality, is_lost):
        self.t_track = now
        self.n_frames += 1
        self.last_meas = (e, th, k, width, quality, is_lost)
        v, r, _ = self._inputs(now)
        self.est.predict(now, v, r)
        if is_lost:
            accepted = False
        elif self.fsm.state == "JUNCTION":
            accepted = self.est.check(stamp, e, th, k, quality)   # 只看，不融合
        else:
            accepted = self.est.update(stamp, e, th, k, quality)
        if not accepted and not is_lost:
            self.n_rejected += 1
        self.fsm.on_frame(is_lost=is_lost, accepted=accepted)

    def _check_imu(self, v, imu_ok):
        """在线自检：IMU 的 r 应该和「舵角 + 自行车模型」算出来的 r 同号、大小相近。
        符号装反时内环会变成正反馈（越打越歪），所以发现反了立即停用并报警。"""
        if not imu_ok or v < 0.1:
            return
        wheel = self.ctl.delta * self.ctl.p["steer_ratio"]
        r_model = v * math.tan(wheel) / self.ctl.p["wheelbase_m"]
        if abs(r_model) < 0.15:
            return
        c = self._chk
        c[0] += 1
        c[1] += self.r_imu * r_model
        c[2] += r_model * r_model
        if c[0] >= self.p["imu_check_n"]:
            self.imu_ratio = c[1] / c[2]
            self._chk = [0, 0.0, 0.0]
            if self.imu_ratio < self.p["imu_check_min_ratio"]:
                self.imu_disabled = True

    # ── 每个控制周期 ──
    def tick(self, now, dt):
        v, r, imu_ok = self._inputs(now)
        self._check_imu(v, imu_ok)
        if self.imu_disabled and imu_ok:
            v, r, imu_ok = self._inputs(now)
        fresh = (self.t_track is not None
                 and now - self.t_track <= self.p["track_timeout_s"])
        self.est.predict(now, v, r)
        cmd = self.fsm.step(dt, v, fresh)
        if cmd["force_straight"] and self.est.initialised:
            self.est.force_straight()

        out = {"state": cmd["state"], "imu_ok": imu_ok, "v_meas": v, "r": r}
        if not self.est.initialised or cmd["v_cap"] == 0.0:
            self.ctl.v_cmd = 0.0
            out.update(v=0.0, steer=self.ctl.delta, e=0.0, th=0.0, k=0.0, info={})
            if self.est.initialised:
                e, th, k = self.est.state()
                out.update(e=e, th=th, k=k)
            return out

        e, th, k = self.est.state()
        v_cmd, steer, info = self.ctl.step(
            e, th, k, v_meas=v, r_meas=(r if imu_ok else None), dt=dt,
            e_ref=cmd["e_ref"], v_cap=cmd["v_cap"])
        out.update(v=v_cmd, steer=steer, e=e, th=th, k=k, info=info)
        return out
