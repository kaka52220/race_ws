# -*- coding: utf-8 -*-
"""
sim_brain.py —— brain 闭环离线仿真（不需要 ROS / 小车）

    python3 tools/sim_brain.py                  # v1 vs v2 在几档速度下对比
    python3 tools/sim_brain.py --speed 0.35     # 只跑一档

模型（尽量贴近实车已知数据）：
    · 赛道：直道 / R0.6 左弯 / R0.55 S 弯 / 路口（0.5 m 内视觉一半是乱跳的野值）/ R0.6 掉头
    · 车：自行车模型，轴距 0.143；舵机 25° → 前轮 15.2°，再加 1.5° 零偏；舵机一阶滞后 50 ms
          速度一阶滞后 0.15 s，按 set_car_run 的 1% = 0.0134 m/s 量化
    · 视觉：30 fps，延迟 70 ms（stamp=拍照时刻），噪声 e 1.5 cm / θ 4° / κ 0.3，
            3% 单帧野值，2% 丢线
    · IMU：25 Hz，噪声 0.02 rad/s，零偏 0.01 rad/s
    · brain：50 Hz，输出晚一个周期生效
"""
import argparse
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from race_decision.brain_core import BrainCore  # noqa: E402

L = 0.143
HALF_LANE_OK = 0.12     # 车道净宽 0.40，车宽一半约 0.08 → |e| > 0.12 视为压线


def build_track(ds=0.01):
    segs = [(1.5, 0.0), (math.pi / 2 * 0.6, 1 / 0.6), (0.6, 0.0),
            (math.pi / 2 * 0.55, -1 / 0.55), (math.pi / 2 * 0.55, 1 / 0.55),
            (0.4, 0.0), ("J", 0.6), (0.6, 0.0),
            (math.pi * 0.6, 1 / 0.6), (1.2, 0.0)]
    pts = []          # (x, y, psi, kappa, s, junction)
    x = y = psi = s = 0.0
    for seg in segs:
        if seg[0] == "J":
            length, k, jn = seg[1], 0.0, True
        else:
            length, k = seg
            jn = False
        n = int(round(length / ds))
        for _ in range(n):
            pts.append((x, y, psi, k, s, jn))
            x += math.cos(psi) * ds
            y += math.sin(psi) * ds
            psi += k * ds
            s += ds
    return pts


class Car:
    def __init__(self, ratio=0.608, offset_deg=1.5):
        self.x = self.y = self.psi = 0.0
        self.y = 0.03                       # 起步偏左 3 cm
        self.v = 0.0
        self.w = 0.0
        self.ratio = ratio
        self.offset = math.radians(offset_deg)

    def step(self, v_cmd, servo, dt):
        w_target = servo * self.ratio + self.offset
        self.w += (w_target - self.w) * min(1.0, dt / 0.05)
        v_q = round(v_cmd / 0.0134) * 0.0134
        self.v += (v_q - self.v) * min(1.0, dt / 0.15)
        r = self.v * math.tan(self.w) / L
        self.x += self.v * math.cos(self.psi) * dt
        self.y += self.v * math.sin(self.psi) * dt
        self.psi += r * dt
        return r


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def lane_pose(track, car, idx):
    """真值：最近点给 e（压线判定用）；视觉测量 = 车前 0.22~0.62 m 中心线在车体系下的二次拟合，
    和 vision 的做法一致：y = −e + x·tanθ + κx²/2。"""
    best, bi = 1e9, idx
    for i in range(max(0, idx - 30), min(len(track), idx + 60)):
        d = (track[i][0] - car.x) ** 2 + (track[i][1] - car.y) ** 2
        if d < best:
            best, bi = d, i
    px, py, psi, _, _, jn = track[bi]
    e_true = (car.x - px) * -math.sin(psi) + (car.y - py) * math.cos(psi)
    c, s_ = math.cos(car.psi), math.sin(car.psi)
    xs, ys = [], []
    for j in range(bi, min(len(track), bi + 90), 3):
        dx, dy = track[j][0] - car.x, track[j][1] - car.y
        fx, fy = dx * c + dy * s_, -dx * s_ + dy * c
        if 0.22 <= fx <= 0.62:
            xs.append(fx)
            ys.append(fy)
    if len(xs) >= 5:
        import numpy as np
        cc = np.polyfit(xs, ys, 2)
        meas = (-cc[2], math.atan(cc[1]), 2 * cc[0])
    else:
        meas = (e_true, wrap(psi - car.psi), 0.0)
    return bi, e_true, meas, jn


class V1:
    """当前 brain_node（v1）的等效逻辑：只用 e 的 Stanley + 两道低通 + 定速。"""

    def __init__(self, speed):
        self.speed = speed
        self.meas = None
        self.t_track = None
        self.lost_cnt = self.ok_cnt = 0
        self.state = "WAIT"
        self.prev = 0.0
        self.f = 0.0
        self.ctl = type("C", (), {"v_cmd": 0.0})()

    def on_track(self, stamp, now, e, th, k, width, quality, is_lost):
        self.meas = (e, is_lost)
        self.t_track = now
        if is_lost:
            self.lost_cnt += 1
            self.ok_cnt = 0
        else:
            self.ok_cnt += 1
            self.lost_cnt = 0

    def on_imu(self, now, r):
        pass

    def on_speed(self, now, v):
        pass

    def junction_hint(self):
        pass

    def tick(self, now, dt):
        if self.meas is None:
            return {"v": 0.0, "steer": 0.0, "state": "WAIT"}
        if self.lost_cnt >= 3 or (self.state == "LOST" and self.ok_cnt < 8):
            self.state = "LOST"
            return {"v": 0.07, "steer": 0.0, "state": "LOST"}
        self.state = "CRUISING"
        e = max(-0.35, min(0.35, self.meas[0]))
        d = -math.atan(0.7 * e / max(self.speed, 0.5))
        d = max(-math.radians(25), min(math.radians(25), d))
        rate = math.radians(200) * dt
        d = max(self.prev - rate, min(self.prev + rate, d))
        self.prev = d
        a = dt / (0.08 + dt)
        self.f = a * d + (1 - a) * self.f
        return {"v": self.speed, "steer": self.f, "state": "CRUISING"}


def simulate(brain, seed=0, t_max=90.0, junction_hint=False, latency=0.07,
             outlier_p=0.03, lost_p=0.02, junction_garbage_p=0.5, noise=1.0,
             imu_bias=0.01, steer_offset_deg=1.5):
    rnd = random.Random(seed)
    track = build_track()
    car = Car(offset_deg=steer_offset_deg)
    dt = 0.02
    t = 0.0
    idx = 0
    pending = []                 # (deliver_time, stamp, meas)
    next_frame = 0.0
    next_imu = 0.0
    imu_val = 0.0
    cmd = (0.0, 0.0)
    r = 0.0
    errs = []
    over = 0
    hinted = False
    states = {}
    while t < t_max:
        t += dt
        r = car.step(cmd[0], cmd[1], dt)
        idx, e, (e_m, th, k), jn = lane_pose(track, car, idx)
        if idx >= len(track) - 70:
            break
        errs.append(abs(e))
        if abs(e) > HALF_LANE_OK:
            over += 1
        if abs(e) > 0.6:
            return {"crashed": True, "t": t, "max_e": max(errs), "over_s": over * dt,
                    "rms_e": math.sqrt(sum(x * x for x in errs) / len(errs)),
                    "states": states}
        # 视觉：拍照
        if t >= next_frame:
            next_frame += 1 / 30
            lost = rnd.random() < lost_p
            me = e_m + rnd.gauss(0, 0.015 * noise)
            mth = th + rnd.gauss(0, math.radians(4) * noise)
            mk = k + rnd.gauss(0, 0.3 * noise)
            q = 0.9
            if jn and rnd.random() < junction_garbage_p:
                me = rnd.choice([1, -1]) * rnd.uniform(0.2, 0.8)
                mth = rnd.uniform(-1.0, 1.0)
                mk = rnd.uniform(-2, 2)
                q = 0.65
            elif rnd.random() < outlier_p:
                me += rnd.choice([1, -1]) * rnd.uniform(0.3, 0.8)
                mth += rnd.uniform(-0.8, 0.8)
                q = 0.6
            pending.append((t + latency, t, (me, mth, mk, 0.45, q, lost)))
        while pending and pending[0][0] <= t:
            _, stamp, m = pending.pop(0)
            brain.on_track(stamp=stamp, now=t, e=m[0], th=m[1], k=m[2],
                           width=m[3], quality=m[4], is_lost=m[5])
        if t >= next_imu:
            next_imu += 0.04
            imu_val = r + imu_bias + rnd.gauss(0, 0.02 * noise)
            brain.on_speed(t, car.v + rnd.gauss(0, 0.005))
        brain.on_imu(t, imu_val)
        if junction_hint and jn and not hinted:
            brain.junction_hint()
            hinted = True
        out = brain.tick(t, dt)
        states[out["state"]] = states.get(out["state"], 0) + 1
        cmd = (out["v"], out["steer"])
    return {"crashed": False, "t": t, "max_e": max(errs), "over_s": over * dt,
            "rms_e": math.sqrt(sum(x * x for x in errs) / len(errs)), "states": states}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=None)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    speeds = [a.speed] if a.speed else [0.21, 0.30, 0.40]
    for v in speeds:
        for name, mk in (("v1", lambda: V1(v)),
                         ("v2", lambda: BrainCore({"max_speed": v}))):
            rs = [simulate(mk(), seed=s) for s in range(a.seeds)]
            crashed = sum(r["crashed"] for r in rs)
            ok = [r for r in rs if not r["crashed"]]
            if ok:
                print(f"{name} v_max={v:.2f}: 冲出 {crashed}/{len(rs)}  "
                      f"用时 {sum(r['t'] for r in ok)/len(ok):5.1f}s  "
                      f"最大|e| {max(r['max_e'] for r in ok)*100:4.1f}cm  "
                      f"RMS {sum(r['rms_e'] for r in ok)/len(ok)*100:4.1f}cm  "
                      f"压线 {sum(r['over_s'] for r in ok)/len(ok):4.2f}s")
            else:
                print(f"{name} v_max={v:.2f}: 冲出 {crashed}/{len(rs)}")


if __name__ == "__main__":
    main()
