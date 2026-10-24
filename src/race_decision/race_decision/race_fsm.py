# -*- coding: utf-8 -*-
"""
race_fsm.py —— 任务二状态机（纯 Python，不依赖 ROS）

状态：
    WAIT       还没收到可信视觉，原地等
    CRUISE     正常巡线
    LOST_HOLD  视觉连续不可信：估计器用 IMU+车速推算继续走，降速；走够 lost_hold_m 还没找回 → LOST_STOP
    LOST_STOP  停车等视觉恢复（盲开冲出赛道比停一下代价大）
    JUNCTION   路口：忽略视觉、曲率置 0、按估计器直行；走过 junction_min_m 且视觉连续可信 → CRUISE
    TIMEOUT    /track 超时（视觉节点挂了），立刻停
    （预留）CROSSWALK_STOP / AVOID：等 YOLO 的 /detections 接进来再实现

每帧视觉调用 on_frame()，每个控制周期调用 step()。
step() 返回给控制器的指令：
    v_cap          限速（None=不限，0=停）
    use_vision     这一帧要不要把视觉喂给估计器
    force_straight 估计器曲率清零（路口）
    e_ref          横向参考偏移（避障时平移车道中心，左正；现在恒 0）
"""

DEFAULTS = {
    "lost_enter_frames": 4,      # 连续多少帧不可信进 LOST_HOLD（30 fps ≈ 0.13 s）
    "lost_exit_frames": 4,       # 最近 lost_exit_window 帧里有这么多帧可信就退出（不要求连续：
    "lost_exit_window": 6,       # 路口里可信帧之间常夹着野值）
    "lost_hold_m": 0.70,         # 推算最多走多远（要比一个路口长，否则会停在路口里）
    "lost_speed": 0.15,          # 推算时的限速
    "junction_min_m": 0.45,      # 路口至少直行多远（≈ 一个路口宽）
    "junction_max_m": 0.90,      # 最多直行多远，超过强制退出
    "junction_exit_frames": 4,
    "junction_speed": 0.20,
}


class RaceFSM:
    def __init__(self, params=None):
        self.p = dict(DEFAULTS)
        if params:
            self.p.update(params)
        self.state = "WAIT"
        self.bad_cnt = 0
        self.good_cnt = 0
        self.dist = 0.0              # 当前状态下走过的距离
        self.ever_good = False
        self.recent = []             # 最近几帧是否可信
        self._junction_req = False
        self.detections = []

    # ── 输入 ──
    def on_frame(self, is_lost, accepted):
        """一帧视觉到了。accepted = 估计器采纳了它（没被当野值）。"""
        ok = (not is_lost) and accepted
        self.recent.append(ok)
        self.recent = self.recent[-int(self.p["lost_exit_window"]):]
        if ok:
            self.good_cnt += 1
            self.bad_cnt = 0
            self.ever_good = True
        else:
            self.bad_cnt += 1
            self.good_cnt = 0

    def junction_hint(self):
        """有人（视觉几何 / YOLO 箭头标志）说前方是路口。"""
        self._junction_req = True

    def on_detections(self, items):
        """/detections 预留口：斑马线停车、挡板避障、计圈以后在这里接。"""
        self.detections = list(items)

    # ── 每个控制周期 ──
    def _go(self, s):
        if s != self.state:
            self.state = s
            self.dist = 0.0

    def step(self, dt, v, track_fresh):
        p = self.p
        self.dist += max(v, 0.0) * dt

        if not track_fresh:
            if self.state != "WAIT":
                self._go("TIMEOUT")
            return self._out()

        if self.state == "WAIT":
            if not self.ever_good:
                return self._out()
            self._go("CRUISE")          # 见过可信帧就发车，同一周期继续往下判
        elif self.state == "TIMEOUT":
            self._go("CRUISE" if self.good_cnt >= 1 else "LOST_HOLD")

        if self._junction_req:
            self._junction_req = False
            if self.state == "CRUISE":
                self._go("JUNCTION")
                self.good_cnt = 0
                return self._out()

        if self.state == "JUNCTION":
            if self.dist >= p["junction_max_m"]:
                self._go("CRUISE" if self.good_cnt >= 1 else "LOST_HOLD")
            elif (self.dist >= p["junction_min_m"]
                  and self.good_cnt >= p["junction_exit_frames"]):
                self._go("CRUISE")
            return self._out()

        if self.state == "CRUISE":
            if self.bad_cnt >= p["lost_enter_frames"]:
                self._go("LOST_HOLD")
        elif self.state in ("LOST_HOLD", "LOST_STOP"):
            if sum(self.recent) >= p["lost_exit_frames"]:
                self._go("CRUISE")
            elif self.state == "LOST_HOLD" and self.dist > p["lost_hold_m"]:
                self._go("LOST_STOP")
        return self._out()

    def _out(self):
        s = self.state
        v_cap = None
        if s in ("WAIT", "TIMEOUT", "LOST_STOP"):
            v_cap = 0.0
        elif s == "LOST_HOLD":
            v_cap = self.p["lost_speed"]
        elif s == "JUNCTION":
            v_cap = self.p["junction_speed"]
        return {
            "state": s,
            "v_cap": v_cap,
            "use_vision": s != "JUNCTION",
            "force_straight": s == "JUNCTION",
            "e_ref": 0.0,
        }
