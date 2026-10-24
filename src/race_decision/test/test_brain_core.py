import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from race_decision.brain_core import BrainCore  # noqa: E402


def feed(core, t0, seconds, e=0.0, th=0.0, k=0.0, lost=False, imu=True, width=0.45):
    t = t0
    out = None
    n = int(round(seconds / 0.02))
    for i in range(n):
        t += 0.02
        if i % 2 == 0:
            core.on_track(stamp=t - 0.05, now=t, e=e, th=th, k=k,
                          width=width, quality=0.9, is_lost=lost)
        if imu:
            core.on_imu(t, 0.0)
            core.on_speed(t, core.ctl.v_cmd)
        out = core.tick(t, 0.02)
    return t, out


def test_no_vision_means_stop():
    core = BrainCore()
    out = core.tick(1.0, 0.02)
    assert out["v"] == 0.0 and out["state"] == "WAIT"


def test_cruises_and_steers_back_to_center():
    core = BrainCore()
    t, out = feed(core, 0.0, 1.0, e=0.10)
    assert out["state"] == "CRUISE"
    assert out["v"] > 0.0
    assert out["steer"] < 0.0          # 车偏左 → 往右打


def test_track_timeout_stops_immediately():
    core = BrainCore({"track_timeout_s": 0.3})
    t, out = feed(core, 0.0, 1.0)
    assert out["v"] > 0
    out = core.tick(t + 0.5, 0.02)
    assert out["state"] == "TIMEOUT" and out["v"] == 0.0


def test_works_without_imu():
    core = BrainCore()
    t, out = feed(core, 0.0, 1.0, e=0.1, imu=False)
    assert out["state"] == "CRUISE" and out["steer"] < 0 and not out["imu_ok"]


def test_lost_vision_keeps_curving_along_last_curvature():
    core = BrainCore()
    t, out = feed(core, 0.0, 1.0, k=1.5)
    steer_curve = out["steer"]
    assert steer_curve > 0.05
    t, out = feed(core, t, 0.3, k=0.0, lost=True)
    assert out["state"] == "LOST_HOLD"
    assert out["steer"] > 0.03          # 不回正，沿最后的曲率继续拐


def test_junction_hint_ignores_vision_garbage():
    core = BrainCore()
    t, out = feed(core, 0.0, 1.0)
    core.junction_hint()
    t, out = feed(core, t, 0.3, e=0.5, th=0.8)
    assert out["state"] == "JUNCTION"
    assert abs(out["steer"]) < 0.02


def test_imu_with_wrong_sign_gets_disabled():
    # IMU 符号装反会让内环变正反馈：在线和舵角模型比对，发现反了就停用 IMU
    core = BrainCore()
    t = 0.0
    for i in range(400):
        t += 0.02
        if i % 2 == 0:
            core.on_track(stamp=t - 0.05, now=t, e=0.0, th=0.0, k=1.5,
                          width=0.45, quality=0.9, is_lost=False)
        wheel = core.ctl.delta * core.ctl.p["steer_ratio"]
        import math
        r_true = core.ctl.v_cmd * math.tan(wheel) / core.ctl.p["wheelbase_m"]
        core.on_imu(t, -r_true)            # 符号反了
        core.on_speed(t, core.ctl.v_cmd)
        out = core.tick(t, 0.02)
    assert not out["imu_ok"]
    assert core.imu_disabled


def test_imu_with_right_sign_stays_enabled():
    core = BrainCore()
    t = 0.0
    import math
    for i in range(400):
        t += 0.02
        if i % 2 == 0:
            core.on_track(stamp=t - 0.05, now=t, e=0.0, th=0.0, k=1.5,
                          width=0.45, quality=0.9, is_lost=False)
        wheel = core.ctl.delta * core.ctl.p["steer_ratio"]
        core.on_imu(t, core.ctl.v_cmd * math.tan(wheel) / core.ctl.p["wheelbase_m"])
        core.on_speed(t, core.ctl.v_cmd)
        out = core.tick(t, 0.02)
    assert out["imu_ok"] and not core.imu_disabled
