import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from race_decision.controller import Controller  # noqa: E402


def ctl(**kw):
    p = dict(steer_rate_deg_per_s=1e6)   # 测试里默认关掉变化率限制
    p.update(kw)
    c = Controller(p)
    c.v_cmd = c.p["max_speed"]           # 已经在巡航速度
    return c


def steer(c, e=0.0, th=0.0, k=0.0, r=None, v=None, e_ref=0.0, n=1):
    v = c.v_cmd if v is None else v
    out = None
    for _ in range(n):
        out = c.step(e, th, k, v_meas=v, r_meas=r, dt=0.02, e_ref=e_ref)
    return out


def test_centered_straight_goes_straight():
    v, d, _ = steer(ctl())
    assert abs(d) < 1e-9


def test_car_left_of_center_steers_right():
    _, d, _ = steer(ctl(), e=0.10)
    assert d < -0.02


def test_heading_error_steers_toward_lane():
    _, d, _ = steer(ctl(), th=0.2)
    assert d > 0.05


def test_curvature_feedforward_matches_geometry():
    # 在车道中心、方向对齐、车道左弯 κ → 纯跟踪应给出 κ_cmd = κ
    c = ctl(kr=0.0)
    _, d, info = steer(c, k=1.0)
    assert abs(info["kappa_cmd"] - 1.0) < 1e-6
    wheel = math.atan(c.p["wheelbase_m"] * 1.0)
    assert abs(d - wheel / c.p["steer_ratio"]) < 1e-6


def test_yaw_rate_loop_adds_steer_when_turning_too_little():
    a, b = ctl(), ctl()
    _, d_ok, info = steer(a, k=1.0, r=a.v_cmd * 1.0)
    _, d_lag, _ = steer(b, k=1.0, r=0.0)
    assert d_lag > d_ok


def test_yaw_loop_disabled_without_imu():
    a, b = ctl(), ctl(kr=0.0)
    _, d1, _ = steer(a, k=1.0, r=None)
    _, d2, _ = steer(b, k=1.0, r=None)
    assert abs(d1 - d2) < 1e-9


def test_steer_is_clamped():
    c = ctl()
    _, d, _ = steer(c, e=-0.3, th=1.0)
    assert d <= math.radians(c.p["steer_max_deg"]) + 1e-9


def test_steer_rate_limited():
    c = Controller({"steer_rate_deg_per_s": 100.0})
    c.v_cmd = 0.2
    _, d, _ = c.step(0.0, 1.0, 0.0, v_meas=0.2, r_meas=None, dt=0.02)
    assert d <= math.radians(100.0) * 0.02 + 1e-9


def test_speed_slows_in_curves_and_recovers_on_straights():
    c = ctl(max_speed=0.4, min_speed=0.15, a_lat=0.15)
    v_curve, _, _ = steer(c, k=2.0, r=0.0, n=200)
    assert v_curve < 0.3
    v_str, _, _ = steer(c, k=0.0, r=0.0, n=200)
    assert abs(v_str - 0.4) < 1e-6


def test_acceleration_is_ramped():
    c = Controller({"max_speed": 0.4, "accel": 0.5})
    c.v_cmd = 0.0
    v, _, _ = c.step(0.0, 0.0, 0.0, v_meas=0.0, r_meas=None, dt=0.02)
    assert abs(v - 0.01) < 1e-9


def test_speed_cap_from_state_machine():
    c = ctl(max_speed=0.4)
    v, _, _ = c.step(0.0, 0.0, 0.0, v_meas=0.4, r_meas=None, dt=0.02, v_cap=0.1)
    assert v < 0.4


def test_lateral_reference_offset():
    _, d, _ = steer(ctl(), e=0.2, e_ref=0.2)
    assert abs(d) < 1e-9
