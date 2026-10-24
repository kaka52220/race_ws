import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from race_decision.lane_estimator import LaneEstimator  # noqa: E402


def make():
    return LaneEstimator()


def test_first_measurement_initialises_state():
    est = make()
    assert not est.initialised
    ok = est.update(0.0, 0.05, 0.1, 0.5, quality=0.9)
    assert ok and est.initialised
    e, th, k = est.state()
    assert abs(e - 0.05) < 1e-9 and abs(th - 0.1) < 1e-9 and abs(k - 0.5) < 1e-9


def test_predict_heading_moves_lateral_error():
    # 车道相对车头往左偏 θ>0 → 车相对车道往右漂 → e 减小
    est = make()
    est.update(0.0, 0.0, 0.2, 0.0, quality=1.0)
    for i in range(50):
        est.predict(0.02 * (i + 1), v=0.3, r=0.0)
    e, th, _ = est.state()
    assert e < -0.05
    assert abs(e - (-0.3 * math.sin(0.2) * 1.0)) < 0.01
    assert abs(th - 0.2) < 1e-6


def test_predict_curvature_and_yaw_rate():
    # 车道曲率 κ，车不转 → θ 以 v·κ 增长；车按 v·κ 转 → θ 不变
    est = make()
    est.update(0.0, 0.0, 0.0, 1.0, quality=1.0)
    est.predict(1.0, v=0.3, r=0.0)
    assert abs(est.state()[1] - 0.3) < 0.02
    est2 = make()
    est2.update(0.0, 0.0, 0.0, 1.0, quality=1.0)
    est2.predict(1.0, v=0.3, r=0.3)
    assert abs(est2.state()[1]) < 1e-6


def test_outlier_is_rejected_and_good_measurement_accepted():
    est = make()
    t = 0.0
    for _ in range(20):
        t += 0.033
        est.predict(t, v=0.3, r=0.0)
        assert est.update(t, 0.02, 0.0, 0.0, quality=0.9)
    t += 0.033
    est.predict(t, v=0.3, r=0.0)
    assert not est.update(t, 0.62, 0.9, 0.0, quality=0.6)   # e 跳 60 cm
    assert abs(est.state()[0] - 0.02) < 0.02
    t += 0.033
    est.predict(t, v=0.3, r=0.0)
    assert est.update(t, 0.03, 0.01, 0.0, quality=0.9)


def test_persistent_shift_is_eventually_accepted():
    # 真的偏了（不是单帧野值）：连续拒绝后要能重新锁上
    est = make()
    t = 0.0
    for _ in range(20):
        t += 0.033
        est.predict(t, v=0.0, r=0.0)
        est.update(t, 0.0, 0.0, 0.0, quality=0.9)
    acc = []
    for _ in range(30):
        t += 0.033
        est.predict(t, v=0.0, r=0.0)
        acc.append(est.update(t, 0.25, 0.0, 0.0, quality=0.9))
    assert acc[-1]
    assert abs(est.state()[0] - 0.25) < 0.05


def test_latency_compensation_shifts_old_measurement_to_now():
    # 车以 θ=0.2 行驶，视觉在 0.1 s 前拍的图说 e=0 → 现在应 ≈ −v·sinθ·0.1
    est = make()
    est.update(0.0, 0.0, 0.2, 0.0, quality=1.0)
    t = 0.0
    for _ in range(10):
        t += 0.01
        est.predict(t, v=0.5, r=0.0)
    est.update(t - 0.1, 0.0, 0.2, 0.0, quality=1.0, )
    expected = -0.5 * math.sin(0.2) * 0.1
    assert abs(est.state()[0] - expected) < 0.004


def test_low_quality_measurement_moves_state_less():
    a, b = make(), make()
    for est in (a, b):
        est.update(0.0, 0.0, 0.0, 0.0, quality=1.0)
        est.predict(0.05, v=0.3, r=0.0)
    a.update(0.05, 0.05, 0.0, 0.0, quality=1.0)
    b.update(0.05, 0.05, 0.0, 0.0, quality=0.2)
    assert a.state()[0] > b.state()[0] > 0.0


def test_uncertainty_grows_without_measurements():
    est = make()
    est.update(0.0, 0.0, 0.0, 0.0, quality=1.0)
    s0 = est.sigma_e()
    est.predict(2.0, v=0.3, r=0.0)
    assert est.sigma_e() > s0


def test_force_straight_zeroes_curvature():
    est = make()
    est.update(0.0, 0.0, 0.0, 1.5, quality=1.0)
    est.force_straight()
    assert est.state()[2] == 0.0


def test_jumping_garbage_never_relocks():
    # 路口虚线：每帧乱跳的测量，再多也不能把估计拽过去
    import random
    random.seed(1)
    est = make()
    t = 0.0
    for _ in range(20):
        t += 0.033
        est.predict(t, v=0.3, r=0.0)
        est.update(t, 0.0, 0.0, 0.0, quality=0.9)
    for i in range(40):
        t += 0.033
        est.predict(t, v=0.3, r=0.0)
        bad = random.choice([0.45, -0.5, 0.6, -0.35])
        est.update(t, bad, random.uniform(-0.8, 0.8), 0.0, quality=0.7)
    assert abs(est.state()[0]) < 0.1


def test_check_does_not_modify_state():
    est = make()
    est.update(0.0, 0.0, 0.0, 0.0, quality=1.0)
    x0 = est.state()
    assert est.check(0.0, 0.01, 0.0, 0.0, quality=1.0)
    assert not est.check(0.0, 0.6, 0.0, 0.0, quality=1.0)
    assert est.state() == x0


def test_big_jump_rejected_even_when_uncertain():
    # 推算了很久（P 很大）之后，门限会变松；绝对跳变上限兜底，防止一帧野值被吃进去
    est = make()
    est.update(0.0, 0.0, 0.0, 0.0, quality=1.0)
    est.predict(3.0, v=0.3, r=0.0)
    assert not est.update(3.0, 0.40, 0.0, 0.0, quality=0.9)
    assert not est.update(3.0, 0.0, 0.9, 0.0, quality=0.9)
