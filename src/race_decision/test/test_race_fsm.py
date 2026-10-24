import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from race_decision.race_fsm import RaceFSM  # noqa: E402


def good(f, n=1):
    for _ in range(n):
        f.on_frame(is_lost=False, accepted=True)


def bad(f, n=1, lost=True):
    for _ in range(n):
        f.on_frame(is_lost=lost, accepted=False)


def run(f, seconds, v=0.3, fresh=True):
    out = None
    for _ in range(int(round(seconds / 0.02))):
        out = f.step(0.02, v, fresh)
    return out


def test_waits_for_vision_then_cruises():
    f = RaceFSM()
    assert run(f, 0.1, v=0.0, fresh=False)["state"] == "WAIT"
    good(f)
    out = run(f, 0.02)
    assert out["state"] == "CRUISE" and out["v_cap"] is None and out["use_vision"]


def test_vision_timeout_stops_and_recovers():
    f = RaceFSM()
    good(f)
    run(f, 0.1)
    out = run(f, 0.02, fresh=False)
    assert out["state"] == "TIMEOUT" and out["v_cap"] == 0.0
    good(f)
    assert run(f, 0.02)["state"] == "CRUISE"


def test_lost_hold_then_stop_after_distance():
    f = RaceFSM({"lost_enter_frames": 3, "lost_hold_m": 0.30, "lost_speed": 0.15})
    good(f)
    run(f, 0.1)
    bad(f, 2)
    assert run(f, 0.02)["state"] == "CRUISE"       # 滞回：2 帧不够
    bad(f, 1)
    out = run(f, 0.02)
    assert out["state"] == "LOST_HOLD" and out["v_cap"] == 0.15
    out = run(f, 1.5, v=0.15)                         # 走了 0.225 m
    assert out["state"] == "LOST_HOLD"
    out = run(f, 1.0, v=0.15)                         # 超过 0.30 m
    assert out["state"] == "LOST_STOP" and out["v_cap"] == 0.0


def test_lost_recovers_after_good_frames():
    f = RaceFSM({"lost_enter_frames": 3, "lost_exit_frames": 4})
    good(f)
    bad(f, 3)
    run(f, 0.02)
    good(f, 3)
    assert run(f, 0.02)["state"] == "LOST_HOLD"
    good(f, 1)
    assert run(f, 0.02)["state"] == "CRUISE"


def test_rejected_but_not_lost_frames_also_count_as_bad():
    f = RaceFSM({"lost_enter_frames": 3})
    good(f)
    bad(f, 3, lost=False)
    assert run(f, 0.02)["state"] == "LOST_HOLD"


def test_junction_hint_holds_straight_and_exits_after_distance():
    f = RaceFSM({"junction_min_m": 0.3, "junction_max_m": 0.8,
                 "junction_exit_frames": 3, "junction_speed": 0.2})
    good(f)
    run(f, 0.1)
    f.junction_hint()
    out = run(f, 0.02)
    assert out["state"] == "JUNCTION"
    assert out["force_straight"] and not out["use_vision"] and out["v_cap"] == 0.2
    good(f, 5)
    assert run(f, 0.5, v=0.2)["state"] == "JUNCTION"   # 0.1 m < junction_min_m
    run(f, 1.2, v=0.2)                                   # 累计 0.34 m
    good(f, 3)
    assert run(f, 0.02, v=0.2)["state"] == "CRUISE"


def test_junction_forced_exit_at_max_distance():
    f = RaceFSM({"junction_min_m": 0.3, "junction_max_m": 0.5})
    good(f)
    f.junction_hint()
    run(f, 0.02)
    bad(f, 50)
    assert run(f, 2.0, v=0.3)["state"] != "JUNCTION"


def test_detections_hook_is_accepted_and_ignored_for_now():
    f = RaceFSM()
    good(f)
    f.on_detections([object()])
    assert run(f, 0.02)["e_ref"] == 0.0


def test_lost_exit_tolerates_sporadic_garbage():
    # 路口：可信帧之间夹着野值，不要求「连续」可信，窗口里够数就恢复
    f = RaceFSM({"lost_enter_frames": 3, "lost_exit_frames": 3, "lost_exit_window": 6})
    good(f)
    bad(f, 3)
    run(f, 0.02)
    for _ in range(3):
        good(f)
        bad(f, lost=False)
    assert run(f, 0.02)["state"] == "CRUISE"


def test_default_lost_hold_covers_a_junction():
    assert RaceFSM().p["lost_hold_m"] >= 0.6
