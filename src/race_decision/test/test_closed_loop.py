"""闭环仿真回归：v2 在 0.35 m/s 下不冲出、不压线，路口全是野值也能过。"""
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

import sim_brain as S  # noqa: E402
from race_decision.brain_core import BrainCore  # noqa: E402


def _runs(n=3, **kw):
    return [S.simulate(BrainCore({"max_speed": 0.35}), seed=s, **kw) for s in range(n)]


def test_v2_stays_in_lane_at_035():
    for r in _runs():
        assert not r["crashed"]
        assert r["max_e"] < 0.12
        assert r["over_s"] == 0.0


def test_v2_crosses_garbage_junction():
    for r in _runs(junction_garbage_p=1.0):
        assert not r["crashed"]
        assert r["max_e"] < 0.12


def test_v2_without_imu_still_completes():
    for s in range(2):
        r = S.simulate(BrainCore({"max_speed": 0.30, "use_imu": False}), seed=s)
        assert not r["crashed"] and r["max_e"] < 0.13
