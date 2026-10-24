"""brain_node 冒烟测试：用假的 rclpy / 消息模块，验证节点能起来、话题回调和定时器跑得通。
（真 ROS 环境里的验证在车上做：colcon build + ros2 launch。）"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Msg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Stamp:
    def __init__(self, t):
        self.sec = int(t)
        self.nanosec = int((t - int(t)) * 1e9)


class _Clock:
    t = 100.0

    def now(self):
        return _Obj(nanoseconds=int(_Clock.t * 1e9))


class _Pub:
    def __init__(self):
        self.msgs = []

    def publish(self, m):
        self.msgs.append(m)


class _Param:
    def __init__(self, name, value):
        self.name, self.value = name, value


class _Node:
    def __init__(self, name):
        self._params = {}
        self.subs = {}
        self.pubs = {}
        self.timer_cb = None
        self.param_cb = None

    def declare_parameter(self, n, v):
        self._params[n] = _Param(n, v)

    def get_parameter(self, n):
        return self._params[n]

    def add_on_set_parameters_callback(self, cb):
        self.param_cb = cb

    def create_publisher(self, typ, topic, q):
        self.pubs[topic] = _Pub()
        return self.pubs[topic]

    def create_subscription(self, typ, topic, cb, q):
        self.subs[topic] = cb

    def create_timer(self, period, cb):
        self.timer_cb = cb

    def get_clock(self):
        return _Clock()

    def get_logger(self):
        return _Obj(info=lambda *a, **k: None, warn=lambda *a, **k: None,
                    error=lambda *a, **k: None)


def _install_fakes():
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m
    mod("rclpy", init=lambda *a, **k: None, spin=lambda n: None, ok=lambda: True,
        shutdown=lambda: None)
    mod("rclpy.node", Node=_Node)
    mod("rcl_interfaces")
    mod("rcl_interfaces.msg", SetParametersResult=lambda successful: _Obj(successful=successful))
    mod("geometry_msgs")
    mod("geometry_msgs.msg", Twist=lambda: _Obj(linear=_Obj(x=0.0), angular=_Obj(z=0.0)),
        TwistStamped=_Msg)
    mod("sensor_msgs")
    mod("sensor_msgs.msg", Imu=_Msg)
    mod("std_msgs")
    mod("std_msgs.msg", Empty=_Msg, String=_Msg)
    mod("custom_msgs")
    mod("custom_msgs.msg", TrackInfo=_Msg, Detections=_Msg)


def test_node_runs_end_to_end():
    _install_fakes()
    sys.modules.pop("race_decision.brain_node", None)
    from race_decision import brain_node
    n = brain_node.BrainNode()
    for i in range(100):
        _Clock.t += 0.02
        if i % 2 == 0:
            n.subs["/track"](_Msg(header=_Obj(stamp=_Stamp(_Clock.t - 0.06)),
                                  lateral_error=0.08, heading_error=0.0, curvature=0.0,
                                  lane_width=0.45, quality=0.9, is_lost=False,
                                  n_points=1000, reject_reason=""))
        n.subs["/imu"](_Msg(angular_velocity=_Obj(z=0.0)))
        n.subs["/wheel_speed"](_Msg(twist=_Obj(linear=_Obj(x=n.core.ctl.v_cmd))))
        n.timer_cb()
    last = n.pubs["/cmd_vel"].msgs[-1]
    assert last.linear.x > 0.1
    assert last.angular.z < 0.0                 # 偏左 → 右打
    assert n.pubs["/brain/state"].msgs[-1].data == "CRUISE"


def test_speed_scale_param_applies():
    _install_fakes()
    sys.modules.pop("race_decision.brain_node", None)
    from race_decision import brain_node
    n = brain_node.BrainNode()
    n.param_cb([_Param("speed_scale", 0.5)])
    assert abs(n.core.ctl.p["max_speed"] - 0.5 * n.get_parameter("max_speed").value) < 1e-9
