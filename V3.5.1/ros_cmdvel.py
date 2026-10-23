#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ros_cmdvel.py —— 把 ROS 的 /cmd_vel 转成 Rosmaster 扩展板(YB-ERF01)串口协议

支持 ROS1(rospy) 和 ROS2(rclpy)，自动识别。
协议已在本车实测（115200 8N1，帧头 FF FC，校验=长度+功能码+参数 &0xFF）。

用法
----
ROS1:
    python3 ros_cmdvel.py _port:=/dev/ttyUSB0 _wheelbase:=0.235 _max_speed:=0.3

ROS2:
    python3 ros_cmdvel.py --ros-args -p port:=/dev/ttyUSB0 -p wheelbase:=0.235 -p max_speed:=0.3

参数
----
port        串口，默认 /dev/ttyUSB0（Windows 上写 COM7）
wheelbase   轴距，米，默认 0.235   ← 明天量出来后填这里或改默认值
max_speed   前进限速，m/s，默认 0.3
rate        下发频率，Hz，默认 20
timeout     多久没收到 cmd_vel 就停车，秒，默认 0.4
steer       auto = 自己按轴距算转角发 Vy（推荐）
            firmware = 发 Vz 让固件用它自己的轴距算

安全
----
* 固件本身没有失联停车，这里做了看门狗：超过 timeout 没收到 cmd_vel 就发停车帧
* Ctrl+C 退出时自动停车
"""

import math
import sys
import threading
import time

import serial


# ----------------------------------------------------------------------------
# 协议层（与 rosmaster_test.py 完全一致，已实测）
# ----------------------------------------------------------------------------
F_MOTION = 0x12


def s16(v):
    v = int(v)
    return bytes([v & 0xFF, (v >> 8) & 0xFF])


def build(func, payload=b""):
    n = len(payload) + 3
    f = bytes([0xFF, 0xFC, n, func]) + bytes(payload)
    return f + bytes([sum(f[2:]) & 0xFF])


class Board:
    def __init__(self, port, baud=115200, wheelbase=0.235, max_speed=0.3,
                 steer="auto", timeout=0.4):
        self.ser = serial.Serial(port, baud, timeout=0.02)
        self.lock = threading.Lock()
        self.wheelbase = float(wheelbase)
        self.max_speed = float(max_speed)
        self.steer = steer
        self.timeout = float(timeout)
        self.last_cmd = 0.0
        self.target = (0.0, 0.0)
        self._stopped = False

    def _send(self, func, payload=b""):
        with self.lock:
            self.ser.write(build(func, payload))

    def stop(self):
        if self._stopped:
            return
        self._send(F_MOTION, b"\x00" + s16(0) + s16(0) + s16(0))
        self._stopped = True

    def cmd_vel(self, v, w):
        """ROS 语义：v 前进 m/s，w 逆时针(左转) rad/s"""
        self.target = (float(v), float(w))
        self.last_cmd = time.time()
        self._stopped = False

    def _to_frame(self, v, w):
        if abs(v) < 0.02:
            # 阿克曼不能原地转，速度太小直接停
            return 0, 0, 0
        v = max(-self.max_speed, min(self.max_speed, v))
        vx = int(v * 1000)
        if self.steer == "firmware":
            return vx, 0, int(w * 1000)
        # auto：自己按轴距算前轮转角，Vy 单位是度，正 = 左转
        delta = math.atan(w * self.wheelbase / v)
        vy = max(-45.0, min(45.0, math.degrees(delta)))
        return vx, int(round(vy)), 0

    def tick(self):
        if time.time() - self.last_cmd > self.timeout:
            self.stop()
            return
        vx, vy, vz = self._to_frame(*self.target)
        self._send(F_MOTION, b"\x00" + s16(vx) + s16(vy) + s16(vz))

    def close(self):
        try:
            self.stop()
            time.sleep(0.05)
            self.ser.close()
        except Exception:
            pass


# ----------------------------------------------------------------------------
# ROS1
# ----------------------------------------------------------------------------
def run_ros1(cfg):
    import rospy
    from geometry_msgs.msg import Twist

    rospy.init_node("rosmaster_cmdvel", anonymous=False)
    port = rospy.get_param("~port", cfg["port"])
    wb = rospy.get_param("~wheelbase", cfg["wheelbase"])
    vmax = rospy.get_param("~max_speed", cfg["max_speed"])
    rate_hz = rospy.get_param("~rate", cfg["rate"])
    to = rospy.get_param("~timeout", cfg["timeout"])
    steer = rospy.get_param("~steer", cfg["steer"])

    board = Board(port, wheelbase=wb, max_speed=vmax, steer=steer, timeout=to)
    rospy.loginfo("rosmaster: %s  轴距=%.3fm  限速=%.2fm/s  steer=%s" % (port, wb, vmax, steer))

    rospy.Subscriber("cmd_vel", Twist,
                     lambda m: board.cmd_vel(m.linear.x, m.angular.z), queue_size=1)

    r = rospy.Rate(rate_hz)
    try:
        while not rospy.is_shutdown():
            board.tick()
            r.sleep()
    finally:
        board.close()


# ----------------------------------------------------------------------------
# ROS2
# ----------------------------------------------------------------------------
def run_ros2(cfg):
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist

    class CmdVelNode(Node):
        def __init__(self):
            super().__init__("rosmaster_cmdvel")
            self.declare_parameter("port", cfg["port"])
            self.declare_parameter("wheelbase", cfg["wheelbase"])
            self.declare_parameter("max_speed", cfg["max_speed"])
            self.declare_parameter("rate", cfg["rate"])
            self.declare_parameter("timeout", cfg["timeout"])
            self.declare_parameter("steer", cfg["steer"])
            p = self.get_parameter
            self.board = Board(p("port").value, wheelbase=p("wheelbase").value,
                               max_speed=p("max_speed").value,
                               steer=p("steer").value, timeout=p("timeout").value)
            self.get_logger().info(
                "rosmaster: %s  轴距=%.3fm  限速=%.2fm/s" % (
                    p("port").value, p("wheelbase").value, p("max_speed").value))
            self.create_subscription(
                Twist, "cmd_vel",
                lambda m: self.board.cmd_vel(m.linear.x, m.angular.z), 1)
            self.create_timer(1.0 / float(p("rate").value), self.board.tick)

        def destroy_node(self):
            self.board.close()
            super().destroy_node()

    rclpy.init()
    node = CmdVelNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


# ----------------------------------------------------------------------------
def main():
    cfg = {"port": "/dev/ttyUSB0", "wheelbase": 0.235, "max_speed": 0.3,
           "rate": 20.0, "timeout": 0.4, "steer": "auto"}

    # 允许命令行直接覆盖（非 ROS 参数写法，临时调试用）
    for a in sys.argv[1:]:
        if a.startswith("port:="):
            cfg["port"] = a.split(":=", 1)[1]
        elif a.startswith("wheelbase:="):
            cfg["wheelbase"] = float(a.split(":=", 1)[1])

    try:
        import rclpy  # noqa: F401
        return run_ros2(cfg)
    except ImportError:
        pass
    try:
        import rospy  # noqa: F401
        return run_ros1(cfg)
    except ImportError:
        pass

    print("没有检测到 ROS 环境（rclpy / rospy 都不可用）。")
    print("如果你只想测试串口和电机，用 rosmaster_test.py 就够了。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
