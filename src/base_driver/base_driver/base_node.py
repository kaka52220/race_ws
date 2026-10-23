#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
base_driver —— 阿克曼底盘驱动节点（ROS 2）

/cmd_vel 约定（和 brain_node 对齐）：
    linear.x   归一化速度 0~1，乘 speed_scale_pct 得到底盘速度百分比。
               实测映射（2026-10-24 新固件，落地编码器标定）：
                   speed  8 → 0.106 m/s
                   speed 10 → 0.138 m/s
                   speed 12 → 0.161 m/s   ← brain 的 max_speed=0.12 走这里
                   speed 15 → 0.208 m/s
               即每单位约 0.0134 m/s，固件侧闭环，带负载补偿。
    angular.z  舵机角（弧度）。底盘约定 steer_deg = -degrees(angular.z)，负角=左。

底盘接口（两条，各管一件事）：
    set_car_run(state, speed)       state 0=停 1=前进 2=后退，speed 0~100
    set_akm_steering_angle(deg)     前轮舵机角，负=左、正=右（2026-10-24 现场验证）
"""
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from Rosmaster_Lib import Rosmaster


def speed_to_run(pct):
    """百分比 → set_car_run 的 (state, speed)。0.5% 以下视为停车。"""
    if abs(pct) < 0.5:
        return 0, 0
    return (1 if pct > 0 else 2), int(min(100.0, abs(round(pct))))


class BaseDriverNode(Node):
    def __init__(self):
        super().__init__('base_node')

        self.declare_parameter('speed_scale_pct', 100.0)  # linear.x=1.0 对应多少百分比
        self.declare_parameter('steer_max_deg', 30.0)     # 舵机限幅
        self.declare_parameter('debug_log', True)

        self.bot = Rosmaster(com='/dev/robot_base')
        self.bot.create_receive_threading()
        time.sleep(0.5)
        self.bot.set_car_type(5)
        self.get_logger().info("✅ 底盘初始化成功，等待 /cmd_vel ...")

        self.create_subscription(Twist, '/cmd_vel', self.on_cmd, 10)

        # 看门狗：0.5 秒没收到指令就停车
        self.last_cmd_time = time.time()
        self.create_timer(0.1, self.watchdog)

    def on_cmd(self, msg):
        g = self.get_parameter
        self.last_cmd_time = time.time()

        pct = float(msg.linear.x) * float(g('speed_scale_pct').value)
        pct = max(-100.0, min(100.0, pct))
        state, spd = speed_to_run(pct)
        self.bot.set_car_run(state, spd)

        lim = float(g('steer_max_deg').value)
        steer = max(-lim, min(lim, -float(msg.angular.z) * 57.2958))
        self.bot.set_akm_steering_angle(int(round(steer)))

        if bool(g('debug_log').value):
            self.get_logger().info(f'🚗 速度={spd}% 转角={int(round(steer))}度')

    def watchdog(self):
        if time.time() - self.last_cmd_time > 0.5:
            self.bot.set_car_run(0, 0)
            self.bot.set_akm_steering_angle(0)


def main(args=None):
    rclpy.init(args=args)
    node = BaseDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.bot.set_car_run(0, 0)
        node.bot.set_akm_steering_angle(0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
