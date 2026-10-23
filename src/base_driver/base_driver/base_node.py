#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
base_driver —— 阿克曼底盘驱动节点（ROS 2）

原版是队友写的，2026-10-23 改造，原因：
    原来是 set_car_motion(v_x, 0, 0) 设速度 —— 那是**麦轮/全向车**的速度矢量接口，
    对阿克曼车语义不对，实车表现为「速度完全不可控，给 0.03 就飞快」。

    阿克曼车正确的组合是：
        set_car_run(state, speed)        # speed 是 [-100, 100] 百分比
        set_akm_steering_angle(angle)    # 前轮舵机角，负=左，正=右
    两个接口各管一件事，不会互相干扰。

/cmd_vel 的约定（和 brain_node 对齐）：
    linear.x  速度，单位取决于 drive_mode：
                run    → 归一化 0~1（内部乘 100 变成百分比）
                motion → 原版语义，直接透传给 set_car_motion
    angular.z 舵机角（弧度）。
              底盘约定 steer_deg = -degrees(angular.z)，即负角=左。

可现场切换：ros2 param set /base_node drive_mode motion
"""
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from Rosmaster_Lib import Rosmaster


class BaseDriverNode(Node):
    def __init__(self):
        super().__init__('base_node')

        # ── 可调参数 ──
        self.declare_parameter('drive_mode', 'motor')    # 'motor'=直接给两路 PWM(推荐) | 'run'=百分比 | 'motion'=原版
        self.declare_parameter('max_speed_pct', 100.0)   # run 模式下 linear.x=1.0 对应多少百分比
        self.declare_parameter('steer_max_deg', 30.0)    # 舵机限幅
        self.declare_parameter('steer_sign', 1.0)        # 保持原版映射不变（steer = -degrees(v_z)）。方向统一在 brain_node 的 steer_sign 里调
        self.declare_parameter('debug_log', True)

        mode = str(self.get_parameter('drive_mode').value)
        self.get_logger().info(f"base_node 启动，drive_mode = {mode}")

        self.bot = Rosmaster(com='/dev/robot_base')
        self.bot.create_receive_threading()
        time.sleep(0.5)
        self.bot.set_car_type(5)
        self.get_logger().info("✅ 底盘初始化成功，等待 ROS2 指令...")

        self.subscription = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)

        # 看门狗：0.5 秒没收到指令就停车
        self.last_cmd_time = time.time()
        self.watchdog_timer = self.create_timer(0.1, self.watchdog_check)

    # ─────────────────────────────────────────────────────────
    def cmd_vel_callback(self, msg):
        g = self.get_parameter
        self.last_cmd_time = time.time()

        v_x = float(msg.linear.x)
        v_z = float(msg.angular.z)

        # 舵机角：底盘约定 steer_deg = -degrees(angular.z)
        steer = -v_z * 57.2958 * float(g('steer_sign').value)
        lim = float(g('steer_max_deg').value)
        steer = max(-lim, min(lim, steer))
        steer_i = int(round(steer))

        mode = str(g('drive_mode').value)
        if mode == 'motor':
            # ★ 2026-10-23 实测：这是唯一可靠的速度通路。
            #   通道映射（逐个给 PWM 看哪路编码器动，已确认）：
            #       通道1 = M1（空）  通道2 = M2（左后）
            #       通道3 = M3（空）  通道4 = M4（右后）
            #   落地实测的 PWM → 车速（车重负载，编码器 1267 脉冲/轮圈）：
            #       6,7  → 完全不动（静摩擦死区）
            #       8    → 0.055 m/s   ← 贴着死区，不稳
            #       9    → 0.185 m/s
            #       10   → 0.170 m/s   ← ★ 推荐巡航值
            #       11   → 0.191 m/s
            #       12   → 0.200 m/s
            #       14   → 0.237 m/s
            #   PWM=1~5 单给不会动，但被 PWM12 踢一下之后能维持（静摩擦 > 动摩擦）。
            #   所以 linear.x 的语义是「归一化 0~1」，×100 得到 PWM。
            #   brain_node 的 max_speed=0.10 → PWM 10 → 0.17 m/s。
            pwm = v_x * float(g('max_speed_pct').value)
            pwm = max(-100.0, min(100.0, pwm))
            pwm_i = int(round(abs(pwm))) if abs(pwm) >= 1.0 else 0
            sign = 1.0 if pwm >= 0 else -1.0
            self.bot.set_motor(0, sign * pwm_i, 0, sign * pwm_i)
            shown = f"PWM={pwm_i:d}"
        elif mode == 'run':
            # 阿克曼百分比油门。注意：实测这个接口对速度值不敏感，别用。
            pct = v_x * float(g('max_speed_pct').value)
            pct = max(-100.0, min(100.0, pct))
            if abs(pct) < 0.5:
                state, mag = 0, 0               # 停
            else:
                state, mag = (1, abs(round(pct))) if pct > 0 else (2, abs(round(pct)))
            self.bot.set_car_run(state, int(mag))
            shown = f"{pct:.0f}%"
        else:
            # 原版：速度矢量接口。注意：实测闭环会跑到满速，别用。
            self.bot.set_car_motion(v_x, 0, 0)
            shown = f"{v_x:.3f}"

        self.bot.set_akm_steering_angle(steer_i)

        if bool(g('debug_log').value):
            self.get_logger().info(
                f'🚗 下发: 速度={shown} 转角={steer_i}度  (mode={mode})')

    # ─────────────────────────────────────────────────────────
    def watchdog_check(self):
        if time.time() - self.last_cmd_time > 0.5:
            self.bot.set_motor(0, 0, 0, 0)
            self.bot.set_car_run(0, 0)
            self.bot.set_car_motion(0, 0, 0)
            self.bot.set_akm_steering_angle(0)


def main(args=None):
    rclpy.init(args=args)
    node = BaseDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.bot.set_motor(0, 0, 0, 0)
        node.bot.set_car_run(0, 0)
        node.bot.set_car_motion(0, 0, 0)
        node.bot.set_akm_steering_angle(0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
