#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
base_driver —— 阿克曼底盘驱动节点（ROS 2）

本节点是**唯一的硬件适配层**：负责单位换算和硬件限幅，不含任何控制逻辑。

/cmd_vel 约定（和 brain_node 对齐）：
    linear.x   车速，单位 **米/秒**。用 mps_per_pct 换算成底盘百分比。
               标定（2026-10-24 落地编码器实测，线性度好）：
                   speed  8 → 0.106 m/s     speed 12 → 0.161 m/s
                   speed 10 → 0.138 m/s     speed 15 → 0.208 m/s
               ⇒ 每 1% ≈ 0.0134 m/s，固件侧闭环、带负载补偿。
    angular.z  舵机角（弧度）。底盘约定 steer_deg = -degrees(angular.z)，负角=左。

底盘接口（两条，各管一件事）：
    set_car_run(state, speed)       state 0=停 1=前进 2=后退，speed 0~100
    set_akm_steering_angle(deg)     前轮舵机角，负=左、正=右（2026-10-24 现场验证）

安全（底盘固件没有指令超时保护，实测）：
    启动先停车、看门狗 0.5s、收到 SIGINT/SIGTERM/SIGHUP 先停车再退出。
"""
import signal
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped
from Rosmaster_Lib import Rosmaster


def speed_to_run(pct):
    """百分比 → set_car_run 的 (state, speed)。0.5% 以下视为停车。

    ★ 停车时 speed 必须传非 0，底盘固件才会主动刹车：
        case MOTION_STOP:
            Motion_Stop(speed==0 ? STOP_FREE : STOP_BRAKE);
      传 (0, 0) 是 STOP_FREE（自由滑行），0.25 m/s 会多滑 2cm，
      斑马线停车正好压到线上。传 (0, 1) 就是 STOP_BRAKE。
    """
    if abs(pct) < 0.5:
        return 0, 1          # state=0 停 + speed≠0 → 主动刹车
    return (1 if pct > 0 else 2), int(min(100.0, abs(round(pct))))


class BaseDriverNode(Node):
    def __init__(self):
        super().__init__('base_node')

        # linear.x 的单位是 **米/秒**。底盘要的是百分比，靠这个实测标定值换算。
        # 标定来源（tools/ground_run_test.py，落地编码器实测，2026-10-24）：
        #     1% → 0.0134 m/s   8% → 0.106   12% → 0.161   15% → 0.208
        # 线性度很好，所以用单一常数即可。换车/换固件要重新标。
        self.declare_parameter('mps_per_pct', 0.0134)
        self.declare_parameter('steer_hw_max_deg', 40.0)  # 硬件舵机物理限幅，兜底用
        self.declare_parameter('debug_log', False)

        self.bot = Rosmaster(com='/dev/robot_base')
        self.bot.create_receive_threading()
        time.sleep(0.5)
        self.bot.set_car_type(5)

        # ★ 启动第一件事就是停车。
        #   底盘固件没有任何指令超时保护（2026-10-24 实测：只发一次
        #   set_car_run(1,6)，之后不再发任何指令，车会一直跑下去）。
        #   所以节点被 kill 后底盘会保持最后一条速度指令。
        #   如果节点被 respawn 重启，这里先把车钉住，避免它带着旧速度跑。
        self.safe_stop()
        self.get_logger().info("✅ 底盘初始化成功，等待 /cmd_vel ...")

        self.create_subscription(Twist, '/cmd_vel', self.on_cmd, 10)

        # 看门狗：0.5 秒没收到指令就停车
        self.last_cmd_time = time.time()
        self.create_timer(0.1, self.watchdog)

        # andy: 里程计原始数据（底盘速度 + 陀螺仪），给 brain 的 mission 用
        self.pub_odom = self.create_publisher(TwistStamped, '/odom_raw', 20)
        self.create_timer(0.02, self.pub_odom_raw)

    def safe_stop(self):
        """停车。连发几次，因为要保证这一帧串口写真的到得了底盘。"""
        for _ in range(5):
            try:
                self.bot.set_car_run(0, 1)      # 1 = 主动刹车（见 speed_to_run 注释）
                self.bot.set_akm_steering_angle(0)
            except Exception:
                pass
            time.sleep(0.04)

    def on_cmd(self, msg):
        g = self.get_parameter
        self.last_cmd_time = time.time()

        # 米/秒 → 底盘百分比。唯一的一处物理换算。
        v_mps = float(msg.linear.x)
        pct = v_mps / float(g('mps_per_pct').value)
        pct = max(-100.0, min(100.0, pct))
        state, spd = speed_to_run(pct)
        self.bot.set_car_run(state, spd)

        lim = float(g('steer_hw_max_deg').value)
        steer = max(-lim, min(lim, -float(msg.angular.z) * 57.2958))
        self.bot.set_akm_steering_angle(int(round(steer)))

        if bool(g('debug_log').value):
            self.get_logger().info(
                f'🚗 {v_mps:.3f}m/s→{spd}% 转角={int(round(steer))}度')

    def pub_odom_raw(self):
        try:
            vx, vy, vz = self.bot.get_motion_data()
            gx, gy, gz = self.bot.get_gyroscope_data()
        except Exception:
            return
        m = TwistStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'base_link'
        m.twist.linear.x = float(vx)
        m.twist.linear.y = float(vy)
        m.twist.angular.x = float(vz)     # 底盘自己算的角速度，仅日志对照
        m.twist.angular.z = float(gz)
        self.pub_odom.publish(m)

    def watchdog(self):
        if time.time() - self.last_cmd_time > 0.5:
            self.bot.set_car_run(0, 1)          # 退出也刹车，别让车带着惯性冲出去
            self.bot.set_akm_steering_angle(0)


def main(args=None):
    rclpy.init(args=args)
    node = BaseDriverNode()

    # ★ 底盘没有指令超时保护，所以「退出前必须停车」这件事不能只靠 finally：
    #   Ctrl+C、kill、关终端分别走 SIGINT / SIGTERM / SIGHUP，
    #   ros2 launch 在子进程不退时会补一个 SIGKILL，那个谁也拦不住。
    #   这里把三个能拦的都拦下来，先停车再退出。
    def _on_signal(signum, frame):
        node.safe_stop()
        try:
            rclpy.try_shutdown()
        except Exception:
            pass
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError):
            pass

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, Exception):
        pass
    finally:
        node.safe_stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
