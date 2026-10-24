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

反馈（2026-10-24 v2 新增，给 brain 的融合估计器用）：
    /imu          sensor_msgs/Imu          angular_velocity.z = 偏航角速度 rad/s，**左正**
                                           （get_gyroscope_data() 的 g_z × gyro_scale × gyro_sign）
                                           linear_acceleration = get_accelerometer_data()
    /wheel_speed  geometry_msgs/TwistStamped linear.x = v_x × speed_fb_scale [m/s]
    底板每 10 ms 发一包、四包轮换，所以每种数据 40 ms 刷新一次；这里按 imu_hz 发布。
    ★ 首次上车先跑 tools/check_imu.py 确认 gyro_scale / gyro_sign / speed_fb_scale。

安全（底盘固件没有指令超时保护，实测）：
    启动先停车、看门狗 0.5s、收到 SIGINT/SIGTERM/SIGHUP 先停车再退出。
"""
import signal
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import Imu
from Rosmaster_Lib import Rosmaster


def speed_to_run(pct):
    """百分比 → set_car_run 的 (state, speed)。0.5% 以下视为停车。"""
    if abs(pct) < 0.5:
        return 0, 0
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
        self.declare_parameter('debug_log', True)
        # ── IMU / 车速反馈 ──
        self.declare_parameter('publish_feedback', True)
        self.declare_parameter('imu_hz', 50.0)
        self.declare_parameter('gyro_scale', 1.0)     # g_z → rad/s（驱动库已换算成 rad/s 时为 1）
        self.declare_parameter('gyro_sign', 1.0)      # 左转为正就是 +1；check_imu.py 实测
        self.declare_parameter('gyro_bias_s', 1.0)    # 启动静止时采多少秒估零偏
        self.declare_parameter('speed_fb_scale', 1.0)  # v_x → m/s 的修正系数

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

        # IMU / 车速反馈
        self.gz_bias = 0.0
        if bool(self.get_parameter('publish_feedback').value):
            self.gz_bias = self.estimate_gyro_bias()
            self.pub_imu = self.create_publisher(Imu, '/imu', 20)
            self.pub_speed = self.create_publisher(TwistStamped, '/wheel_speed', 20)
            hz = max(float(self.get_parameter('imu_hz').value), 1.0)
            self.create_timer(1.0 / hz, self.publish_feedback)
            self.get_logger().info(
                f"📡 发布 /imu /wheel_speed，{hz:.0f} Hz，陀螺零偏 {self.gz_bias:+.4f}")

    def estimate_gyro_bias(self):
        """启动时车是停着的：采 gyro_bias_s 秒 g_z 取平均当零偏。"""
        n, acc = 0, 0.0
        t_end = time.time() + float(self.get_parameter('gyro_bias_s').value)
        while time.time() < t_end:
            try:
                acc += float(self.bot.get_gyroscope_data()[2])
                n += 1
            except Exception:
                pass
            time.sleep(0.02)
        return acc / n if n else 0.0

    def publish_feedback(self):
        g = self.get_parameter
        try:
            ax, ay, az = self.bot.get_accelerometer_data()
            gx, gy, gz = self.bot.get_gyroscope_data()
            vx, vy, vz = self.bot.get_motion_data()
        except Exception as ex:          # 串口偶发异常不能把节点带崩
            self.get_logger().warn(f'读 IMU 失败: {ex}', throttle_duration_sec=2.0)
            return
        stamp = self.get_clock().now().to_msg()
        k = float(g('gyro_scale').value) * float(g('gyro_sign').value)

        m = Imu()
        m.header.stamp = stamp
        m.header.frame_id = 'base_link'
        m.orientation_covariance[0] = -1.0           # 不提供姿态
        m.angular_velocity.x = float(gx) * k
        m.angular_velocity.y = float(gy) * k
        m.angular_velocity.z = (float(gz) - self.gz_bias) * k
        m.linear_acceleration.x = float(ax)
        m.linear_acceleration.y = float(ay)
        m.linear_acceleration.z = float(az)
        self.pub_imu.publish(m)

        s = TwistStamped()
        s.header.stamp = stamp
        s.header.frame_id = 'base_link'
        s.twist.linear.x = float(vx) * float(g('speed_fb_scale').value)
        self.pub_speed.publish(s)

    def safe_stop(self):
        """停车。连发几次，因为要保证这一帧串口写真的到得了底盘。"""
        for _ in range(5):
            try:
                self.bot.set_car_run(0, 0)
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

    def watchdog(self):
        if time.time() - self.last_cmd_time > 0.5:
            self.bot.set_car_run(0, 0)
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
