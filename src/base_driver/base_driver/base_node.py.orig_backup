#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from Rosmaster_Lib import Rosmaster
import time

class BaseDriverNode(Node):
    def __init__(self):
        super().__init__('base_driver_node')
        
        self.bot = Rosmaster(com='/dev/robot_base')
        self.bot.create_receive_threading()
        time.sleep(0.5)
        self.bot.set_car_type(5) 
        self.get_logger().info("✅ 底盘初始化成功，等待 ROS2 指令...")

        self.subscription = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        
        # 🐕 新增看门狗定时器：每 0.1 秒检查一次
        self.last_cmd_time = time.time()
        self.watchdog_timer = self.create_timer(0.1, self.watchdog_check)

    def cmd_vel_callback(self, msg):
        self.last_cmd_time = time.time() # 更新收到指令的时间
        
        v_x = msg.linear.x
        v_z = msg.angular.z
        steer_angle = int(-v_z * 57.3)
        steer_angle = max(-30, min(30, steer_angle))
        
        self.bot.set_car_motion(v_x, 0, 0)
        self.bot.set_akm_steering_angle(steer_angle)
        # 打印日志
        self.get_logger().info(f'🚗 驱动下发: 速度={v_x:.2f} m/s, 转角={steer_angle}度')

    def watchdog_check(self):
        # 如果超过 0.5 秒没收到新指令，强制停车
        if time.time() - self.last_cmd_time > 0.5:
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
        node.bot.set_car_motion(0, 0, 0)
        node.bot.set_akm_steering_angle(0)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
