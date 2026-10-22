#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import os

class VoiceNode(Node):
    def __init__(self):
        super().__init__('voice_node')
        self.sub = self.create_subscription(String, '/voice_cmd', self.voice_callback, 10)
        self.get_logger().info("🔊 语音播报节点已启动...")

    def voice_callback(self, msg):
        text = msg.data
        self.get_logger().warn(f'🔈 触发语音播报: 【{text}】')
        
        # 将你测试成功的完美播放指令直接写死在这里！
        # 加上 & 后台播放，绝对不卡主线程
        if "人行道" in text:
            os.system("aplay -D plughw:3,0 /home/orangepi/piper/piper/yield_walk.wav &")
        else:
            # 如果以后有红灯或其他语音，可以在这里扩展
            pass

def main(args=None):
    rclpy.init(args=args)
    node = VoiceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
