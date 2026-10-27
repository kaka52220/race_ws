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
        
        # 输出走 PulseAudio 默认 sink（USB 音箱）。
        # aplay -D plughw:3,0 在这台机器上没有 card 3，会直接失败。
        if "人行道" in text:
            os.system("paplay /home/orangepi/piper/piper/yield_walk.wav "
                      "> /tmp/aplay.log 2>&1 &")
        else:
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
