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
        
        # 输出走 PulseAudio 的默认 sink：
        #   pactl list sinks 显示 USB 音箱（JieLi USB Composite）是 RUNNING，
        #   而 aplay -D plughw:2,0 是绕过 PulseAudio 直接写 ES8388 的 3.5mm 口，
        #   所以之前"aplay 成功但没声音"。改用 paplay 就送到音箱。
        # & 后台播放，不阻塞订阅回调。
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
