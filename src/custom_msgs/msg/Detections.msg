# 一帧里的所有检测目标（由 vision_node 以 15 Hz 发布，brain_node 订阅）
# 用数组而不是单条，是因为一帧里可能出现多个目标（比如两块挡板）

std_msgs/Header header

Detection[] items
