# 单个检测目标

# 目标类别常量
uint8 CLASS_CROSSWALK=1
uint8 CLASS_BARRIER=2
uint8 CLASS_TRAFFIC_LIGHT=3
uint8 CLASS_START_LINE=4
uint8 CLASS_ARROW_SIGN=5
uint8 CLASS_LANE_LEFT=6
uint8 CLASS_LANE_RIGHT=7

uint8   class_id         # 类别，取值见上面常量
float32 confidence       # 置信度 0~1

float32 cx               # 框中心 x，归一化 0~1
float32 cy               # 框中心 y，归一化 0~1
float32 w                # 框宽，归一化 0~1
float32 h                # 框高，归一化 0~1

# 下面两个由 bbox 底边经 IPM 反投到地面得到
float32 dist_m           # 距车头的纵向距离 [m]，无效填 -1
float32 bearing_rad      # 相对车头的方位角 [rad]，左正，无效填 0

# 仅对红绿灯有意义
uint8 COLOR_UNKNOWN=0
uint8 COLOR_RED=1
uint8 COLOR_YELLOW=2
uint8 COLOR_GREEN=3
uint8   color_state      # 灯色，取值见上面常量
