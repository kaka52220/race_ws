# 车道几何信息（由 vision_node 以 30 Hz 发布，brain_node 订阅）
# 所有角度/偏移的符号约定：**左正**
#   车偏左 → lateral_error 为正
#   车道要往左拐 → heading_error 为正
#   车道向左弯 → curvature 为正

std_msgs/Header header

float32 lateral_error    # 车体中心相对车道中心线的横向偏差 [m]
float32 heading_error    # 车道方向相对车头方向的夹角 [rad]
float32 curvature        # 车前三 0.6 m 处的车道曲率 [1/m]
float32 lane_width       # 车道净宽 [m]
float32 quality          # 拟合置信度 0~1，越低越不可信

bool    is_lost          # 是否丢线（两条线都拟合不出来）
uint8   mode             # 拟合模式，见下面常量

# ── 诊断字段（调参用，控制逻辑不要依赖它们）──
uint32  n_points         # 这一帧在鸟瞰图里捞到多少个黄像素
                         #   0        → 完全没黄（没对着赛道 / 相机没图 / 阈值太严）
                         #   几十个   → 有零星黄点，不够拟合
                         #   几千上万 → 点够了，但可能拟合跑偏
string  reject_reason    # is_lost=true 时的具体原因；可信时为空字符串
                         #   例："黄像素仅 12 个" / "有效点只跨 0.25m，太短" /
                         #       "theta=-60° 超限"

uint8 MODE_BOTH=0
uint8 MODE_LEFT=1
uint8 MODE_RIGHT=2
uint8 MODE_LOST=3
