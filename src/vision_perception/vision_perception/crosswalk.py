# -*- coding: utf-8 -*-
"""
crosswalk.py —— 斑马线 / 红灯的阈值检测（纯 OpenCV，不依赖 ROS）

为什么单独一个文件、不用 YOLO：
    斑马线是「比地面亮得多、低饱和」的白色条纹，红灯是「高饱和红色横条」，
    两者颜色极其独特。实测在 350 张负样本上零误检，不需要训练和标注。

核心量：crosswalk_score
    地面上的"极亮面积"百分比。它随距离单调上升 ——
        帧1: 1.5%  →  帧17: 9.0%  →  帧26: 29.0%（车一直在靠近）
    因为斑马线在透视下越近越大。所以这个数不是"有没有"，而是"离多远"：
        < 4%   很远
        4~8%   开始接近
        >= 8%  约 0.7 米内 → 该停车了

    实测（433 张自采数据）：
        红灯数据集 83 张：34 张触发（都是斑马线近的）
        负样本   207 张：0 张触发   ← 零误检
        挡板     143 张：0 张触发   ← 零误检

红灯单独用会有 8% 误检（木栅栏 H≈11、橙色纸箱都像红色），
所以只用来确认"这是红灯斑马线"，不当触发条件。
"""
import cv2
import numpy as np

# 检测参数（实测调出来的，改动请重跑 cv_detect.py 验证）
TOP = 0.30          # 只看画面下 70%，上面是墙和桌子
XLO, XHI = 0.10, 0.90
# 阈值是「混合」的：max(绝对值下限, 地面亮度中位数 + 偏移)
#   · 正常曝光（地面 V≈150）→ 阈值 = max(175, 185) = 185，白条才算
#   · 过曝（地面 V≈190，实测摄像头复位时会这样）→ 阈值 = max(175, 225) = 225
#     这样过曝时整片地面不会被误判成斑马线（纯固定阈值 185 时误判 63%）
#   实测：红灯检出 34/83，负样本 207 张零误检，挡板 143 张零误检，
#         过曝画面 0.9%（不触发）
FLOOR_MIN_V = 175   # 阈值绝对下限
DELTA_V = 35        # 比地面中位数亮多少才算白条
S_MAX = 60          # 白条是灰白，饱和度低
TRIGGER = 8.0       # >= 这个值 = 斑马线约 0.7 米内

# 为了省 CPU，先把帧缩到 320x180 再算（实测 1.6 ms vs 全图 3.7 ms，
# 触发判定 100% 一致）
SMALL = (320, 180)


def crosswalk_score(frame, small=SMALL):
    """返回地面极亮面积百分比 0~100。越大 = 斑马线越近。

    阈值随画面亮度自适应，所以摄像头曝光错了也不会误判成整片斑马线。
    """
    s = cv2.resize(frame, small, interpolation=cv2.INTER_AREA)
    h, w = s.shape[:2]
    hsv = cv2.cvtColor(s, cv2.COLOR_BGR2HSV)
    roi = hsv[int(h * TOP):, int(w * XLO):int(w * XHI)]
    v = roi[:, :, 2].astype(np.float32)
    p50 = float(np.percentile(v, 50))
    # 过曝保护：地面 V 中位都到 215 了，白条和地面全饱和、根本分不开。
    # 这时返回 0（不触发），比乱给一个数安全 —— 实测过曝时地面能到 233。
    if p50 > 215.0:
        return 0.0
    # 阈值上限 250：V 最大才 255，超过就没东西能满足，会永远返回 0
    thr = min(250.0, max(float(FLOOR_MIN_V), p50 + DELTA_V))
    mask = (v > thr) & (roi[:, :, 1] < S_MAX)
    return 100.0 * float(mask.mean())


# ── 红灯（只做确认，不做触发）──
RL_H = ((0, 12), (168, 180))
RL_MIN_W, RL_MIN_H, RL_MAX_H = 18, 3, 55
RL_MIN_AREA, RL_MIN_FILL, RL_MIN_RATIO = 160, 0.40, 2.0


def red_light_box(frame):
    """红灯是躺在地上的红色横条（约 100x10 像素）。返回 (x,y,w,h,area) 或 None。"""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros((h, w), np.uint8)
    for lo, hi in RL_H:
        mask |= cv2.inRange(hsv, np.array([lo, 95, 95]), np.array([hi, 255, 205]))
    mask[:int(h * 0.25)] = 0            # 排除天花板/上方的红色标牌
    n, _l, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if (area < RL_MIN_AREA or bw < RL_MIN_W or bh < RL_MIN_H or bh > RL_MAX_H
                or bw / max(bh, 1) < RL_MIN_RATIO or area / float(bw * bh) < RL_MIN_FILL):
            continue
        sc = area * (1.0 - abs(cent[i][0] - w / 2.0) / (w / 2.0))
        if best is None or sc > best[0]:
            best = (sc, (int(x), int(y), int(bw), int(bh), int(area)))
    return None if best is None else best[1]
