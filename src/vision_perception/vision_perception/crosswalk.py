# -*- coding: utf-8 -*-
"""
crosswalk.py —— 斑马线 / 红灯的阈值检测（纯 OpenCV，不依赖 ROS）

为什么不用 YOLO：
    斑马线是「比地面亮、低饱和」的白色条纹，红灯是「高饱和红色横条」。
    实测在 350 张负样本上零误检，不需要训练和标注。

怎么区分「斑马线」和「起点线」（两者都是白线，面积都会超阈值）：
    斑马线 = 5~6 条**等宽**的平行白条；
    起点线 = 1 条粗白带（不分段）。
    所以除了看面积，还要看「这一行被分成几段、各段宽度是否一致」：
        斑马线：段数 6，各段宽度 [13,13,13,13,13,13]  →  宽度 CV ≈ 0.00
        起点线：段数 3，各段宽度 [22,61,15]           →  宽度 CV ≈ 0.75
    规则：段数 >= 4 且 宽度 CV <= 0.35，才算斑马线。
    （实测：起点线面积 8.4% 会超阈值，但 CV=0.75 被拒 → 起点不停车）

两个量：
    crosswalk_score(img)   0 = 不是斑马线或离得很远；>0 = 白条面积百分比。
                           实测它随距离单调上升：
                               帧1: 1.5%  →  帧17: 9.0%  →  帧26: 29.0%
                           因为斑马线在透视下越近越大。所以它测的是「离多远」：
                                < 4%   很远
                                4~8%   开始接近
                                >= 8%  约 0.7 米内，该停车了
    red_light_box(img)     红灯是躺在地上的红色横条，只用来确认「红灯斑马线」。

实测（433 张自采数据 + 起点线现场帧）：
    面积 >= 8% 且 是等宽条纹：
        斑马线   34/83 触发（全是近的）
        负样本    0/207 误检
        挡板      0/143 误检
        起点线    不触发
"""
import cv2
import numpy as np

# 检测参数（实测调出来的，改动请重跑 probe_line.py / cv_detect.py 验证）
TOP = 0.30          # 只看画面下 70%，上面是墙和桌子
XLO, XHI = 0.10, 0.90
# 阈值是「混合」的：max(绝对值下限, 地面亮度中位数 + 偏移)
#   · 正常曝光（地面 V≈165）→ 阈值 = max(175, 200) = 200
#   · 过曝（地面 V≈233，摄像头复位时会这样）→ p50>215 直接判「测不了」，返回 0
FLOOR_MIN_V = 175   # 阈值绝对下限
DELTA_V = 35        # 比地面中位数亮多少才算白条
S_MAX = 60          # 白条是灰白，饱和度低
TRIGGER = 8.0       # >= 这个值 = 斑马线约 0.7 米内

# 条纹判定：把一行白像素分成若干段，看段数和宽度是否均匀
STRIPE_MIN_RUNS = 4     # 至少要 4 段（斑马线实测 6 段）
STRIPE_MAX_CV = 0.35    # 各段宽度的变异系数上限（斑马线 ≈0.00，起点线 ≈0.75）
STRIPE_MIN_RUN_PX = 9   # 段宽小于这个的当噪声丢掉（缩到 320 宽后）
STRIPE_MIN_COV = 0.25   # 这一行白像素占比低于这个不参与判定
STRIPE_MAX_COV = 0.95   # 整行全白（贴脸）也排除

# 为了省 CPU，先把帧缩到 320x180 再算
SMALL = (320, 180)


def _stripe_ok(mask):
    """这一帧是不是「等宽平行条纹」（斑马线）。

    mask: uint8 0/1，ROI 内的白像素图。
    逐行统计白像素被分成几段、各段宽度是否接近；
    任意一行满足「段数 >= 4 且 宽度 CV <= 0.35」就判为条纹。
    """
    for r in range(mask.shape[0]):
        row = mask[r]
        cov = float(row.mean())
        # 太稀（噪声）或整行全白（贴脸的一条线）都不看
        if cov < STRIPE_MIN_COV or cov > STRIPE_MAX_COV:
            continue
        d = np.diff(np.concatenate(([0], row, [0])))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        widths = [int(e - s) for s, e in zip(starts, ends)
                  if (e - s) >= STRIPE_MIN_RUN_PX]
        if len(widths) < STRIPE_MIN_RUNS:
            continue
        mean_w = float(np.mean(widths))
        if mean_w > 0 and float(np.std(widths)) / mean_w <= STRIPE_MAX_CV:
            return True
    return False


def crosswalk_metrics(frame, small=SMALL):
    """返回 (面积百分比, 是否等宽条纹)。面积判距离，条纹判身份。"""
    s = cv2.resize(frame, small, interpolation=cv2.INTER_AREA)
    h, w = s.shape[:2]
    hsv = cv2.cvtColor(s, cv2.COLOR_BGR2HSV)
    roi = hsv[int(h * TOP):, int(w * XLO):int(w * XHI)]
    v = roi[:, :, 2].astype(np.float32)
    p50 = float(np.percentile(v, 50))
    # 过曝保护：地面 V 中位到 215 说明白条和地面全饱和、分不开。
    # 这时返回 0（不触发），比乱给一个数安全 —— 实测过曝时地面能到 233。
    if p50 > 215.0:
        return 0.0, False
    thr = min(250.0, max(float(FLOOR_MIN_V), p50 + DELTA_V))
    mask = ((v > thr) & (roi[:, :, 1] < S_MAX)).astype(np.uint8)
    area = 100.0 * float(mask.mean())
    if area < 1.0:                       # 几乎没有白的，不用数条纹了
        return area, False
    return area, _stripe_ok(mask)


def crosswalk_score(frame, small=SMALL):
    """返回 0~100。0 = 不是斑马线；>0 = 白条面积百分比（越大越近）。

    起点线是一整条白带、不是等宽条纹，会返回 0 —— 所以车在起点不会停。
    """
    area, stripe = crosswalk_metrics(frame, small)
    return area if stripe else 0.0


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
