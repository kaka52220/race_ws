# -*- coding: utf-8 -*-
"""
lane_tracker.py —— 车道巡线核心算法 v2（纯 Python，不依赖 ROS）

为什么重写（旧版 RANSAC 二次曲线的根本问题）：
    1. 旧版把鸟瞰图里所有黄像素当一堆点，用 RANSAC 找两条 x = f(d) 的抛物线。
       这条赛道弯道半径只有 ~0.6 m，视野里一条线能转过 90°，x = f(d) 根本表示不了，
       而且隔壁赛道的黄线也会被拟合进来。
    2. 只看到一条线时，按「线在车左还是车右」判断它是左线还是右线。
       急弯里外侧线会扫过车正前方，一判反，中心就推到车道外面去了。
    3. 用「水平平移一个车道宽」推另一条线，弯里真正的水平间距是 w/cosφ。

v2 的做法 —— 沿着路面「走」出中心线（射线行进 / ray marching）：
    · 起点（种子）是车正前方最近处的路面，方向取上一帧的车道方向。
    · 每走一步，从当前中心点沿**法向**向左、向右各打一条射线，
      找到的第一个黄像素就是左 / 右边界。左右由「种子在路面上」这一事实决定。
    · 只认离中心最近的线 → 隔壁赛道的线天然被挡在外面。
    · 一侧看不到（出视野）→ 另一侧沿法向偏移半个车道宽；射线本身就是法向，
      所以弯里不会有 w/cosφ 的误差。
    · 一侧距离突然跳变（路口虚线缺口 / 宽区）→ 这一侧作废，用另一侧推。
    · 中心点压在黄线上（路口横向虚线）→ 直接往前跨过去，跨太久才停。
    · 方向跟着中心点走，所以车道拐成横向、甚至 U 型回旋也能一直跟下去。

输出（和旧版接口一致，vision_node / brain 不用改）：
    e_m        车体相对车道中心的横向偏差。车偏左 → 正
    theta_rad  车道方向相对车头的夹角。车道往左拐 → 正
    curvature  车道曲率（1/m）。向左弯 → 正
    新增：look_x_m / look_d_m  中心线上距车 look_d_m 远的预瞄点（给纯追踪用）
          path                 中心线点列 (N,2)，每行 (d, x)
"""
import math

import cv2
import numpy as np

try:
    from .lane_params import DEFAULTS
except ImportError:
    from lane_params import DEFAULTS


class LaneTracker:
    def __init__(self, overrides=None):
        self.p = dict(DEFAULTS)
        if overrides:
            self.p.update(overrides)
        self._build_ipm()
        self.w_ref = float(self.p["lane_width_m"])     # 线中心到线中心
        self.prev_seed_x = None      # 上一帧起点处的车道中心 x（米）
        self.prev_dir = None         # 上一帧起点处的车道方向（单位向量 (d, x)）

    def reset(self):
        self.w_ref = float(self.p["lane_width_m"])
        self.prev_seed_x = None
        self.prev_dir = None

    # ═══════════════════════════════════════════════════════════
    # IPM：地面 (d, x) → 原图 (u, v)，查找表只算一次
    #   相机坐标 X 右、Y 下、Z 前；离地 h，俯角 θ，偏航 yaw（右正）
    # ═══════════════════════════════════════════════════════════
    def _build_ipm(self):
        p = self.p
        W_img, H_img = p["img_w"], p["img_h"]
        vfov = 2 * math.atan(math.tan(math.radians(p["hfov_deg"]) / 2) * H_img / W_img)
        self.f = (H_img / 2) / math.tan(vfov / 2)
        self.cx, self.cy = W_img / 2, H_img / 2

        mpp = p["bev_m_per_px"]
        Wb = max(2, int(round(p["bev_width_m"] / mpp)))
        Hb = max(2, int(round(p["bev_depth_m"] / mpp)))
        dmin = p["bev_d_min"]
        dmax = dmin + (Hb - 1) * mpp
        self.Wb, self.Hb, self.mpp = Wb, Hb, mpp
        self.d_min, self.d_max = dmin, dmax
        self.d_of_row = np.linspace(dmax, dmin, Hb)            # 第 0 行最远
        self.x_of_col = (np.arange(Wb) - (Wb - 1) / 2) * mpp    # 右正

        d = self.d_of_row.reshape(-1, 1)
        x = self.x_of_col.reshape(1, -1)
        u, v = self.ground_to_image(d, x)
        th = math.radians(p["pitch_deg"])
        self.horizon_row = self.cy - self.f * math.tan(th)
        v_top = max(self.horizon_row + p["horizon_margin_px"], p["roi_top"])
        ok = ((u >= 0) & (u < W_img - 1) & (v >= v_top)
              & (v < min(H_img - 1, p["roi_bottom"])))
        self.map_x = np.where(ok, u, -1).astype(np.float32)
        self.map_y = np.where(ok, v, -1).astype(np.float32)
        self.bev_valid = ok

    def ground_to_image(self, d, x):
        """地面点 (d 前, x 右，米) → 原图像素 (u, v)。"""
        p = self.p
        th = math.radians(p["pitch_deg"])
        yaw = math.radians(p["yaw_deg"])
        d = np.asarray(d, float)
        x = np.asarray(x, float)
        dc = d * math.cos(yaw) + x * math.sin(yaw)
        xc = -d * math.sin(yaw) + x * math.cos(yaw)
        denom = p["h_m"] * math.sin(th) + dc * math.cos(th)
        denom = np.where(denom > 1e-3, denom, np.nan)
        yc = p["h_m"] * math.cos(th) - dc * math.sin(th)
        return self.f * xc / denom + self.cx, self.f * yc / denom + self.cy

    def to_bev(self, img):
        return cv2.remap(img, self.map_x, self.map_y, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # 米 ↔ 鸟瞰图像素
    def _rc(self, d, x):
        return (self.d_max - d) / self.mpp, x / self.mpp + (self.Wb - 1) / 2

    # ═══════════════════════════════════════════════════════════
    # 颜色分割
    # ═══════════════════════════════════════════════════════════
    def color_masks(self, bev):
        p = self.p
        hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
        yel = cv2.inRange(hsv, np.array([p["color_h_min"], p["color_s_min"], p["color_v_min"]]),
                          np.array([p["color_h_max"], 255, 255]))
        k = p["morph_open_px"]
        if k > 1:
            yel = cv2.morphologyEx(yel, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        k = p["morph_close_px"]
        if k > 1:
            yel = cv2.morphologyEx(yel, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        blue = cv2.inRange(hsv, np.array([p["blue_h_min"], p["blue_s_min"], p["blue_v_min"]]),
                           np.array([p["blue_h_max"], 255, 255]))
        # 灰色路面：低饱和 + 不太暗 + 非黄非蓝，再开运算去噪点
        road = cv2.inRange(hsv, np.array([0, 0, p["road_v_min"]]),
                           np.array([180, p["road_s_max"], 255]))
        road[(yel > 0) | (blue > 0)] = 0
        k = p["road_open_px"]
        if k > 1:
            road = cv2.morphologyEx(road, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        v = self.bev_valid
        return (yel > 0) & v, (blue > 0) & v, (road > 0) & v

    # ═══════════════════════════════════════════════════════════
    # 种子：最近处、离上一帧中心最近的一块路面
    # ═══════════════════════════════════════════════════════════
    def _find_seed(self, yel, blue, road_m=None):
        p = self.p
        x_pref = self.prev_seed_x if self.prev_seed_x is not None else 0.0
        c_pref = (x_pref / self.mpp + (self.Wb - 1) / 2)
        max_shift = p["seed_search_m"] / self.mpp
        min_run = int(p["seed_min_road_m"] / self.mpp)
        for j in range(self.Hb - 1, -1, -1):
            v = self.bev_valid[j]
            if v.sum() < 2 * min_run:
                continue
            road = v & ~yel[j] & ~blue[j]
            if road_m is not None:
                road = road & road_m[j]
            if road.sum() < min_run:
                continue
            # 找连续路面段，取离偏好位置最近、且足够宽的一段
            idx = np.nonzero(road)[0]
            cuts = np.nonzero(np.diff(idx) > 1)[0]
            starts = np.r_[idx[0], idx[cuts + 1]]
            ends = np.r_[idx[cuts], idx[-1]]
            best = None
            for a, b in zip(starts, ends):
                if b - a + 1 < min_run:
                    continue
                k = float(np.clip(c_pref, a, b))
                dist = abs(k - c_pref)
                if best is None or dist < best[0]:
                    best = (dist, k)
            if best is None:
                continue
            if best[0] > max_shift:
                return None
            return self.d_of_row[j], (best[1] - (self.Wb - 1) / 2) * self.mpp
        return None

    def _init_dir(self, yel, d0, x0):
        """起始方向：优先上一帧；否则用种子附近黄线的主方向（PCA）；再不行就正前方。"""
        if self.prev_dir is not None:
            return np.array(self.prev_dir, float)
        r0, c0 = self._rc(d0, x0)
        rad = int(0.30 / self.mpp)
        r1, r2 = max(0, int(r0) - rad), min(self.Hb, int(r0) + rad)
        c1, c2 = max(0, int(c0) - rad), min(self.Wb, int(c0) + rad)
        rr, cc = np.nonzero(yel[r1:r2, c1:c2])
        if len(rr) > 50:
            dd = -(rr - rr.mean())            # 行号向上 = d 增大
            xx = cc - cc.mean()
            cov = np.cov(np.stack([dd, xx]))
            w, vec = np.linalg.eigh(cov)
            if w[1] > 4 * max(w[0], 1e-9):
                t = vec[:, 1]
                if t[0] < 0:
                    t = -t
                if t[0] > 0.3:                # 太横的不信（可能是路口横线）
                    return t / np.linalg.norm(t)
        return np.array([1.0, 0.0])

    # ═══════════════════════════════════════════════════════════
    # 射线：从 P 沿方向 n 找第一个黄像素
    # ═══════════════════════════════════════════════════════════
    def _rays_lr(self, pd, px, n0, n1, reach_px):
        """左右两条射线一次算完（逐像素，与旧的逐条 _ray 规则一致）。
        (pd, px) 起点，(n0, n1) 右法向；左 = -n。
        self._labp 是四周补了 self._pad 像素零边的标签图（0 视野外 / 1 其它 / 2 黄 / 3 蓝），
        射线永远不会越界，所以不用 clip；查找用 bytes.find（C 速度）。
        返回 ((sl, 状态), (sr, 状态))。"""
        if getattr(self, "_ray_key", None) != reach_px:
            k = np.arange(1, reach_px + 1, dtype=np.float64)
            self._ray_kk = np.concatenate([-k, k])      # 前 N = 左，后 N = 右
            self._ray_key = reach_px
        N = reach_px
        mpp = self.mpp
        pad = self._pad
        kk = self._ray_kk
        r0 = (self.d_max - pd) / mpp + pad
        c0 = px / mpp + (self.Wb - 1) / 2 + pad
        r = np.rint(r0 - n0 * kk).astype(np.intp)
        c = np.rint(c0 + n1 * kk).astype(np.intp)
        buf = self._labp[r, c].tobytes()
        need = self._blue_need
        lw = self.p["line_width_m"]
        out = []
        for li in (buf[:N], buf[N:]):
            fb = li.find(b"\x00")
            if fb < 0:
                fb = N
            fh = li.find(b"\x02", 0, fb)
            if fh >= 0:
                out.append(((fh + 0.5) * mpp, "hit"))
                continue
            res = None
            if fb > need:
                kb = li.find(self._blue_run, 0, fb)
                if kb >= 0:
                    dist = (kb + 0.5) * mpp - lw
                    if dist > 0:
                        res = (dist, "edge")
            if res is None:
                res = (None, "fov") if fb < N else (None, "none")
            out.append(res)
        return out[0], out[1]

    def _lab_at(self, d, x):
        """(d, x) 处标签（0 视野外 / 1 其它 / 2 黄 / 3 蓝），纯标量。"""
        r = int(round((self.d_max - d) / self.mpp)) + self._pad
        c = int(round(x / self.mpp + (self.Wb - 1) / 2)) + self._pad
        H, W = self._labp.shape
        if 0 <= r < H and 0 <= c < W:
            return int(self._labp[r, c])
        return 0

    # ═══════════════════════════════════════════════════════════
    # 射线行进：沿车道走出中心线
    # ═══════════════════════════════════════════════════════════
    def march(self, yel, blue, road=None):
        p = self.p
        res = {"C": [], "L": [], "R": [], "side": [], "reason": ""}
        lab = self.bev_valid.astype(np.uint8)
        lab[blue] = 3
        lab[yel] = 2
        # 四周补零：最长射线 + 余量，射线查表永不越界
        reach = int((p["partner_max_m"] - p["line_width_m"] + 0.05) / self.mpp)
        self._pad = pad = reach + 2
        self._labp = cv2.copyMakeBorder(lab, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        self._blue_need = need = max(2, int(p["blue_edge_min_m"] / self.mpp))
        self._blue_run = b"\x03" * need
        seed = self._find_seed(yel, blue, road)
        if seed is None:
            res["reason"] = "近处找不到路面（种子丢失）"
            return res
        d0, x0 = seed
        t = self._init_dir(yel, d0, x0)
        # 每步只涉及几个 2 维量：全部用 Python 浮点，避免小 numpy 数组的调用开销
        t0, t1 = float(t[0]), float(t[1])
        pd, px = float(d0), float(x0)

        lw = p["line_width_m"]
        w_in = self.w_ref - lw                          # 内沿到内沿
        w_in_min = p["partner_min_m"] - lw
        w_in_max = p["partner_max_m"] - lw
        reach_px = int((w_in_max + 0.05) / self.mpp)
        step = p["march_step_m"]
        n_steps = int(p["march_len_m"] / step)
        jump = p["edge_jump_m"]
        max_turn = p["max_curvature_1pm"] * step * 1.5  # 每步方向最多转多少（弧度）
        skip_max = int(p["block_stop_m"] / step)
        lim_off = p["max_off_step_m"]
        start_lim = int(p["start_search_m"] / step)
        lab_at = self._lab_at
        Cs, Ls, Rs, sides = res["C"], res["L"], res["R"], res["side"]
        prev_sl = prev_sr = None
        off_exp = 0.0
        skip = 0
        for _ in range(n_steps):
            lb = lab_at(pd, px)
            if lb == 0:
                res["reason"] = "中心线走出相机视野"
                break
            if lb == 2:
                # 中心压在黄线上（路口横向虚线、起步时车身压线）→ 往前跨
                skip += 1
                if skip > skip_max:
                    res["reason"] = "前方被黄线挡住"
                    break
                pd += step * t0
                px += step * t1
                continue
            n0, n1 = -t1, t0                            # 右法向
            (sl, _), (sr, _) = self._rays_lr(pd, px, n0, n1, reach_px)
            # 连续性：同一侧边界距离不能突然跳（中心点每步都在动，用宽度变化判断）
            if sl is not None and prev_sl is not None and abs(sl - prev_sl) > jump + 0.5 * step:
                if sr is not None and prev_sr is not None and abs(sr - prev_sr) <= jump:
                    sl = None
            if sr is not None and prev_sr is not None and abs(sr - prev_sr) > jump + 0.5 * step:
                if sl is not None and prev_sl is not None and abs(sl - prev_sl) <= jump:
                    sr = None
            if sl is not None and sr is not None:
                w = sl + sr
                if w < w_in_min:
                    skip += 1
                    if skip > skip_max:
                        res["reason"] = "两边黄线挨得太近（横线缺口）"
                        break
                    pd += step * t0
                    px += step * t1
                    continue
                if w > w_in_max:
                    # 太宽：留下离「当前宽度估计」更吻合的那一侧
                    if abs(sl - w_in / 2) <= abs(sr - w_in / 2):
                        sr = None
                    else:
                        sl = None
            if sl is None and sr is None:
                skip += 1
                # 起步阶段（最近处视野很窄，两条线都还没进画面）允许多走一段
                lim = skip_max if Cs else start_lim
                if skip > lim:
                    res["reason"] = "两侧边界都看不到了"
                    break
                pd += step * t0
                px += step * t1
                continue
            skip = 0
            if sl is not None and sr is not None:
                off = 0.5 * (sr - sl)
                side = 0
            elif sl is not None:
                off = 0.5 * w_in - sl
                side = 1
            else:
                off = sr - 0.5 * w_in
                side = 2
            # 锁定之后，中心每步的横向修正有上限：车道不会一步横移好几厘米，
            # 大跳一定是射线打到了别的线（路口、隔壁车道）
            #（限制的是「相对上一步预期」的变化：上一步重新居中了预期 0，
            #  没居中（中心在视野外、沿着线走）预期就是上一步的偏移）
            if len(Cs) >= 3:
                off = max(off_exp - lim_off, min(off_exp + lim_off, off))
            cd_, cx_ = pd + off * n0, px + off * n1
            Cs.append((cd_, cx_))
            Ls.append((pd - sl * n0, px - sl * n1) if sl is not None else None)
            Rs.append((pd + sr * n0, px + sr * n1) if sr is not None else None)
            sides.append(side)
            # 下一步射线起点：中心在视野内就挪到中心；中心出了视野（车贴着一侧线、
            # 另一半车道在画面外）就留在原来的偏移上，沿着看得见的那条线走
            recenter = lab_at(cd_, cx_) != 0
            o = off if recenter else 0.0
            off_exp = 0.0 if recenter else off
            prev_sl = None if sl is None else sl + o
            prev_sr = None if sr is None else sr - o
            if recenter:
                pd, px = cd_, cx_
            # 方向：用最近几个中心点
            if len(Cs) >= 3:
                a0 = Cs[-5] if len(Cs) >= 5 else Cs[0]
                dv0, dv1 = cd_ - a0[0], cx_ - a0[1]
                nv = math.hypot(dv0, dv1)
                if nv > 1e-6:
                    u0, u1 = dv0 / nv, dv1 / nv
                    ang = math.atan2(t0 * u1 - t1 * u0, t0 * u0 + t1 * u1)
                    ang = max(-max_turn, min(max_turn, ang))
                    ca, sa = math.cos(ang), math.sin(ang)
                    t0, t1 = t0 * ca - t1 * sa, t0 * sa + t1 * ca
            if t0 < -0.2 and len(Cs) * step > 0.6:
                res["reason"] = "车道已经掉头"
                break
            pd += step * t0
            px += step * t1
        return res

    def _trim_start(self, C):
        """起步钩子：起点那 3 步的走向，和「后面两段走向线性外推回来」的预测差太多就丢掉。
        用外推而不是直接比较，是为了急弯里正常的航向变化不被误删。最多丢 trim_max 步。"""
        p = self.p
        kmax = int(p["trim_max_steps"])
        if len(C) < kmax + 10:
            return 0
        lim = math.radians(p["trim_angle_deg"])

        def ang(i, j):
            v = C[j] - C[i]
            return math.atan2(v[1], v[0])

        def wrap(x):
            return (x + math.pi) % (2 * math.pi) - math.pi
        k = 0
        while k < kmax:
            a0, a1, a2 = ang(k, k + 3), ang(k + 3, k + 6), ang(k + 6, k + 9)
            pred = a1 - wrap(a2 - a1)
            if abs(wrap(a0 - pred)) <= lim:
                break
            k += 1
        return k

    @staticmethod
    def _smooth_path(C, win):
        """局部二次拟合平滑（端点用最近的完整窗口），win 为奇数步数。"""
        n = len(C)
        if win < 5 or n < win:
            return C
        win |= 1
        h = win // 2
        k = np.arange(-h, h + 1, dtype=float)
        A = np.stack([np.ones_like(k), k, k * k], 1)
        P = np.linalg.pinv(A)                  # 3 × win
        out = np.empty_like(C)
        # 中间点：窗口中心处的拟合值 = 系数行 P[0] 与窗口的卷积（向量化）
        ker = P[0][::-1]
        out[h:n - h, 0] = np.convolve(C[:, 0], ker, "valid")
        out[h:n - h, 1] = np.convolve(C[:, 1], ker, "valid")
        # 两端：用最近的完整窗口的二次式外插
        for c, idx in ((h, range(0, h)), (n - 1 - h, range(n - h, n))):
            coef = P @ C[c - h:c + h + 1]
            for i in idx:
                tt = i - c
                out[i] = coef[0] + coef[1] * tt + coef[2] * tt * tt
        return out

    # ═══════════════════════════════════════════════════════════
    # 对外主函数
    # ═══════════════════════════════════════════════════════════
    def process(self, img):
        p = self.p
        if img.shape[1] != p["img_w"] or img.shape[0] != p["img_h"]:
            img = cv2.resize(img, (p["img_w"], p["img_h"]))
        bev = self.to_bev(img)
        yel, blue, road = self.color_masks(bev)
        out = {
            "bev": bev, "mask": yel.astype(np.uint8) * 255,
            "coefL": None, "coefR": None, "mid_coef": None,
            "mode": 3, "is_lost": True, "reject_reason": "",
            "e_m": 0.0, "theta_rad": 0.0, "curvature": 0.0,
            "lane_width_m": self.w_ref, "quality": 0.0,
            "n_pts": int(yel.sum()), "look_x_m": 0.0, "look_d_m": 0.0,
            "path": np.zeros((0, 2)), "Lpts": np.zeros((0, 2)), "Rpts": np.zeros((0, 2)),
        }
        res = self.march(yel, blue, road)
        C = np.array(res["C"]).reshape(-1, 2)
        Lp = np.array([q for q in res["L"] if q is not None]).reshape(-1, 2)
        Rp = np.array([q for q in res["R"] if q is not None]).reshape(-1, 2)
        side = np.array(res["side"], int)
        out.update(path=C, Lpts=Lp, Rpts=Rp)
        step = p["march_step_m"]
        if len(C) < p["min_steps"]:
            out["reject_reason"] = f"只走出 {len(C)*step:.2f} m 车道（{res['reason']}）"
            self.prev_dir = None
            self.prev_seed_x = float(C[0, 1]) if len(C) else None
            return out

        # ── 去掉起步钩子：刚起步时方向还没对准，最初几个中心点会歪出去一截 ──
        k_trim = self._trim_start(C)
        if k_trim:
            C, side = C[k_trim:], side[k_trim:]
            res["L"], res["R"] = res["L"][k_trim:], res["R"][k_trim:]
        # ── 平滑：Savitzky-Golay 式局部二次拟合（保弯道形状，去 1~2 px 抖动）──
        C = self._smooth_path(C, int(p["smooth_n"]))
        out["path"] = C
        # ── 弧长参数化 ──
        seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
        s = np.r_[0.0, np.cumsum(seg)]
        length = float(s[-1])

        # ── 起点附近一段中心线做参数二次拟合 d(s)、x(s)：
        #    theta、kappa、e 都从同一条拟合曲线上取，彼此一致，也比逐点差分稳 ──
        nf = int(np.searchsorted(s, p["curv_len_m"])) + 1
        nf = max(min(len(C), 4), min(nf, len(C)))
        sf = s[:nf]
        deg = 2 if sf[-1] > 0.15 and nf >= 6 else 1
        cd = np.polyfit(sf, C[:nf, 0], deg)
        cx = np.polyfit(sf, C[:nf, 1], deg)
        d1, x1 = np.polyval(np.polyder(cd), 0.0), np.polyval(np.polyder(cx), 0.0)
        if deg == 2:
            d2, x2 = 2 * cd[0], 2 * cx[0]
        else:
            d2 = x2 = 0.0
        nrm = math.hypot(d1, x1)
        if nrm < 1e-6:
            d1, x1, nrm = 1.0, 0.0, 1.0
        t0 = np.array([d1, x1]) / nrm
        # 端点导数对噪声敏感：theta 改用起点后 head_len 一段的主方向（PCA）
        nh = max(3, min(len(C), int(np.searchsorted(s, p["head_len_m"])) + 1))
        A = C[:nh]
        _, vec = np.linalg.eigh(np.cov((A - A.mean(0)).T))
        th_dir = vec[:, 1]
        if th_dir @ (A[-1] - A[0]) < 0:
            th_dir = -th_dir
        theta = math.atan2(-th_dir[1], th_dir[0])      # 车道往左 → 正
        kappa = (x1 * d2 - d1 * x2) / nrm ** 3          # 往左弯 → 正
        if sf[-1] < 0.08:
            kappa = 0.0
        kappa = float(np.clip(kappa, -p["max_curvature_1pm"], p["max_curvature_1pm"]))
        # 弦方向代表的是这段中点的航向，弯道里要按曲率退回到起点
        theta -= kappa * 0.5 * float(s[nh - 1])

        # ── 起点往往已在相机前 0.3~0.4 m，弯道里那里的车道方向早已转过二三十度。
        #    theta / e 要的是车所在处（d = eval_d）：沿等曲率圆弧倒推回去 ──
        d_e = p["eval_d_m"]
        d0c, x0c = float(np.polyval(cd, 0.0)), float(np.polyval(cx, 0.0))
        if d0c > d_e:
            n_bk = 20
            ds = (d0c - d_e) / n_bk
            pd, px, psi = d0c, x0c, theta
            for _ in range(n_bk):
                c = max(math.cos(psi), 0.3)
                step_s = ds / c
                pd -= step_s * math.cos(psi)
                px += step_s * math.sin(psi)       # 航向左正、x 右正
                # 只在离观测起点 bk_curve_m 以内按曲率倒推，再远就沿切线（看不见的路段不臆测）
                if (d0c - pd) <= p["bk_curve_m"]:
                    psi -= kappa * step_s
            xe, theta = px, psi
        else:
            ss = np.linspace(0.0, sf[-1], 60)
            dd = np.polyval(cd, ss)
            if np.all(np.diff(dd) > 0) and dd[-1] >= d_e:
                xe = float(np.interp(d_e, dd, np.polyval(cx, ss)))
            else:
                xe = x0c
        theta = max(-math.pi, min(math.pi, theta))

        # ── 预瞄点：离相机直线距离 look_d 的第一个中心点 ──
        rng = np.hypot(C[:, 0], C[:, 1])
        il = int(np.argmax(rng >= p["look_d_m"])) if (rng >= p["look_d_m"]).any() else len(C) - 1
        look = C[il]

        # ── 车道宽：双边步的中位数 ──
        both = side == 0
        if both.sum() >= p["min_steps"]:
            idx = np.nonzero(both)[0]
            Ls = np.array([res["L"][i] for i in idx])
            Rs = np.array([res["R"][i] for i in idx])
            w_meas = float(np.median(np.linalg.norm(Rs - Ls, axis=1))) + p["line_width_m"]
            w_meas = float(np.clip(w_meas, p["partner_min_m"], p["partner_max_m"]))
            # 只用「像正常车道」的测量更新先验：宽区、路口的宽度不能带进单边补线
            if abs(w_meas - p["lane_width_m"]) <= p["lane_width_tol_m"]:
                self.w_ref += p["lane_width_alpha"] * (w_meas - self.w_ref)
            lane_w = w_meas
        else:
            lane_w = self.w_ref

        n0, n1, n2 = int((side == 0).sum()), int((side == 1).sum()), int((side == 2).sum())
        # ★ 2026-10-29：mode 抖动会让 kappa 的拟合窗口在两种假设间反复切换，
        #   表现为舵角"抽搐"。加滞回：掉到单边界立即生效，但回到双边要连续
        #   mode_hold（默认 8 帧 ≈ 0.27s）都判成双边才切回。
        mode_raw = 0 if n0 >= max(n1, n2) else (1 if n1 >= n2 else 2)
        if mode_raw == 0:
            self.mode_ok_cnt = getattr(self, "mode_ok_cnt", 0) + 1
        else:
            self.mode_ok_cnt = 0
        prev_mode = getattr(self, "mode_prev", 0)
        need = int(p.get("mode_hold_frames", 8))
        if prev_mode != 0 and self.mode_ok_cnt < need:
            mode = prev_mode          # 还没连续够久 → 沿用单边界
        else:
            mode = mode_raw
        self.mode_prev = mode

        # ── 兼容旧字段：x = f(d) 的二次拟合（只用中心线单调前进的那段）──
        mono = np.r_[True, np.diff(C[:, 0]) > 0]
        stop = int(np.argmin(mono)) if not mono.all() else len(C)
        Cm = C[:max(stop, 2)]
        if len(Cm) >= 3 and np.ptp(Cm[:, 0]) > 0.05:
            deg = 2 if np.ptp(Cm[:, 0]) > 0.3 else 1
            cm = np.polyfit(Cm[:, 0] - self.d0_fit, Cm[:, 1], deg)
            mid = cm if deg == 2 else np.r_[0.0, cm]
        else:
            mid = np.array([0.0, -math.tan(theta), xe])
        half = 0.5 * lane_w

        # ── 质量：走出的长度 + 双边占比 ──
        q_len = min(1.0, length / p["good_len_m"])
        q_both = n0 / max(1, len(side))
        quality = float(q_len * (0.6 + 0.4 * q_both))

        # ★ 2026-10-29 实测（斑马线后右线出画面）：只剩一条线时视觉配错边，
        #   画面是左弯却报 kappa=-0.86（符号反），前馈往右打，车冲出蓝色区。
        #   曲率本质依赖左右线的相对关系，单边界（mode!=0）不可能算准 —— 直接置 0。
        # 2026-10-29 实测 f123->f133（S 弯换向）：theta 从 +24 涨到 +50，kappa 从
        # +1.07 翻成 -1.03（mode=0，两条线都在），舵角因此从 +21.7 甩到 -0.7。
        # 不管几条线，theta 与 kappa 反号且都够大就是不可能的组合 —— kappa 的符号
        # 交给 theta 定（theta 是一阶量，比二阶的曲率可靠）。
        if abs(kappa) > 0.2 and abs(math.degrees(theta)) > 8.0 and theta * kappa < 0.0:
            # 单边界时 kappa 的符号不可靠（实测画面左弯却报 -0.86）。
            # 幅度保留，符号改由 theta（一阶量，更可靠）决定 —— 不能直接清零，
            # 否则弯道里"车没打算转"，直着冲出去。
            kappa = math.copysign(abs(kappa), theta)
            quality *= 0.9

        # ★ 2026-10-30 实测（路口推车 7073 帧）：路口左边有分支赛道时，视觉会把
        #   分支的黄线当本车道左边界，theta 冲到 -100°、kappa 填满、e 到 ±30cm，
        #   车被往左带。kappa 符号翻转 25 次、mode 只有 42% 是双边。
        #   对策：目标跟踪思路 —— 新的中心线必须落在上一帧预测的附近（默认 ±15cm）。
        #   偏离太多说明跳到别的分支了，判丢线，让上层保持转向。
        tol_m = float(p.get("continuity_tol_m", 0.15))
        if self.prev_seed_x is not None and self.prev_dir is not None:
            pred_d = self.prev_seed_x
            if abs(float(xe) - pred_d) > tol_m:
                out.update(is_lost=True,
                           reject_reason="中心线跳变 %.0fcm（可能配到别的分支）"
                                         % ((float(xe) - pred_d) * 100))
                quality *= 0.3
                # 不更新 prev_seed_x，保持上一帧的锚点
                self.prev_seed_x = pred_d if not np.isfinite(pred_d) else pred_d

        out.update({
            "coefL": mid + np.array([0, 0, -half]), "coefR": mid + np.array([0, 0, half]),
            "mid_coef": mid, "mode": mode, "is_lost": False, "reject_reason": "",
            "e_m": float(xe) - p["e_offset_m"],
            "theta_rad": theta - math.radians(p["theta_offset_deg"]),
            "curvature": kappa, "lane_width_m": lane_w, "quality": quality,
            "look_x_m": float(look[1]), "look_d_m": float(look[0]),
            "path_len_m": length,
        })

        limit = min(lane_w / 2 + p["e_margin_m"], p["e_abs_limit_m"])
        if abs(out["e_m"]) > limit:
            out.update(is_lost=True, reject_reason=f"e={out['e_m']*100:+.0f}cm 超出车道")
        elif abs(math.degrees(out["theta_rad"])) > p["theta_limit_deg"]:
            out.update(is_lost=True, reject_reason=f"theta={math.degrees(theta):+.0f}° 超限")
        elif quality < p["quality_min"]:
            out.update(is_lost=True,
                       reject_reason=f"quality={quality:.2f} 太低（{res['reason']}）")

        # ── 下一帧的先验 ──
        self.prev_seed_x = float(C[0, 1])
        self.prev_dir = (float(t0[0]), float(t0[1])) if not out["is_lost"] else None
        return out

    @property
    def d0_fit(self):
        return 0.5 * (self.d_min + self.d_max)

    # 兼容旧代码里用到的 d0
    @property
    def d0(self):
        return self.d0_fit

    # ═══════════════════════════════════════════════════════════
    # 调试图：左 = 原图叠加，右 = 鸟瞰图
    # ═══════════════════════════════════════════════════════════
    def draw_overlay(self, img, r, thick=3):
        """在原图上画：左边界（青）、右边界（橙）、中心线（黄绿）、预瞄点（品红）。"""
        out = img.copy()

        def poly(pts, color, t):
            if len(pts) < 2:
                return
            u, v = self.ground_to_image(pts[:, 0], pts[:, 1])
            ok = np.isfinite(u) & np.isfinite(v)
            q = np.stack([u[ok], v[ok]], 1).astype(np.int32)
            if len(q) >= 2:
                cv2.polylines(out, [q], False, color, t, cv2.LINE_AA)

        for pts, color in ((r["Lpts"], (255, 255, 0)), (r["Rpts"], (0, 140, 255))):
            if len(pts):
                u, v = self.ground_to_image(pts[:, 0], pts[:, 1])
                for a, b in zip(u, v):
                    if np.isfinite(a) and np.isfinite(b):
                        cv2.circle(out, (int(a), int(b)), thick, color, -1)
        poly(r["path"], (0, 255, 120), thick + 1)
        if not r["is_lost"] or len(r["path"]):
            u, v = self.ground_to_image(r["look_d_m"], r["look_x_m"])
            if np.isfinite(u) and np.isfinite(v) and len(r["path"]):
                cv2.circle(out, (int(u), int(v)), 4 * thick, (255, 0, 255), thick)
        return out

    def visualize(self, img, r):
        p = self.p
        if img.shape[1] != p["img_w"] or img.shape[0] != p["img_h"]:
            img = cv2.resize(img, (p["img_w"], p["img_h"]))
        left = self.draw_overlay(img, r)
        right = r["bev"].copy()
        right[~self.bev_valid] = (right[~self.bev_valid] * 0.2).astype(np.uint8)
        right[r["mask"] > 0] = (0, 200, 0)

        def to_px(pts):
            rr, cc = self._rc(pts[:, 0], pts[:, 1])
            return np.stack([cc, rr], 1).astype(np.int32)

        for pts, color in ((r["Lpts"], (255, 255, 0)), (r["Rpts"], (0, 140, 255))):
            for q in to_px(pts) if len(pts) else []:
                cv2.circle(right, tuple(int(v) for v in q), 2, color, -1)
        if len(r["path"]) >= 2:
            cv2.polylines(right, [to_px(r["path"])], False, (0, 255, 255), 2)
            q = to_px(np.array([[r["look_d_m"], r["look_x_m"]]]))[0]
            cv2.circle(right, (int(q[0]), int(q[1])), 7, (255, 0, 255), 2)
        cv2.line(right, (self.Wb // 2, 0), (self.Wb // 2, self.Hb), (255, 255, 255), 1)

        mode_txt = {0: "both", 1: "left", 2: "right", 3: "LOST"}[r["mode"]]
        lines = [f"{mode_txt}  q={r['quality']:.2f}",
                 f"e  {r['e_m']*100:+.1f} cm",
                 f"th {math.degrees(r['theta_rad']):+.1f} deg",
                 f"k  {r['curvature']:+.2f} 1/m",
                 f"w  {r['lane_width_m']:.2f} m"]
        colr = (0, 0, 255) if r["is_lost"] else (255, 255, 255)
        for i, s in enumerate(lines):
            cv2.putText(left, s, (12, 38 + i * 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6)
            cv2.putText(left, s, (12, 38 + i * 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, colr, 2)
        if r["is_lost"] and r["reject_reason"]:
            cv2.putText(left, "LOST", (12, 38 + 5 * 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                        (0, 0, 255), 3)
        h1 = left.shape[0]
        rw = max(1, int(self.Wb * h1 / self.Hb))
        canvas = np.hstack([left, cv2.resize(right, (rw, h1), interpolation=cv2.INTER_NEAREST)])
        if p["debug_scale"] != 1.0:
            canvas = cv2.resize(canvas, None, fx=p["debug_scale"], fy=p["debug_scale"])
        return canvas
