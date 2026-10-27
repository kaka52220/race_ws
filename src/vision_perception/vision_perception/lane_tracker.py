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
        self.last_good = None        # 最近一帧可信输出，用于短时遮挡保持
        self.hold_count = 0

    def reset(self):
        self.w_ref = float(self.p["lane_width_m"])
        self.prev_seed_x = None
        self.prev_dir = None
        self.last_good = None
        self.hold_count = 0

    def _hold_last_good(self, out):
        p = self.p
        if self.last_good is None or self.hold_count >= int(p.get("occlusion_hold_frames", 0)):
            return out
        held = dict(self.last_good)
        held.update(bev=out["bev"], mask=out["mask"], n_pts=out["n_pts"],
                    tracking=True, is_lost=False,
                    quality=float(p.get("occlusion_hold_quality", 0.25)),
                    reject_reason="挡板短时遮挡，沿用上一条可信中心线")
        self.hold_count += 1
        return held

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
        return (yel > 0) & self.bev_valid, (blue > 0) & self.bev_valid

    # ═══════════════════════════════════════════════════════════
    # 种子：最近处、离上一帧中心最近的一块路面
    # ═══════════════════════════════════════════════════════════
    def _find_seed(self, yel, blue):
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
    def _ray(self, yel, P, n, reach_px):
        """返回 (距离米 或 None, 状态)。状态：'hit' / 'fov'（先出了视野）/ 'none'。"""
        s = np.arange(1, reach_px + 1) * self.mpp
        d = P[0] + s * n[0]
        x = P[1] + s * n[1]
        r, c = self._rc(d, x)
        r = np.round(r).astype(int)
        c = np.round(c).astype(int)
        inside = (r >= 0) & (r < self.Hb) & (c >= 0) & (c < self.Wb)
        valid = np.zeros(len(s), bool)
        hit = np.zeros(len(s), bool)
        valid[inside] = self.bev_valid[r[inside], c[inside]]
        hit[inside] = yel[r[inside], c[inside]]
        first_hit = np.argmax(hit) if hit.any() else None
        first_bad = np.argmax(~valid) if (~valid).any() else None
        if first_hit is not None and (first_bad is None or first_hit < first_bad):
            return float(s[first_hit]) - 0.5 * self.mpp, "hit"
        if first_bad is not None:
            return None, "fov"
        return None, "none"

    def _is_yel(self, yel, P):
        r, c = self._rc(P[0], P[1])
        r, c = int(round(r)), int(round(c))
        if 0 <= r < self.Hb and 0 <= c < self.Wb:
            return bool(yel[r, c])
        return False

    def _is_valid(self, P):
        r, c = self._rc(P[0], P[1])
        r, c = int(round(r)), int(round(c))
        return 0 <= r < self.Hb and 0 <= c < self.Wb and bool(self.bev_valid[r, c])

    # ═══════════════════════════════════════════════════════════
    # 射线行进：沿车道走出中心线
    # ═══════════════════════════════════════════════════════════
    def march(self, yel, blue):
        p = self.p
        res = {"C": [], "L": [], "R": [], "side": [], "reason": ""}
        seed = self._find_seed(yel, blue)
        if seed is None:
            res["reason"] = "近处找不到路面（种子丢失）"
            return res
        d0, x0 = seed
        t = self._init_dir(yel, d0, x0)
        P = np.array([d0, x0], float)

        lw = p["line_width_m"]
        w_in = self.w_ref - lw                          # 内沿到内沿
        w_in_min = p["partner_min_m"] - lw
        w_in_max = p["partner_max_m"] - lw
        w_in_wide_max = p.get("wide_lane_max_m", p["partner_max_m"]) - lw
        reach_px = int((max(w_in_max, w_in_wide_max) + 0.05) / self.mpp)
        step = p["march_step_m"]
        n_steps = int(p["march_len_m"] / step)
        jump = p["edge_jump_m"]
        max_turn = p["max_curvature_1pm"] * step * 1.5  # 每步方向最多转多少（弧度）
        skip_max = int(p["block_stop_m"] / step)
        prev_sl = prev_sr = None
        skip = 0
        for k in range(n_steps):
            if not self._is_valid(P):
                res["reason"] = "中心线走出相机视野"
                break
            if self._is_yel(yel, P):
                # 中心压在黄线上（路口横向虚线、起步时车身压线）→ 往前跨
                skip += 1
                if skip > skip_max:
                    res["reason"] = "前方被黄线挡住"
                    break
                P = P + step * t
                continue
            nr = np.array([-t[1], t[0]])                # 右法向
            sl, stl = self._ray(yel, P, -nr, reach_px)
            sr, str_ = self._ray(yel, P, nr, reach_px)
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
                    P = P + step * t
                    continue
                if w > w_in_max:
                    # 挡板区可能是真正的宽车道：只在两侧射线都连续、
                    # 且宽度仍在物理上限内时保留双边中心；固定 54 cm
                    # 先验不被改写，离开宽区后自动恢复。
                    if p.get("wide_lane_enable", False) and w <= w_in_wide_max:
                        res["wide_count"] = res.get("wide_count", 0) + 1
                    else:
                        # 太宽：留下离「当前宽度估计」更吻合的那一侧
                        if abs(sl - w_in / 2) <= abs(sr - w_in / 2):
                            sr = None
                        else:
                            sl = None
            if sl is None and sr is None:
                skip += 1
                # 起步阶段（最近处视野很窄，两条线都还没进画面）允许多走一段
                lim = skip_max if res["C"] else int(p["start_search_m"] / step)
                if skip > lim:
                    res["reason"] = "两侧边界都看不到了"
                    break
                P = P + step * t
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
            if len(res["C"]) >= 3:
                lim_off = p["max_off_step_m"]
                off = max(-lim_off, min(lim_off, off))
            C = P + off * nr
            res["C"].append(C)
            res["L"].append(P - sl * nr if sl is not None else None)
            res["R"].append(P + sr * nr if sr is not None else None)
            res["side"].append(side)
            # 下一步射线起点：中心在视野内就挪到中心；中心出了视野（车贴着一侧线、
            # 另一半车道在画面外）就留在原来的偏移上，沿着看得见的那条线走
            recenter = self._is_valid(C)
            o = off if recenter else 0.0
            prev_sl = None if sl is None else sl + o
            prev_sr = None if sr is None else sr - o
            base = C if recenter else P
            # 方向：用最近几个中心点
            if len(res["C"]) >= 3:
                a = np.array(res["C"][-5:])
                dv = a[-1] - a[0]
                nv = np.linalg.norm(dv)
                if nv > 1e-6:
                    t_new = dv / nv
                    ang = math.atan2(t[0] * t_new[1] - t[1] * t_new[0], float(t @ t_new))
                    ang = max(-max_turn, min(max_turn, ang))
                    ca, sa = math.cos(ang), math.sin(ang)
                    t = np.array([t[0] * ca - t[1] * sa, t[0] * sa + t[1] * ca])
            if t[0] < -0.2 and len(res["C"]) * step > 0.6:
                res["reason"] = "车道已经掉头"
                break
            P = base + step * t
        return res

    # ═══════════════════════════════════════════════════════════
    # 对外主函数
    # ═══════════════════════════════════════════════════════════
    def process(self, img):
        p = self.p
        if img.shape[1] != p["img_w"] or img.shape[0] != p["img_h"]:
            img = cv2.resize(img, (p["img_w"], p["img_h"]))
        bev = self.to_bev(img)
        yel, blue = self.color_masks(bev)
        out = {
            "bev": bev, "mask": yel.astype(np.uint8) * 255,
            "coefL": None, "coefR": None, "mid_coef": None,
            "mode": 3, "is_lost": True, "reject_reason": "",
            "e_m": 0.0, "theta_rad": 0.0, "curvature": 0.0,
            "lane_width_m": self.w_ref, "quality": 0.0,
            "n_pts": int(yel.sum()), "look_x_m": 0.0, "look_d_m": 0.0,
            "path": np.zeros((0, 2)), "Lpts": np.zeros((0, 2)), "Rpts": np.zeros((0, 2)),
        }
        res = self.march(yel, blue)
        C = np.array(res["C"]).reshape(-1, 2)
        Lp = np.array([q for q in res["L"] if q is not None]).reshape(-1, 2)
        Rp = np.array([q for q in res["R"] if q is not None]).reshape(-1, 2)
        side = np.array(res["side"], int)
        out.update(path=C, Lpts=Lp, Rpts=Rp)
        step = p["march_step_m"]
        if len(C) < p["min_steps"]:
            out["reject_reason"] = f"只走出 {len(C)*step:.2f} m 车道（{res['reason']}）"
            held = self._hold_last_good(out)
            if not held["is_lost"]:
                return held
            self.prev_dir = None
            self.prev_seed_x = float(C[0, 1]) if len(C) else None
            return out

        # ── 平滑：射线每步都有 1~2 个像素的抖动，滑动平均一下再求导 ──
        k_sm = int(p["smooth_n"])
        if k_sm > 1 and len(C) > k_sm:
            pad = k_sm // 2
            Cp = np.vstack([np.repeat(C[:1], pad, 0), C, np.repeat(C[-1:], pad, 0)])
            ker = np.ones(k_sm) / k_sm
            C = np.stack([np.convolve(Cp[:, 0], ker, "valid"),
                          np.convolve(Cp[:, 1], ker, "valid")], 1)
            out["path"] = C
        # ── 弧长参数化 ──
        seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
        s = np.r_[0.0, np.cumsum(seg)]
        length = float(s[-1])

        # ── 起点附近的方向（用前 head_len_m 弧长做直线拟合，抗噪）──
        nh = max(3, int(np.searchsorted(s, p["head_len_m"])) + 1)
        A = C[:nh]
        cov = np.cov((A - A.mean(0)).T)
        w_, vec = np.linalg.eigh(cov)
        t0 = vec[:, 1]
        if t0 @ (A[-1] - A[0]) < 0:
            t0 = -t0
        theta = math.atan2(-t0[1], t0[0])              # 车道往左 → 正

        # ── e：中心线在 d = eval_d 处的横向位置（沿起点方向外推/内插）──
        d_e = p["eval_d_m"]
        if C[0, 0] >= d_e or abs(t0[0]) < 0.2:
            xe = C[0, 1] + (d_e - C[0, 0]) * (t0[1] / max(t0[0], 0.2))
        else:
            k = np.searchsorted(C[:, 0], d_e) if np.all(np.diff(C[:nh, 0]) > 0) else 0
            xe = float(np.interp(d_e, C[:nh, 0], C[:nh, 1])) if k else C[0, 1]

        # ── 曲率：航向角对弧长的斜率（前 curv_len_m）──
        psi = np.unwrap(np.arctan2(-np.gradient(C[:, 1]), np.gradient(C[:, 0])))
        nc = max(5, int(np.searchsorted(s, p["curv_len_m"])) + 1)
        nc = min(nc, len(C))
        if s[nc - 1] > 0.08:
            kappa = float(np.polyfit(s[:nc], psi[:nc], 1)[0])
        else:
            kappa = 0.0
        kappa = float(np.clip(kappa, -p["max_curvature_1pm"], p["max_curvature_1pm"]))

        # ── 预瞄点：离相机直线距离 look_d 的第一个中心点 ──
        rng = np.hypot(C[:, 0], C[:, 1])
        il = int(np.argmax(rng >= p["look_d_m"])) if (rng >= p["look_d_m"]).any() else len(C) - 1
        look = C[il]

        # ── 车道宽：双边步的中位数 ──
        both = side == 0
        wide_count = int(res.get("wide_count", 0))
        wide_stable = wide_count >= int(p.get("wide_lane_min_steps", 6))
        if both.sum() >= p["min_steps"] and (wide_count == 0 or wide_stable):
            idx = np.nonzero(both)[0]
            Ls = np.array([res["L"][i] for i in idx])
            Rs = np.array([res["R"][i] for i in idx])
            w_meas = float(np.median(np.linalg.norm(Rs - Ls, axis=1))) + p["line_width_m"]
            w_meas = float(np.clip(w_meas, p["partner_min_m"],
                                   p.get("wide_lane_max_m", p["partner_max_m"])))
            self.w_ref += p["lane_width_alpha"] * (w_meas - self.w_ref)
            # 普通区域对外仍报告固定 54 cm；只有连续确认的宽区才临时报告实测宽度。
            lane_w = w_meas if wide_stable else self.w_ref
        else:
            lane_w = self.w_ref

        n0, n1, n2 = int((side == 0).sum()), int((side == 1).sum()), int((side == 2).sum())
        mode = 0 if n0 >= max(n1, n2) else (1 if n1 >= n2 else 2)

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

        out.update({
            "coefL": mid + np.array([0, 0, -half]), "coefR": mid + np.array([0, 0, half]),
            "mid_coef": mid, "mode": mode, "is_lost": False, "reject_reason": "",
            "e_m": float(xe) - p["e_offset_m"],
            "theta_rad": theta - math.radians(p["theta_offset_deg"]),
            "curvature": kappa, "lane_width_m": lane_w, "quality": quality,
            "look_x_m": float(look[1]), "look_d_m": float(look[0]),
            "path_len_m": length,
        })

        if wide_count and not wide_stable:
            out.update(is_lost=True,
                       reject_reason=f"宽区仅确认 {wide_count} 步，拒绝一次性改中心")

        limit = min(lane_w / 2 + p["e_margin_m"], p["e_abs_limit_m"])
        if abs(out["e_m"]) > limit:
            out.update(is_lost=True, reject_reason=f"e={out['e_m']*100:+.0f}cm 超出车道")
        elif abs(math.degrees(out["theta_rad"])) > p["theta_limit_deg"]:
            out.update(is_lost=True, reject_reason=f"theta={math.degrees(theta):+.0f}° 超限")
        elif quality < p["quality_min"]:
            out.update(is_lost=True,
                       reject_reason=f"quality={quality:.2f} 太低（{res['reason']}）")

        if out["is_lost"]:
            held = self._hold_last_good(out)
            if not held["is_lost"]:
                return held
        else:
            self.last_good = dict(out)
            self.hold_count = 0

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
