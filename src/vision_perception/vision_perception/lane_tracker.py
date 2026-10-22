# -*- coding: utf-8 -*-
"""
lane_tracker.py —— 车道线巡线核心算法（纯 Python，不依赖 ROS）

设计原则（比"多加几个阈值"重要得多）：
    · **不预设左右**。旧版假设"左线在画面左半、右线在右半"，一进急弯就崩。
      新版把 BEV 里所有黄像素当成一堆点，用 RANSAC 依次找出两条曲线，
      谁左谁右由数据决定。
    · **有效性来自几何，不来自阈值**。地平线以上的像素在物理上就不可能是地面，
      直接按公式算出来剔掉，不需要"调一个参数"。
    · **鲁棒拟合**。用 RANSAC 而不是最小二乘，离群点再多也拉不歪。
    · **物理约束代替魔法数**。判断"这是不是两条线"用"间距 ≈ 车道宽"，
      而不是"距离小于某个神秘常数"。

整体流程：
    原图 ─①裁剪&IPM(带地平线有效性)─→ 鸟瞰图
         ─②颜色分割─→ 黄像素
         ─③坐标转米 (d, x)
         ─④顺序 RANSAC 找两条曲线
         ─⑤按车道宽配对，定出左右
         ─⑥算 e / theta / kappa

符号约定（**全部"左正"**）：
    e_m        车体中心相对车道中心线的横向偏差。车偏左 → 正
    theta_rad  车道前进方向相对车头方向的夹角。车道往左拐 → 正
    curvature  车道曲率（1/米）。向左弯 → 正
    控制层的直觉：三个数哪个是正的，方向盘就往左打。
"""
import math
import warnings

import cv2
import numpy as np

# 两种 import 方式都要支持：
#   在 ROS 包里被当成子模块导入时 → 走相对导入
#   在项目根目录跑 offline_test.py 时 → 走绝对导入
try:
    from .lane_params import DEFAULTS
except ImportError:
    from lane_params import DEFAULTS


def _polyfit2(u, x):
    """二次拟合，顺手压掉"样本几乎共线"时的 RankWarning。

    RANSAC 随机抽 3 个点，偶尔抽到挤在一条线上的，Vandermonde 矩阵条件数很差。
    结果本身仍合法（退化成低阶开口），没必要让警告刷屏。
    """
    try:
        kind = np.exceptions.RankWarning
    except AttributeError:          # numpy < 1.25
        kind = np.RankWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", kind)
        return np.polyfit(u, x, 2)


class LaneTracker:
    # ═══════════════════════════════════════════════════════════
    # 初始化：IPM 查找表只算一次
    # ═══════════════════════════════════════════════════════════
    def __init__(self, overrides=None):
        self.p = dict(DEFAULTS)
        if overrides:
            self.p.update(overrides)
        self._build_ipm()

    # ── IPM：地面 (d, x) → 原图 (u, v) ─────────────────────────
    # 相机坐标 X 右、Y 下、Z 前；相机离地 h，俯角 θ。
    # 地面上「前方 d 米、右方 x 米」的点投影到图像：
    #     u = f·x / (h·sinθ + d·cosθ) + cx
    #     v = f·(h·cosθ − d·sinθ) / (h·sinθ + d·cosθ) + cy
    def _build_ipm(self):
        p = self.p
        h = p["h_m"]
        th = math.radians(p["pitch_deg"])
        W_img, H_img = p["img_w"], p["img_h"]

        vfov = 2 * math.atan(math.tan(math.radians(p["hfov_deg"]) / 2) * H_img / W_img)
        f = (H_img / 2) / math.tan(vfov / 2)
        cx, cy = W_img / 2, H_img / 2
        self.f, self.cx, self.cy = f, cx, cy

        Wb = max(2, int(round(p["bev_width_m"] / p["bev_m_per_px"])))
        Hb = max(2, int(round(p["bev_depth_m"] / p["bev_m_per_px"])))
        mpp, dmin = p["bev_m_per_px"], p["bev_d_min"]
        dmax = dmin + Hb * mpp
        self.Wb, self.Hb, self.mpp = Wb, Hb, mpp
        self.d_min, self.d_max = dmin, dmax

        # 鸟瞰图第 j 行 ↔ 地面距离 d（j=0 最远）
        self.d_of_row = np.linspace(dmax, dmin, Hb)
        # 第 i 列 ↔ 地面横向 x（i=0 最左；中心 = 车体中轴）
        self.x_of_col = (np.arange(Wb) - (Wb - 1) / 2) * mpp
        # 拟合时把距离原点挪到中点，否则 d² 项数值条件很差（会出现 RankWarning）
        self.d0 = 0.5 * (dmin + dmax)

        d = self.d_of_row.reshape(-1, 1)
        x = self.x_of_col.reshape(1, -1)
        denom = h * math.sin(th) + d * math.cos(th)
        yc = h * math.cos(th) - d * math.sin(th)
        u = f * x / denom + cx
        v = f * yc / denom + cy

        # ★ 几何有效性：地平线以上的像素不可能是地面，直接剔掉
        self.horizon_row = cy - f * math.tan(th)
        ok = (denom > 1e-3) & (u >= 0) & (u < W_img) \
             & (v >= self.horizon_row + p["horizon_margin_px"]) \
             & (v < H_img)

        self.map_x = np.where(ok, u, -1).astype(np.float32)
        self.map_y = np.where(ok, v, -1).astype(np.float32)
        self.bev_valid = ok

        # 只保留"至少有 30% 宽度有效"的行，省得算出半行黑的深度范围
        rows_ok = ok.mean(axis=1) > 0.3
        self.valid_rows = np.where(rows_ok)[0]
        self.d_valid_max = (float(self.d_of_row[self.valid_rows[0]])
                            if len(self.valid_rows) else dmin)

    # ── ① 裁剪 + IPM ─────────────────────────────────────────
    def to_bev(self, img):
        y0 = self.p["roi_top"]
        roi = img[y0:self.p["roi_bottom"], :]
        map_y = self.map_y - y0          # 裁了 ROI，行号要减偏移
        return cv2.remap(roi, self.map_x, map_y,
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # ── ② 颜色分割 ───────────────────────────────────────────
    def color_mask(self, bev):
        p = self.p
        hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
        lo = np.array([p["color_h_min"], p["color_s_min"], p["color_v_min"]])
        hi = np.array([p["color_h_max"], 255, 255])
        mask = cv2.inRange(hsv, lo, hi)
        if p["use_white"]:
            lo_w = np.array([0, 0, p["white_v_min"]])
            hi_w = np.array([180, p["white_s_max"], 255])
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo_w, hi_w))
        k1, k2 = p["morph_open_px"], p["morph_close_px"]
        if k1 > 1:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k1, k1), np.uint8))
        if k2 > 1:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k2, k2), np.uint8))
        # 只保留几何上有效的那部分
        mask[~self.bev_valid] = 0
        return mask

    # ── ③ 像素 → 米坐标 ──────────────────────────────────────
    def points_in_meters(self, mask):
        rows, cols = np.nonzero(mask)
        d = self.d_of_row[rows]              # 距离（米）
        x = self.x_of_col[cols]              # 横向（米，右正）
        return np.stack([d, x], axis=1)      # (N, 2)

    # ── ④ RANSAC 拟合二次曲线 x = a·u² + b·u + c（u = d − d0）─────
    def ransac_poly(self, pts, rng):
        """返回 (coef, inlier_mask) 或 (None, None)。

        ⚠ 性能关键：内点判定用**拉格朗日插值**做纯算术求值，
        绝不能在循环里调 np.polyfit —— 3 个点的最小二乘会走 LAPACK/BLAS，
        而 ARM 上的多线程 BLAS 对这么小的矩阵是纯亏本：
        实测 80 次调用能把一帧从 42 ms 拖到 290 ms（7 倍）。
        """
        p = self.p
        n_all = len(pts)
        if n_all < p["min_points_curve"]:
            return None, None
        thr = p["ransac_thresh_m"]

        # ⚠ 性能：点数太多时先降采样。RANSAC 只需几千个点就能找到主趋势，
        #   上万个点只会让每轮内点统计变慢，精度没有提升。
        cap = p["max_ransac_points"]
        if n_all > cap:
            sel = np.unique(rng.integers(0, n_all, cap))   # O(cap)，不能用 choice
        else:
            sel = None
        sub = pts if sel is None else pts[sel]
        n = len(sub)

        u = sub[:, 0] - self.d0          # 平移后的距离，数值更稳
        x = sub[:, 1]
        best_cnt, best_idx = -1, None

        for _ in range(p["ransac_iters"]):
            # ⚠ 必须用 integers 而不是 choice(replace=False)：
            #   choice 在 replace=False 时内部会生成整个数组的随机排列，复杂度 O(n)。
            #   n 上万时 80 次迭代能吃掉半秒 —— 实测就是这个把一帧拖到 500 ms。
            idx = rng.integers(0, n, 3)
            u1, u2, u3 = u[idx]
            d12, d13, d23 = u1 - u2, u1 - u3, u2 - u3
            if min(abs(d12), abs(d13), abs(d23)) < 0.05:
                continue                  # 三点太挤，插值会爆
            x1, x2, x3 = x[idx]
            # 过三点的抛物线，用拉格朗日基函数直接求值（纯加减乘除，不碰 BLAS）
            l1 = (u - u2) * (u - u3) / (d12 * d13)
            l2 = (u - u1) * (u - u3) / (-d12 * d23)
            l3 = (u - u1) * (u - u2) / (d13 * d23)
            pred = x1 * l1 + x2 * l2 + x3 * l3
            cnt = int((np.abs(pred - x) < thr).sum())
            if cnt > best_cnt:
                best_cnt, best_idx = cnt, idx

        if best_idx is None or best_cnt < p["min_points_curve"]:
            return None, None

        # 只在这一步做真正的最小二乘（此时点已很多，BLAS 才划算）
        coef = _polyfit2(u, x)            # 先粗拟合
        for _ in range(2):                # 两轮迭代剔除离群点
            resid = np.abs(np.polyval(coef, u) - x)
            inl = resid < thr
            if inl.sum() < p["min_points_curve"]:
                return None, None
            coef = _polyfit2(u[inl], x[inl])

        resid = np.abs(np.polyval(coef, u) - x)
        mask_sub = resid < thr
        # 把内点掩膜还原到**原始点数**，调用方要用它挑"剩下的点"去拟合第二条线
        if sel is not None:
            full = np.zeros(n_all, dtype=bool)
            full[sel] = mask_sub
            return coef, full
        return coef, mask_sub

    # ── 便捷：在距离 d 处的横坐标 / 切向 ──────────────────────
    def poly_at(self, coef, d):
        return float(np.polyval(coef, d - self.d0))

    def poly_deriv(self, coef, d):
        u = d - self.d0
        return float(2 * coef[0] * u + coef[1])

    # ── ⑤ 按车道宽配对 ───────────────────────────────────────
    def pair_curves(self, curves, d_ref):
        """从若干条曲线里挑出「间距最接近车道宽」的一对，并定出左右。"""
        p = self.p
        want = p["lane_width_m"]
        best = None
        for i in range(len(curves)):
            for j in range(i + 1, len(curves)):
                xi = self.poly_at(curves[i], d_ref)
                xj = self.poly_at(curves[j], d_ref)
                gap = abs(xj - xi)
                score = abs(gap - want)
                if best is None or score < best[0]:
                    best = (score, i, j, gap)
        if best is None:
            return None
        score, i, j, gap = best

        # 间距偏差超过容忍度，说明这俩不是同一条车道的两条边
        if score > want * p["pair_gap_tol"]:
            return None
        # 谁在左边
        xi = self.poly_at(curves[i], d_ref)
        xj = self.poly_at(curves[j], d_ref)
        return (i, j) if xi < xj else (j, i)

    # ═══════════════════════════════════════════════════════════
    # 对外主函数
    # ═══════════════════════════════════════════════════════════
    def process(self, img):
        p = self.p
        rng = np.random.default_rng(p["ransac_seed"])

        bev = self.to_bev(img)
        mask = self.color_mask(bev)
        pts = self.points_in_meters(mask)

        out = {
            "bev": bev, "mask": mask, "pts": pts,
            "coefL": None, "coefR": None, "mid_coef": None,
            "mode": 3, "is_lost": True, "reject_reason": "无有效像素",
            "e_m": 0.0, "theta_rad": 0.0, "curvature": 0.0,
            "lane_width_m": p["lane_width_m"], "quality": 0.0,
            "n_pts": len(pts),
        }
        if len(pts) < p["min_points_curve"]:
            return out

        # 顺序 RANSAC：找第一条，剔掉它的内点，再找第二条
        c1, in1 = self.ransac_poly(pts, rng)
        if c1 is None:
            return out
        rest = pts[~in1]
        c2 = None
        if len(rest) >= p["min_points_curve"]:
            c2, _ = self.ransac_poly(rest, rng)

        d_ref = p["eval_d_m"]
        d_head = p["heading_d_m"]
        curves = [c1] if c2 is None else [c1, c2]
        pair = self.pair_curves(curves, d_ref) if len(curves) == 2 else None

        # ── 数据支撑范围：拟合只在有点的地方可信，不能凭空调外推 ──
        inl = pts[in1]
        d_lo, d_hi = float(inl[:, 0].min()), float(inl[:, 0].max())
        span = d_hi - d_lo
        out["d_span"] = span
        if span < p["min_span_m"]:
            out.update(reject_reason=f"有效点只跨 {span:.2f}m，太短，拟合不可信")
            return out
        # 评估点夹到数据范围内（外推不可信）
        d_ref = min(max(d_ref, d_lo), d_hi)
        d_head = min(max(d_head, d_lo), d_hi)

        if pair is not None:
            iL, iR = pair
            coefL, coefR = curves[iL], curves[iR]
            mode = 0
            lane_w = abs(self.poly_at(coefR, d_ref) - self.poly_at(coefL, d_ref))
        else:
            # 只有一条线：靠"车道宽"推另一条。哪边？看这条线在车轴的哪一侧。
            c = c1
            xc = self.poly_at(c, d_ref)
            if xc < 0:          # 这条线在车的左边 → 它是左线
                coefL = c
                coefR = c + np.array([0, 0, p["lane_width_m"]])
                mode = 1
            else:               # 在右边 → 右线
                coefR = c
                coefL = c - np.array([0, 0, p["lane_width_m"]])
                mode = 2
            lane_w = p["lane_width_m"]

        mid_coef = (coefL + coefR) / 2
        # 用两条线的平均曲率更稳；单边时就是它自己的曲率
        if mode == 0:
            mid_coef[0] = (coefL[0] + coefR[0]) / 2

        xc_eval = self.poly_at(mid_coef, d_ref)
        dd = self.poly_deriv(mid_coef, d_head)

        out.update({
            "coefL": coefL, "coefR": coefR, "mid_coef": mid_coef,
            "mode": mode, "is_lost": False, "reject_reason": "",
            "e_m": xc_eval - p["e_offset_m"],
            "theta_rad": (-math.atan(dd)
                          - math.radians(p["theta_offset_deg"])),
            "curvature": -2 * mid_coef[0] / (1 + dd * dd) ** 1.5,
            "lane_width_m": lane_w,
            # 置信度 = 有效像素数 × 数据跨度，两个都重要：
            #   点再多但都挤在一小段，曲率照样拟合不准
            "quality": float(min(1.0, (len(pts) / (0.015 * self.Wb * self.Hb))
                                 * min(1.0, span / (self.d_max - self.d_min)))),
        })

        # ── 物理合理性：车不可能离车道中心超过车道半宽 + 一点余量 ──
        # 这不是"拍脑袋的阈值"，是个几何事实：|e| > 半宽 + 15cm 就意味着已经冲出边界。
        limit = lane_w / 2 + p["e_margin_m"]
        if abs(out["e_m"]) > limit:
            out.update(is_lost=True, quality=0.0,
                       reject_reason=f"e={out['e_m']*100:+.0f}cm > 半宽+余量 {limit*100:.0f}cm")
        elif abs(math.degrees(out["theta_rad"])) > p["theta_limit_deg"]:
            out.update(is_lost=True, quality=0.0,
                       reject_reason=f"theta={math.degrees(out['theta_rad']):+.0f}° 超限")
        elif abs(out["curvature"]) > p["max_curvature_1pm"]:
            # 弯得比车能转的还急，肯定是拟合跑飞了（几何上不可能）
            out.update(is_lost=True, quality=0.0,
                       reject_reason=f"kappa={out['curvature']:+.1f} 超过车的最小转弯能力")
        return out

    # ═══════════════════════════════════════════════════════════
    # 调试图
    # ═══════════════════════════════════════════════════════════
    def visualize(self, img, r):
        p = self.p
        Wb, Hb = self.Wb, self.Hb

        left = img.copy()
        cv2.rectangle(left, (0, p["roi_top"]), (left.shape[1] - 1, p["roi_bottom"]),
                      (80, 200, 80), 2)
        cv2.putText(left, "ROI", (10, p["roi_top"] + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 200, 80), 2)
        # 地平线标出来
        hr = int(self.horizon_row)
        if 0 <= hr < left.shape[0]:
            for x in range(0, left.shape[1], 26):
                cv2.line(left, (x, hr), (x + 13, hr), (60, 60, 230), 2)
            cv2.putText(left, "horizon", (10, max(hr - 8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 230), 2)

        right = r["bev"].copy()
        inv = ~self.bev_valid
        right[inv] = (right[inv] * 0.15).astype(np.uint8)
        right[r["mask"] > 0] = (0, 255, 0)

        def draw(coef, color, thick=2):
            if coef is None:
                return
            ds = np.linspace(self.d_min, self.d_max, 160)
            xs = np.polyval(coef, ds - self.d0)
            cols = xs / self.mpp + (Wb - 1) / 2
            rows = (self.d_max - ds) / (self.d_max - self.d_min) * (Hb - 1)
            ok = (cols >= 0) & (cols < Wb) & (rows >= 0) & (rows < Hb)
            pts = np.stack([cols[ok], rows[ok]], axis=1).astype(np.int32)
            if len(pts) > 1:
                cv2.polylines(right, [pts], False, color, thick)

        draw(r["coefL"], (255, 255, 0))
        draw(r["coefR"], (0, 200, 255))
        draw(r["mid_coef"], (0, 255, 255), 2)

        cv2.line(right, (Wb // 2, 0), (Wb // 2, Hb), (255, 255, 255), 1)
        row_e = int((self.d_max - p["eval_d_m"]) / (self.d_max - self.d_min) * (Hb - 1))
        if 0 <= row_e < Hb:
            cv2.line(right, (0, row_e), (Wb, row_e), (200, 200, 200), 1)

        mode_txt = {0: "双边", 1: "仅左", 2: "仅右", 3: "丢线"}[r["mode"]]
        lines = [
            f"mode    {mode_txt}",
            f"e       {r['e_m']*100:+.1f} cm   (左正)",
            f"theta   {math.degrees(r['theta_rad']):+.2f} deg  (左正)",
            f"kappa   {r['curvature']:+.3f} 1/m   (左正)",
            f"width   {r['lane_width_m']:.2f} m",
            f"quality {r['quality']:.2f}   pts {r.get('n_pts', 0)}",
        ]
        for i, s in enumerate(lines):
            cv2.putText(right, s, (8, 26 + i * 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255) if r["is_lost"] else (255, 255, 255), 2)
        if r["is_lost"] and r.get("reject_reason"):
            cv2.putText(right, r["reject_reason"][:34], (8, 26 + 6 * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 255), 2)

        h1 = left.shape[0]
        rw = max(1, int(right.shape[1] * h1 / right.shape[0]))
        canvas = np.hstack([left, cv2.resize(right, (rw, h1),
                                             interpolation=cv2.INTER_NEAREST)])
        if p["debug_scale"] != 1.0:
            canvas = cv2.resize(canvas, None, fx=p["debug_scale"], fy=p["debug_scale"])
        return canvas


if __name__ == "__main__":
    trk = LaneTracker()
    print(f"焦距 f = {trk.f:.1f} px，地平线在第 {trk.horizon_row:.0f} 行")
    print(f"鸟瞰图 {trk.Wb}x{trk.Hb}px = {trk.Wb*trk.mpp:.2f}m x "
          f"{trk.d_max-trk.d_min:.2f}m")
    print(f"有效地面最远 {trk.d_valid_max:.2f} m（再远就采到地平线以上了）")
    for d in (trk.d_min, trk.p["eval_d_m"], trk.p["heading_d_m"], trk.d_max):
        row = (trk.d_max - d) / (trk.d_max - trk.d_min) * (trk.Hb - 1)
        print(f"  d={d:.2f}m -> 第 {row:6.1f} 行  "
              f"{'OK' if 0 <= row < trk.Hb else '超出鸟瞰图'}  "
              f"{'有效' if d <= trk.d_valid_max else '★ 超过有效地面距离'}")
