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

        # ★ 相机偏航修正：相机镜头方向不一定和车头方向一致（实测装了约 9°）
        #   把「车体坐标 (d, x)」先转到「相机坐标 (dc, xc)」，再做针孔投影。
        #   yaw_deg > 0 表示相机相对车头向右偏。
        #   不做这一步的话，鸟瞰图整体会被转一个角度，theta 就会有一个固定偏置，
        #   而固定的 theta 偏置会让 Stanley 一直往一个方向打。
        yaw = math.radians(p["yaw_deg"])
        if abs(yaw) > 1e-6:
            dc = d * math.cos(yaw) + x * math.sin(yaw)
            xc = -d * math.sin(yaw) + x * math.cos(yaw)
        else:
            dc, xc = d, x

        denom = h * math.sin(th) + dc * math.cos(th)
        yc = h * math.cos(th) - dc * math.sin(th)
        u = f * xc / denom + cx
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
            pred = self._lagrange(u, x, idx)
            if pred is None:
                continue                  # 三点太挤，插值会爆
            cnt = int((np.abs(pred - x) < thr).sum())
            if cnt > best_cnt:
                best_cnt, best_idx = cnt, idx

        if best_idx is None or best_cnt < p["min_points_curve"]:
            return None, None

        # ── 关键：用**胜出的那三个点的内点**做初始最小二乘，而不是全部点 ──
        #   全点拟合会被上万离群点拖飞，之后内点筛选就再也回不来了。
        #   （这个 bug 表现是"有一万多个黄像素却凑不出线"，正是诊断字段抓出来的）
        pred = self._lagrange(u, x, best_idx)
        inl = np.abs(pred - x) < thr
        if inl.sum() < p["min_points_curve"]:
            return None, None
        coef = _polyfit2(u[inl], x[inl])

        for _ in range(2):                # 两轮迭代进一步剔除离群点
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

    @staticmethod
    def _lagrange(u, x, idx):
        """过 idx 三个点的抛物线在所有 u 上的取值（纯算术，不碰 BLAS）。

        返回 None 表示这三点太挤，插值数值上不可信（RANSAC 里直接跳过这一轮）。
        """
        u1, u2, u3 = u[idx]
        d12, d13, d23 = u1 - u2, u1 - u3, u2 - u3
        if min(abs(d12), abs(d13), abs(d23)) < 0.05:
            return None
        x1, x2, x3 = x[idx]
        l1 = (u - u2) * (u - u3) / (d12 * d13)
        l2 = (u - u1) * (u - u3) / (-d12 * d23)
        l3 = (u - u1) * (u - u2) / (d13 * d23)
        return x1 * l1 + x2 * l2 + x3 * l3

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
            "mode": 3, "is_lost": True,
            "reject_reason": "鸟瞰图里一个黄像素都没有（没对着赛道/相机没图/阈值太严）",
            "e_m": 0.0, "theta_rad": 0.0, "curvature": 0.0,
            "lane_width_m": p["lane_width_m"], "quality": 0.0,
            "n_pts": len(pts),
        }
        if len(pts) < p["min_points_curve"]:
            out["reject_reason"] = (f"黄像素仅 {len(pts)} 个，"
                                    f"少于要求的 {p['min_points_curve']} 个")
            return out

        # 顺序 RANSAC：找第一条，剔除它，再找第二条
        c1, in1 = self.ransac_poly(pts, rng)
        if c1 is None:
            out["reject_reason"] = (f"有 {len(pts)} 个黄像素，"
                                    f"但凑不出连续的线（可能都是噪点/其他黄色物体）")
            return out

        # ★ 剔除第一条线时要用「线半宽」，不能用内点阈值
        #   黄线本身约 10 cm 宽，而内点阈值只有 3 cm。
        #   如果只按 3 cm 剔除，同一条线的剩余像素会被第二次 RANSAC
        #   当成「另一条线」拟合出来 —— 表现为两条线间距 0.28 m 而不是 0.54 m，
        #   于是配对被拒，退化成单边模式。2026-10-23 从数据里查出来的。
        dist1 = np.abs(np.polyval(c1, pts[:, 0] - self.d0) - pts[:, 1])
        rest = pts[dist1 > p["line_half_width_m"]]
        c2 = None
        if len(rest) >= p["min_points_curve"]:
            c2, _ = self.ransac_poly(rest, rng)

        d_ref = p["eval_d_m"]
        d_head = p["heading_d_m"]

        # ── 数据支撑范围：拟合只在有点的地方可信，不能凭空调外推 ──
        inl = pts[in1]
        d_lo, d_hi = float(inl[:, 0].min()), float(inl[:, 0].max())
        span = d_hi - d_lo
        out["d_span"] = span
        if span < p["min_span_m"]:
            out.update(reject_reason=f"有效点只跨 {span:.2f}m，太短，拟合不可信")
            return out
        # ★ 必须在配对之前把评估点夹进数据范围：
        #   否则配对用 0.6 m 判断间距，宽度却算在夹过的距离上，两者不一致，
        #   会出现「间距 0.14 m 却判成双边」的情况（2026-10-23 从效果图里发现）。
        d_ref = min(max(d_ref, d_lo), d_hi)
        d_head = min(max(d_head, d_lo), d_hi)

        curves = [c1] if c2 is None else [c1, c2]
        pair = self.pair_curves(curves, d_ref) if len(curves) == 2 else None

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


def _print_geometry(trk):
    """打印几何自检表 —— 标定的时候对着这张表看。

    注意这里给的是**原图行号**（不是鸟瞰图行号），因为标定时你是在原图上量位置。
    """
    p = trk.p
    h = p["h_m"]
    th = math.radians(p["pitch_deg"])
    print(f"焦距 f = {trk.f:.1f} px，地平线在第 {trk.horizon_row:.0f} 行")
    print(f"鸟瞰图 {trk.Wb}x{trk.Hb}px = {trk.Wb*trk.mpp:.2f}m x "
          f"{trk.d_max-trk.d_min:.2f}m（车前 {trk.d_min:.2f} ~ {trk.d_max:.2f} m）")
    print()
    print("  地面距离 → 原图第几行（★ 标定时用这张表对照）")
    print("  " + "-" * 46)
    for d in (0.30, p["eval_d_m"], 0.80, p["heading_d_m"], 1.50, 2.00, 3.00):
        Z = h * math.sin(th) + d * math.cos(th)
        Yc = h * math.cos(th) - d * math.sin(th)
        row = trk.cy + trk.f * Yc / Z
        mark = ""
        if row < trk.horizon_row + p["horizon_margin_px"]:
            mark = "  ★ 在地平线以上，无效"
        elif row >= p["img_h"]:
            mark = "  ★ 超出画面"
        elif row < p["roi_top"]:
            mark = "  ★ 被 ROI 裁掉了（roi_top=%d）" % p["roi_top"]
        print(f"    {d:4.2f} m  →  第 {row:6.1f} 行{mark}")


if __name__ == "__main__":
    _print_geometry(LaneTracker())
