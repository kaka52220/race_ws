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
        self.prev_pair = None      # 上一帧的双边拟合结果，用作本帧搜索先验
        self.prev_mid = None       # 上一帧的车道中线，用于跳变剔除
        self.prev_theta = None     # 上一帧的航向角，用于跳变剔除
        # 车道宽是「状态」不是常数：这条赛道 1→2 直道 0.54 m，
        # 3→4 顶部回旋区和 5/6 中间广场宽得多。每帧实测后用 EMA 更新。
        self.w_ref = float(self.p["lane_width_m"])

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

    # ── 在「距第一条线 want ± band」的带子里，左右各拟合一条候选 ──
    def _fit_line(self, pts, rng):
        """拟合一条线，并验证它**真的是线**（点数够 + 跨度够）。

        为什么必须验证：
            之前配对线完全没有质量检查，只要带子里有几粒黄点就能拟合出一条"线"。
            实测（2026-10-24）：左转时画面里只有右边一条线，算法却报「双边」，
            第二条是从远处散落的黄点里凭空造出来的，两条线的方向都不对。
            主拟合线一直有这两道检查，配对线也应该有 —— 这不是加限制，
            是把「一条线」的定义统一。
        """
        p = self.p
        if len(pts) < p["min_points_curve"]:
            return None
        coef, inl = self.ransac_poly(pts, rng)
        if coef is None or inl is None:
            return None
        inlier_pts = pts[inl]
        if len(inlier_pts) < p["min_points_curve"]:
            return None
        span = float(inlier_pts[:, 0].max() - inlier_pts[:, 0].min())
        if span < p["min_span_m"]:
            return None                          # 只在很短一段里有点 → 不是线
        return coef

    def _partner_cands(self, rest, dx, want, band, rng):
        out = []
        for side in (-1.0, 1.0):
            sel = np.abs(dx - side * want) < band
            c = self._fit_line(rest[sel], rng)
            if c is not None:
                out.append(c)
        return out

    # ── ⑤ 配对：挑「间距稳定、且最接近当前宽度估计」的一对 ──
    def pair_curves(self, curves, d_ref):
        """从若干条曲线里挑出一对车道边，并定出左右。

        三条判据，都来自几何事实，不是拍脑袋的阈值：
          1) 间距落在 [partner_min_m, partner_max_m] 内（物理上可能的车道宽）
          2) **间距在整段深度上保持稳定**——同一条车道的两条边是平行的，
             间距处处相等；两条相交的线间距会趋近 0。
             2026-10-24 静态扫描发现的：弯道入口处配对的两条线在画面上方交汇成 V 形，
             theta 于是报出 +35.8°，而车其实是摆正的。
          3) 在满足前两条的组合里，取最接近自适应宽度估计 w_ref 的那对
        """
        p = self.p
        lo, hi = p["partner_min_m"], p["partner_max_m"]
        ratio_max = p["pair_gap_ratio_max"]
        # 在 d_ref 上下各取一段深度做间距稳定性检查
        d_lo = max(self.d_min, d_ref - 0.35)
        d_hi = min(self.d_max, d_ref + 0.35)
        ds = np.linspace(d_lo, d_hi, 5)

        best = None
        for i in range(len(curves)):
            for j in range(i + 1, len(curves)):
                gap = abs(self.poly_at(curves[j], d_ref) - self.poly_at(curves[i], d_ref))
                if not (lo <= gap <= hi):
                    continue
                # 间距稳定性：任何一处都不能比 d_ref 处小太多或大太多
                gaps = np.abs(np.array([self.poly_at(curves[j], d) - self.poly_at(curves[i], d)
                                        for d in ds]))
                if gaps.min() < gap / ratio_max or gaps.max() > gap * ratio_max:
                    continue                        # 两条线在收敛/发散 → 不是同一条车道
                score = abs(gap - self.w_ref)
                if best is None or score < best[0]:
                    best = (score, i, j, gap)
        if best is None:
            return None
        _, i, j, gap = best
        # 宽度估计平滑跟上实测值（限幅在物理范围内）
        self.w_ref = float(np.clip(
            self.w_ref + p["lane_width_alpha"] * (gap - self.w_ref), lo, hi))
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
            "lane_width_m": self.w_ref, "quality": 0.0,
            "n_pts": len(pts), "tracking": False, "n_lines": 0, "line_x_m": None,
        }
        if len(pts) < p["min_points_curve"]:
            out["reject_reason"] = (f"黄像素仅 {len(pts)} 个，"
                                    f"少于要求的 {p['min_points_curve']} 个")
            return out

        # ── 时序先验：上一帧的车道在哪，这一帧就只在附近找 ──
        #   这条赛道有路口、地面箭头、多条并行的黄线，全局搜索经常配错对
        #   （效果图里出现过 w=0.38 m 的斜穿）。有了上一帧的约束，
        #   路口也能锁住「我正在走的这条车道」。
        #   跟踪失败（点太少）就退回全局搜索，不会卡死。
        pts_search = pts
        prev = self.prev_pair
        if prev is not None:
            d_all = pts[:, 0]
            near = np.zeros(len(pts), dtype=bool)
            for c in prev:
                dist = np.abs(np.polyval(c, d_all - self.d0) - pts[:, 1])
                near |= dist < p["track_gate_m"]
            if near.sum() >= p["min_points_curve"]:
                pts_search = pts[near]
                out["tracking"] = True

        # 顺序 RANSAC：找第一条，剔除它，再找第二条
        c1, in1 = self.ransac_poly(pts_search, rng)
        if c1 is None:
            out["reject_reason"] = (f"有 {len(pts)} 个黄像素，"
                                    f"但凑不出连续的线（可能都是噪点/其他黄色物体）")
            return out

        # ★ 剔除第一条线时要用「线半宽」，不能用内点阈值
        #   黄线本身约 10 cm 宽，而内点阈值只有 3 cm。
        #   如果只按 3 cm 剔除，同一条线的剩余像素会被第二次 RANSAC
        #   当成「另一条线」拟合出来 —— 表现为两条线间距 0.28 m 而不是 0.54 m，
        #   于是配对被拒，退化成单边模式。2026-10-23 从数据里查出来的。
        dx = pts_search[:, 1] - np.polyval(c1, pts_search[:, 0] - self.d0)
        rest = pts_search[np.abs(dx) > p["line_half_width_m"]]
        dx = dx[np.abs(dx) > p["line_half_width_m"]]

        # ── 找配对线：只在「距第一条约一个车道宽」的带子里找 ──
        #   原来是「在剩下的所有点里随便找第二条，再检查间距对不对」。
        #   路口和地面标线多的时候，随便找到的那条往往不是同一车道的另一条边，
        #   配对检查才发现已经晚了（实测出现过 w=0.38 m 的斜穿假配对）。
        #   把约束放到搜索里：两个方向各试一次，各拟合一条候选线。
        want = self.w_ref
        band = max(p["partner_band_m"], 0.20 * want)   # 宽车道按比例放宽
        cands = self._partner_cands(rest, dx, want, band, rng)
        if not cands:
            # 兜底：宽度估计可能滞后（刚进更宽的区域），
            # 就在物理范围内找一条最强的，交给配对判据决定。
            # 同样必须过 _fit_line 的质量检查，否则又会凭空造线。
            lo, hi = p["partner_min_m"], p["partner_max_m"]
            sel = (np.abs(dx) >= lo) & (np.abs(dx) <= hi)
            c = self._fit_line(rest[sel], rng)
            if c is not None:
                cands = [c]

        d_ref = p["eval_d_m"]
        d_head = p["heading_d_m"]

        # ── 数据支撑范围：拟合只在有点的地方可信，不能凭空调外推 ──
        inl = pts_search[in1]
        d_lo, d_hi = float(inl[:, 0].min()), float(inl[:, 0].max())
        span = d_hi - d_lo
        out["d_span"] = span
        if span < p["min_span_m"]:
            out.update(reject_reason=f"有效点只跨 {span:.2f}m，太短，拟合不可信")
            return out
        # 评估点只夹上限，不夹下限。
        # ★ 为什么允许外推到车的位置：
        #   e 和 theta 应该在【车所在位置】量，量在车前就有超前。
        #   R=1.0m 的弯，0.35m 的超前就是 20° —— 表现是「幅度对、时机早」，
        #   车提前入弯、切内线、再从外侧修回来（2026-10-24 现场观察）。
        #   拟合段是 0.35~1.45m，二次曲线外推 0.35m 是可接受的。
        # 配对仍然用夹过的 d_ref（间距必须在有数据的地方算）。
        d_pair = min(max(d_ref, d_lo), d_hi)
        d_ref = min(d_ref, d_hi)
        d_head = min(d_head, d_hi)

        curves = [c1] + cands
        pair = self.pair_curves(curves, d_pair) if len(curves) >= 2 else None

        if pair is not None:
            iL, iR = pair
            coefL, coefR = curves[iL], curves[iR]
            mode = 0
            lane_w = abs(self.poly_at(coefR, d_pair) - self.poly_at(coefL, d_pair))
        else:
            # 只有一条线：靠"车道宽"推另一条。哪边？看这条线在车轴的哪一侧。
            c = c1
            xc = self.poly_at(c, d_ref)
            if xc < 0:          # 这条线在车的左边 → 它是左线
                coefL = c
                coefR = c + np.array([0, 0, self.w_ref])
                mode = 1
            else:               # 在右边 → 右线
                coefR = c
                coefL = c - np.array([0, 0, self.w_ref])
                mode = 2
            lane_w = self.w_ref
            # 诊断用：单边模式下，这条线在车心横向多远。
            # 车在车道里时，自己车道的边界离车心不可能超过「半车道宽+余量」。
            # 超过就说明跟的是别的赛道的线，theta 也不可信。
            out["line_x_m"] = float(xc)
        out["n_lines"] = 1 + len(cands)

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

        # ── 可信度判定：全部收在这里，对外只输出一个 is_lost ──
        # 这是架构上的「唯一判定点」。brain_node 不再重复判质量、判跳变，
        # 只按 is_lost 做状态机滞回。改判定规则只改这一个地方。

        # ★ 时序连续性：车道不可能在相邻两帧之间跳到别处。
        #   实测日志（2026-10-24 推车过弯）：theta 从 +6° 一帧一帧爬到 +46°，
        #   delta 跟着打满 25°，车就提前切内线、然后冲出外侧。
        #   30fps、0.21 m/s 时车每帧只走 7mm，车道中线的横移不可能超过 20cm。
        #   超过就是拟合跳到另一条车道了，宁可判丢线（保持方向）也不要跟着错。
        jump = None
        if self.prev_mid is not None:
            jump = abs(self.poly_at(mid_coef, d_ref) - self.poly_at(self.prev_mid, d_ref))

        # ★ 航向角也不能一帧翻脸。
        #   实测（同一份推车日志）：弯道全程只有一条线可配对（mode=2 仅右），
        #   theta 在 8.28s→8.32s 之间从 +7.6° 直接变成 -24.3°，一帧反向 32°。
        #   车一帧只走 7mm，航向角物理上不可能这样变 —— 是那条线换成别的了。
        th_jump = None
        if self.prev_theta is not None:
            th_jump = abs(math.degrees(out["theta_rad"] - self.prev_theta))

        limit = min(lane_w / 2 + p["e_margin_m"], p["e_abs_limit_m"])
        if jump is not None and jump > p["jump_max_m"]:
            out.update(is_lost=True,
                       reject_reason=f"车道中线一帧跳了 {jump*100:.0f}cm（> {p['jump_max_m']*100:.0f}cm）")
        elif th_jump is not None and th_jump > p["theta_jump_deg"]:
            out.update(is_lost=True,
                       reject_reason=f"theta 一帧跳了 {th_jump:.0f}°（> {p['theta_jump_deg']:.0f}°）")
        elif abs(out["e_m"]) > limit:
            out.update(is_lost=True, quality=0.0,
                       reject_reason=f"e={out['e_m']*100:+.0f}cm > 允许 {limit*100:.0f}cm")
        elif abs(math.degrees(out["theta_rad"])) > p["theta_limit_deg"]:
            out.update(is_lost=True, quality=0.0,
                       reject_reason=f"theta={math.degrees(out['theta_rad']):+.0f}° 超限")
        elif abs(out["curvature"]) > p["max_curvature_1pm"]:
            # 弯得比车能转的还急，肯定是拟合跑飞了（几何上不可能）
            out.update(is_lost=True, quality=0.0,
                       reject_reason=f"kappa={out['curvature']:+.1f} 超过车的最小转弯能力")
        elif out["quality"] < p["quality_min"]:
            # 质量门限：点数 × 数据跨度都不够，拟合出来的方向是外推的幻想值。
            # 实测（2026-10-24 日志）：弯道入口 q 掉到 0.38，theta 同时从 +6° 飙到
            # +46°，纠偏项跟着打满舵 —— 就是「还没进弯方向盘就死了」。
            # ★ 保留 quality 原值，别清零，调参时要能看到真实数值。
            out.update(is_lost=True,
                       reject_reason=f"quality={out['quality']:.2f} < {p['quality_min']:.2f}")

        # ── 更新时序先验 ──
        #   只在「可信」时更新：双边，或至少点数够多。单边时另一条是推算出来的，
        #   把它当先验会让下一帧继续沿着推算值漂走，所以单边不更新。
        self.prev_pair = (coefL, coefR) if (mode == 0 and not out["is_lost"]) else None
        # 中线/航向先验：跳变被拒之后不要用它更新，否则会一路锁在错误的车道上
        if not out["is_lost"]:
            self.prev_mid = mid_coef
            self.prev_theta = out["theta_rad"]
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
