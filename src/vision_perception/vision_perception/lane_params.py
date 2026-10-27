# -*- coding: utf-8 -*-
"""
lane_params.py —— 巡线算法 v2 的全部可调参数（只改这个文件）

调参顺序：
    1. 鸟瞰图对不对        → h_m / pitch_deg / hfov_deg / yaw_deg
    2. 黄线提得出来吗      → color_*
    3. 扫描稳不稳          → edge_jump_m / block_stop_m / max_slope
    4. 数值合不合理        → e_offset_m
"""

DEFAULTS = {
    # ══════════ 一、相机几何（决定 IPM 准不准）══════════
    "h_m": 0.16,          # 镜头离地高度（米），实测
    "pitch_deg": 12.45,   # 俯角（度）。用车前 1.00 m 标记反解
    "hfov_deg": 110.6,    # 水平视场角。规格书 120° 是对角角 → 水平 110.6°
    "yaw_deg": 9.0,       # 相机相对车头的偏航，正 = 向右偏。2026-10-23 实测 ~9°
    "img_w": 1280,
    "img_h": 720,
    "roi_top": 0,         # v2 不再裁剪原图（地平线由几何算出）
    "roi_bottom": 720,    # 太靠底边会拍到车体时再改小
    "horizon_margin_px": 12,   # 地平线往下让出多少行才算有效

    # ══════════ 二、鸟瞰图范围 ══════════
    #   v2 加宽加深：弯道半径 ~0.6 m，窄的鸟瞰图一进弯线就出画了。
    "bev_width_m": 2.0,   # 横向覆盖（左右各 1.0 m）
    "bev_depth_m": 1.5,   # 纵向覆盖
    "bev_m_per_px": 0.005,
    "bev_d_min": 0.22,    # 最近处距相机（米）

    # ══════════ 三、颜色（HSV：H 0~180，S/V 0~255）══════════
    "color_h_min": 18,
    "color_h_max": 45,    # 过曝时黄的 H 会偏大
    "color_s_min": 45,    # 过曝时 S 掉到 ~100
    "color_v_min": 80,
    "morph_open_px": 3,
    "morph_close_px": 5,
    # 蓝色底（赛道外）。只用来判断「种子是不是落在路面上」
    "blue_h_min": 90,
    "blue_h_max": 130,
    "blue_s_min": 70,
    "blue_v_min": 60,

    # ══════════ 四、逐行扫描 ══════════
    "line_width_m": 0.10,      # 黄线本身的宽度（实物 100 mm）
    "lane_width_m": 0.54,      # 车道宽初值（线中心到线中心），之后自适应
    "lane_width_alpha": 0.0,   # 固定赛道宽度：54 cm；宽区不改写主车道先验
    "partner_min_m": 0.35,     # 车道宽物理下限（更窄 = 横向黄线的缺口，跳过）
    "partner_max_m": 1.30,     # 车道宽物理上限（更宽 = 某一侧不是自己的边，丢掉）
    "wide_lane_enable": True,   # 挡板区两边都看见且连续时，临时按实测宽度走
    "wide_lane_max_m": 1.80,    # 挡板/宽区允许的最大宽度；不写回 54 cm 主先验
    "wide_lane_min_steps": 6,   # 连续足够多步才承认宽区，防止十字口误配对
    "seed_search_m": 0.40,     # 种子离上一帧中心最多偏多远（再远算丢）
    "seed_min_road_m": 0.06,   # 种子所在的路面段至少多宽
    "march_step_m": 0.02,      # 沿车道每步走多远
    "march_len_m": 1.6,        # 最多走多远（弧长）
    "edge_jump_m": 0.05,       # 相邻两步同一侧边界距离最多变多少（路口虚线缺口靠它挡）
    "start_search_m": 0.40,    # 起步时两侧都看不到边，最多往前找多远
    "max_off_step_m": 0.015,    # 锁定后每步中心横向修正上限
    "block_stop_m": 0.32,      # 挡板遮挡/横向虚线时允许沿中心线跨过的距离
    "occlusion_hold_frames": 30,  # 黄线被挡板遮住时沿用上一帧中心线约 1.0 s
    "occlusion_hold_quality": 0.25,

    # ══════════ 五、拟合与输出 ══════════
    "min_steps": 6,            # 至少走出多少步才输出
    "head_len_m": 0.25,        # theta 用起点后多长一段中心线算
    "smooth_n": 5,             # 中心线滑动平均窗口（步）
    "curv_len_m": 0.40,        # kappa 用起点后多长一段中心线算
    "good_len_m": 0.50,        # 走出这么长就算满分质量
    "eval_d_m": 0.22,          # e 在相机前多远处量
    "look_d_m": 0.50,          # 预瞄点距离（给纯追踪）

    "e_offset_m": 0.0,         # 安装偏差（车摆正读到的 e）
    "theta_offset_deg": 0.0,
    "e_margin_m": 0.15,        # |e| 允许超出半车道宽多少
    "e_abs_limit_m": 0.60,
    "quality_min": 0.30,       # 低于它 → is_lost
    "theta_limit_deg": 110.0,  # 急弯里车被甩到外侧时，车道方向真的能到 80~90°（实拍 lap2_0130 等）
    # 最小转弯半径：前轮最大实际 15.2°（舵机 25° 指令），R=0.143/tan15.2°=0.53 m
    "max_curvature_1pm": 2.5,

    "debug_scale": 1.0,
}
