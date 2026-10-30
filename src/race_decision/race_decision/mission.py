# -*- coding: utf-8 -*-
"""
mission.py —— race_ws_andy 新增：里程计 + 路段状态机 + 避障 v2 + 路口直行 + 终点停车

挂在 brain_node 上（BrainNode 继承 MissionMixin），原来的 Stanley 巡线一行没动，
只在 on_timer 里多一个钩子：mission_step() 返回 True 时本周期由它出指令，否则走原巡线。

模式：
    LANE     原巡线（队友调好的 Stanley）
    AVOID    避障 v2：挡板位置记在里程计坐标里（出视野也不忘），
             选左/右缝 → 纯追踪到目标横向位置 → 保持到车身越过挡板 → 回中 → 交还巡线
    XING     路口直行：按「斑马线后里程窗口」进入，锁 IMU 航向直行，不看线
    LOSTYAW  直道上丢线：锁 IMU 航向直行（弯道丢线仍走原来的「保持转向」）
    FINISH   第 N 圈过起点线后停车

坐标：里程计 x 前 / y 左 / yaw 逆时针为正（左转为正），单位米/弧度。
"""
import math
import os
import time

from geometry_msgs.msg import TwistStamped
from std_msgs.msg import String, Float32MultiArray


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


class MissionMixin:
    # ═════════════════════════════════════════════════════════
    def mission_init(self):
        P = self.declare_parameter
        # ── 里程计 ──
        P("gyro_sign", -1.0)           # 2026-10-29 实车自检：左转时底盘 gz 为负 → -1。0 = 自动判定
        P("odom_v_scale", 1.0)         # 底盘上报速度的修正倍率
        P("odom_cmd_fallback", True)   # 底盘速度一直为 0 但在发油门时，用指令速度代替
        # ── 避障 v2 ──
        P("av2_enable", True)          # v2（世界坐标规划+纯追踪）已停用，见 av3
        # ── 避障 v3：车道坐标系偏置巡线 ──
        #   挡板和车道中心线用同一帧、同一距离比较 → 挡板相对车道的横向位置不受测距误差/航向参考影响；
        #   不自己开车，只把「车道中心」平移 y 交给原巡线控制器；看不见线时用里程计+陀螺仪推算 e/θ 顶上。
        P("av3_enable", False)         # v3 已被 v4（贴右线）取代
        P("av4_enable", True)
        P("av4_d_normal", 0.22)        # 正常时车中心离右黄线（车道半宽）
        P("av4_car_half", 0.09)
        P("av4_margin", 0.04)          # 离挡板的余量
        P("av4_line_margin", 0.04)     # 车边离右线的余量
        P("av4_d_max", 0.65)           # 从左侧绕时最多离右线这么远
        P("av4_far_m", 0.70)           # 挡板右沿离右线超过它 → 不是本车道的（0.95 时把弯外别的路段的板也收进来了）
        P("av4_yaw_gate_deg", 15.0)    # 避障中右线方向跳变超过它 → 拟合被挡板带偏，改推算
        P("av4_jump_m", 0.15)          # 避障中离线距离跳变超过它 → 同上
        P("av4_exit_good", 10)         # 交还巡线前要连续这么多帧右线可信且车头摆正
        P("av4_exit_deg", 10.0)
        P("av4_exit_max_m", 1.0)
        P("av4_speed", 0.40)           # 避障区（RLINE）限速：0.4 已验证完赛，全局 max_speed 提高时这段不跟着提
        P("av4_post_m", 0.8)           # 交还巡线后再按 av4_speed 走这么远（出口接大左弯）
        P("av4_ret_lane", True)        # 过完最后一块板：原巡线（双线）连续看清就直接交还，不再靠右线推算回中（录包：出口双线 q≈0.8–0.97、w≈0.47）
        P("av4_ret_after_m", 0.25)     # 车越过最后一块板这么远后才允许
        P("av4_ret_bars", 2)           # 本区已见到这么多块板才允许（防止两板之间就交还）；不够时要越过最后一块 av4_ret_far_m
        P("av4_ret_far_m", 0.80)
        P("av4_ret_n", 4)              # 连续这么多帧双线可信
        P("av4_ret_q", 0.75)
        P("av4_ret_e", 0.30)
        P("av4_ret_th_deg", 45.0)
        P("av4_max_ang_deg", 22.0)     # 横移时车头和线的最大夹角（Stanley 对 0.3 m 误差会给到 50° → 出线）           # 挡板右沿离右线超过它 → 不是本车道的
        P("av4_meas_min_m", 0.30)
        P("av4_meas_max_m", 1.15)      # 右线只拟合到 0.75 m，外推太远（弯道出口）会把挡板和线的关系算错
        P("av4_dr_max_m", 1.50)        # 右线被挡住时最多推算这么远
        P("av4_dr_tail_m", 1.50)       # 过完最后一块板后，推算可再多走这么远（回中阶段右线常看不到）
        P("av4_reacq_n", 5)            # 方向一致但离线跳变的右线连续这么多帧 → 认为推算漂了，重新对准
        P("lostyaw_av_max_m", 2.0)     # 避障交还后的航向锁最多走这么远
        P("lostyaw_av_good", 8)        # 巡线连续这么多帧可信（q≥0.5、|e|<0.15、|θ|<12°）才退出航向锁
        P("av4_lock_m", 0.55)          # 挡板近于此距离后不再轻易换边
        P("av4_pass_m", 0.12)          # 相机越过挡板这么远就可以开始横移（板很薄，车尾跟着前轮走）
        P("av3_trigger_m", 1.40)       # 挡板（沿路程）进入这个距离开始偏
        P("av3_meas_min_m", 0.45)
        P("av3_meas_max_m", 2.20)
        P("av3_rel_gate_m", 0.40)      # 挡板离车道中心超过它 → 场边杂物
        P("av3_assoc_m", 0.35)         # 沿路程多近算同一块
        P("av3_min_hits", 2)           # 至少看到几次才信
        P("av3_clear_m", 0.33)         # 车中心离挡板中心至少这么远（板半宽 0.20 + 车半宽 0.09 + 余量 0.04）
        P("av3_y_max", 0.34)           # 偏置上限
        P("av3_pass_m", 0.30)          # 越过挡板这么远才算过
        P("av3_lock_m", 0.70)          # 进入这个距离后不再换边（除非原来那边过不去）
        P("av3_slew", 1.2)             # 偏置变化率：每走 1 m 最多变这么多 m
        P("av3_dr_max_m", 1.5)         # 最后一块过后视觉一直不靠谱时，最多再推算这么远
        P("av3_min_s", 0.30)           # 发车后走过这么远才开始找挡板
        P("av3_yl_gain", 0.03)         # 偏置中车道航向的视觉校正增益（每帧）
        P("av3_e_gate", 0.08)          # 偏置中视觉 e 和推算差超过它就当跳变丢掉
        P("av3_e_gain", 0.06)          # 偏置中视觉 e 的校正增益（每帧）
        P("av3_hold_m", 0.70)          # 最后一块挡板过后再用推算走这么远，出宽车道再交还视觉
        # 挡板位置用框底边（落地点）按和巡线同一套相机几何反投到地面，和车道中心线同尺度
        #   （检测节点给的 dist_m 是按框高估的，实测误差 −13%~−42%，横向会跟着错一个比例）
        P("av3_use_ipm", True)
        P("av3_cam_h", 0.206)          # = lane_params.h_m
        P("av3_cam_pitch_deg", 15.13)  # = lane_params.pitch_deg
        P("av3_cam_yaw_deg", 6.56)     # = lane_params.yaw_deg（正 = 相机朝右）
        P("av3_cam_f_px", 542.0)
        P("av3_img_w", 1280.0)
        P("av3_img_h", 720.0)         # 偏置中丢线时，最多推算这么远
        P("av2_require_cw", True)
        P("av2_straight_kappa", 0.5)   # 弯道里不准进避障：|κ| 小于它
        P("av2_straight_gz", 0.35)     # 且 |角速度| rad/s 小于它
        P("av2_straight_frames", 8)
        P("av2_straight_q", 0.45)
        P("av2_ref_alpha", 0.04)       # 避障中用视觉 (航向+θ) 慢慢修正参考方向，每帧比例
        P("av2_ref_max_dev_deg", 35.0) # 视觉给的方向和参考差太多就不信      # 挡板区视觉 quality 常年 0.6，别卡太高   # 连续这么多帧都是直道才允许触发      # 过了斑马线才允许避障（赛道顺序：斑马线 → 挡板区）
        P("av2_lat_gate_m", 0.35)      # 挡板离车道中心横向超过这个就不是赛道上的挡板（场边杂物）
        P("av2_abort_psi_deg", 55.0)   # 避障中车头偏离参考方向超过这个 → 放弃避障交还巡线
        P("av2_trigger_m", 1.30)       # 挡板（沿车道方向）进入这个距离开始规划
        P("av2_meas_max_m", 2.40)      # 超过这个距离的测量不入库（太远不准）
        P("av2_meas_min_m", 0.40)      # 比这更近的测量不更新位置（框被裁、宽度不准）
        P("av2_merge_m", 0.30)         # 新测量离已记挡板多近算同一块
        P("av2_barrier_half_w", 0.20)  # 挡板半宽（规格 40 cm）
        P("av2_car_half_w", 0.09)      # 车半宽
        P("av2_margin_m", 0.07)        # 额外安全余量
        P("av2_y_max", 0.32)           # 横向目标上限（宽区 0.90/2 − 车半宽 − 余量）
        P("av2_max_psi_deg", 30.0)     # 避障中车头相对车道最多偏这么多（防过冲、防回中时打死）
        P("av2_meas_max_psi_deg", 38.0)# 车头偏车道超过这个时，挡板测量不入库（斜看测距误差变成横向误差）
        P("av2_flip_hyst", 0.10)       # 已选好一边后，另一边要好这么多代价才换
        P("av2_lookahead_m", 0.45)     # 纯追踪预瞄距离
        P("av2_speed", 0.35)
        P("av2_pass_m", 0.28)          # 相机越过挡板这么远才算整车过了（车身长）
        P("av2_side_lock_m", 0.70)     # 挡板进入这个距离后锁定走哪边，不再来回改
        P("av2_return_tol_m", 0.08)    # 回中到这个横向误差以内交还巡线
        P("av2_max_len_m", 3.5)        # 避障最多走这么远，超了强制交还巡线
        P("av2_cam_x_m", 0.0)          # 相机在里程计参考点前方多远
        P("av2_cam_f_px", 542.0)       # 1280 宽、99.5° 水平视场
        P("av2_cam_yaw_deg", 6.56)     # 相机右偏角（和 lane_params.yaw_deg 一致）
        P("av2_img_w", 1280.0)
        P("av2_lane_th_max_deg", 35.0)  # 2026-10-29 实车：弯出口 θ 常有 25°，15° 夹死导致参考偏 20°  # 用视觉 θ 修正参考航向时的上限
        # ── 路口 ──
        #   斑马线停车点之后的里程窗口：[起点1, 长度1, 起点2, 长度2, ...]，单位米。
        #   [0.0] = 关闭（第一次跑完用 CSV 日志标出来再填）。
        P("xing_windows", [0.0])
        P("xing_speed", 0.35)
        P("xing_k_yaw", 1.2)           # 航向保持增益：前轮角 = −k × 航向误差
        # ── 直道丢线 → 航向保持 ──
        P("lost_yaw_hold", True)
        P("lost_straight_kappa", 0.6)  # |κ| 小于它、且最近转向小于 lost_straight_steer_deg 才算直道
        P("lost_straight_steer_deg", 6.0)
        P("lostyaw_max_m", 0.8)        # 航向保持最多走这么远，之后退回原丢线逻辑
        # ── 圈数 / 终点 ──
        P("laps", 2)
        P("finish_enable", False)
        P("finish_extra_m", 0.35)      # 过了起点线再多走多远停（整车过线）
        P("cw_voice_text", "人行道前停车，礼让行人")
        # ── 日志 ──
        P("log_csv", True)
        P("log_dir", os.path.expanduser("~/race_logs"))

        # 里程计状态
        self.od_x = self.od_y = self.od_yaw = 0.0
        self.od_s = 0.0
        self.od_v = 0.0
        self.od_gz = 0.0
        self.od_t = None
        self.od_sign = None
        self._sign_votes = 0
        self._sign_n = 0
        self._v_zero_cnt = 0
        self._v_fallback = False
        self._cmd_v = 0.0
        self.create_subscription(TwistStamped, "/odom_raw", self.on_odom, 20)
        self.rl_a, self.rl_b, self.rl_n, self.rl_t = 0.0, 0.0, 0, 0.0
        self.create_subscription(Float32MultiArray, "/lane_right", self._on_rline, 10)

        # 状态机
        self.mode = "LANE"
        self.n_cw = 0
        self.o_bars = []               # av3 挡板：{"s": 路程位置, "rel": 相对车道中心横向(左正), "n", "side"}
        self.o_y = 0.0                 # 当前偏置
        self.o_on = False
        self.o_det_seen = -1.0
        self.o_e = 0.0; self.o_yaw_lane = None; self.o_dr_s0 = None
        self.s_cw = []                 # 每次斑马线触发时的里程
        self.s_cw_last = None
        self.barriers = []             # [{X, Y, n, t, side}]，世界坐标
        self._det_seen = 0.0
        self.av_ref = None             # (x0, y0, yaw0)
        self.av_s0 = 0.0
        self.av_y_t = 0.0
        self.xing_ref = None
        self.xing_end = 0.0
        self.ly_ref = None
        self.ly_s0 = 0.0
        self._lost_streak = 0
        self._ok_streak = 0
        self._lane_yaw = None          # 最近一次可信的车道航向（世界）
        self.pub_mode = self.create_publisher(String, "/mission_state", 10)

        # CSV
        self._csv = None
        self._csv_t = 0.0
        if bool(self.get_parameter("log_csv").value):
            try:
                d = self.get_parameter("log_dir").value
                os.makedirs(d, exist_ok=True)
                fn = os.path.join(d, time.strftime("run_%m%d_%H%M%S.csv"))
                self._csv = open(fn, "w", buffering=1)
                self._csv.write("t,s,s_cw,x,y,yaw_deg,v,gz,mode,state,e,th_deg,kappa,"
                                "lane_w,q,lost,n_bar,steer_deg,cw,reason\n")
                self._csv_fs = 0.0
                try:
                    os.fsync(self._csv.fileno())
                    dfd = os.open(d, os.O_RDONLY); os.fsync(dfd); os.close(dfd)
                except Exception:
                    pass
                self.get_logger().info(f"[andy] CSV 日志: {fn}")
            except Exception as ex:
                self.get_logger().warn(f"[andy] CSV 打不开: {ex}")
        self.get_logger().info("[andy] mission 已加载：避障v2 / 路口直行 / 直道航向保持 / 终点停车")

    # ═════════════════════════════════════════════════════════
    # 里程计：底盘上报 vx + 陀螺仪 gz
    # ═════════════════════════════════════════════════════════
    def on_odom(self, msg):
        g = self.get_parameter
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        vx = float(msg.twist.linear.x) * float(g("odom_v_scale").value)
        gz = float(msg.twist.angular.z)
        # 底盘速度失效兜底
        if bool(g("odom_cmd_fallback").value):
            if abs(vx) < 0.01 and self._cmd_v > 0.12:
                self._v_zero_cnt += 1
            elif abs(vx) > 0.03:
                self._v_zero_cnt = 0
                self._v_fallback = False
            if self._v_zero_cnt > 25 and not self._v_fallback:
                self._v_fallback = True
                self.get_logger().warn("[andy] 底盘速度一直是 0，改用指令速度积分里程")
            if self._v_fallback:
                vx = self._cmd_v
        # 陀螺仪方向自动判定：左打（steer_prev>0）时 gz 应为正
        gs = float(g("gyro_sign").value)
        if gs != 0.0:
            sign = 1.0 if gs > 0 else -1.0
        else:
            if self.od_sign is None and abs(vx) > 0.10 and abs(self.steer_prev) > math.radians(10) \
                    and abs(gz) > 0.15:
                self._sign_votes += 1 if (gz > 0) == (self.steer_prev > 0) else -1
                self._sign_n += 1
                if self._sign_n >= 25:
                    self.od_sign = 1.0 if self._sign_votes > 0 else -1.0
                    self.get_logger().warn(
                        f"[andy] 陀螺仪方向自检完成：gyro_sign={self.od_sign:+.0f}"
                        f"（投票 {self._sign_votes}/{self._sign_n}），建议写死到参数里")
            sign = self.od_sign if self.od_sign is not None else 1.0
        if self.od_t is None:
            self.od_t = t
            return
        dt = t - self.od_t
        self.od_t = t
        if dt <= 0.0 or dt > 0.2:
            return
        self.od_v, self.od_gz = vx, gz * sign
        self.od_yaw = wrap(self.od_yaw + self.od_gz * dt)
        self.od_x += vx * math.cos(self.od_yaw) * dt
        self.od_y += vx * math.sin(self.od_yaw) * dt
        self.od_s += abs(vx) * dt

    # ═════════════════════════════════════════════════════════
    # 斑马线触发时由 brain 调用
    # ═════════════════════════════════════════════════════════
    def on_crosswalk_event(self, now):
        self.n_cw += 1
        self.s_cw.append(self.od_s)
        self.s_cw_last = self.od_s
        self.barriers = []
        self.get_logger().warn(f"[andy] 第 {self.n_cw} 次斑马线，里程 {self.od_s:.2f} m")

    # ═════════════════════════════════════════════════════════
    def _lane_yaw_now(self):
        """当前车道方向（世界）= 车航向 + 视觉 θ（夹一下）。"""
        th = self.th_s if self.th_s is not None else 0.0
        lim = math.radians(float(self.get_parameter("av2_lane_th_max_deg").value))
        return wrap(self.od_yaw + clamp(th, -lim, lim))

    def _emit(self, v_target, dw):
        """dw = 前轮角（左正，弧度）。走和原巡线一样的限幅 / 变化率 / 斜坡。"""
        g = self.get_parameter
        ratio = max(float(g("steer_ratio").value), 1e-3)
        delta = (dw + math.radians(float(g("steer_trim_deg").value))) / ratio
        dmax = math.radians(float(g("steer_max_deg").value))
        delta = clamp(delta, -dmax, dmax)
        rate = math.radians(float(g("steer_rate_deg_per_s").value)) * 0.02
        delta = clamp(delta, self.steer_prev - rate, self.steer_prev + rate)
        self.steer_prev = delta
        self.delta_f = delta
        self.steer_hist.append(float(delta))
        v_max = float(g("max_speed").value) * float(g("speed_scale").value)
        v_target = min(v_target, v_max)
        if v_target > self.v_cmd:
            v = min(v_target, self.v_cmd + float(g("accel").value) * 0.02)
        else:
            v = max(v_target, self.v_cmd - float(g("decel").value) * 0.02)
        self.v_cmd = v
        self.publish(v, delta * float(g("steer_sign").value))

    # ═════════════════════════════════════════════════════════
    # 挡板入库
    # ═════════════════════════════════════════════════════════
    def _ingest_barriers(self):
        g = self.get_parameter
        if self.det_time <= self._det_seen:
            return
        self._det_seen = self.det_time
        f = float(g("av2_cam_f_px").value)
        W = float(g("av2_img_w").value)
        yc = math.radians(float(g("av2_cam_yaw_deg").value))
        dmin, dmax = float(g("av2_meas_min_m").value), float(g("av2_meas_max_m").value)
        merge = float(g("av2_merge_m").value)
        cy, sy = math.cos(self.od_yaw), math.sin(self.od_yaw)
        if self.mode == "AVOID" and self.av_ref is not None:
            dev = abs(wrap(self.od_yaw - self.av_ref[2]))
        else:
            dev = abs(self.th_s) if self.th_s is not None else 0.0
        if dev > math.radians(float(g("av2_meas_max_psi_deg").value)):
            return
        for it in self.det_items:
            if it.class_id != 2 or it.dist_m <= 0.0:
                continue
            d = float(it.dist_m)
            clipped = (it.cx - it.w / 2.0) < 0.01 or (it.cx + it.w / 2.0) > 0.99
            good = (dmin <= d <= dmax) and not clipped
            latc = d * (W / 2.0 - it.cx * W) / f            # 相机系，左正
            fx = d * math.cos(yc) + latc * math.sin(yc) + float(g("av2_cam_x_m").value)
            fy = -d * math.sin(yc) + latc * math.cos(yc)
            X = self.od_x + fx * cy - fy * sy
            Y = self.od_y + fx * sy + fy * cy
            best, bd = None, merge
            for b in self.barriers:
                dd = math.hypot(b["X"] - X, b["Y"] - Y)
                if dd < bd:
                    best, bd = b, dd
            if best is not None:
                if good:
                    a = 0.3
                    best["X"] += a * (X - best["X"])
                    best["Y"] += a * (Y - best["Y"])
                    best["n"] += 1
                best["t"] = self.det_time
            elif good:
                self.barriers.append({"X": X, "Y": Y, "n": 1, "t": self.det_time, "side": 0})
        # 太旧且已远在身后的删掉
        self.barriers = [b for b in self.barriers
                         if self._fwd_of(b) > -1.0 or self.det_time - b["t"] < 2.0]

    def _fwd_of(self, b, yaw_ref=None):
        """挡板在（参考方向）前方多远。"""
        yr = self.od_yaw if yaw_ref is None else yaw_ref
        return (b["X"] - self.od_x) * math.cos(yr) + (b["Y"] - self.od_y) * math.sin(yr)

    def _lat_of(self, b, yaw_ref=None):
        """挡板在（参考方向）左侧多远。"""
        yr = self.od_yaw if yaw_ref is None else yaw_ref
        return -(b["X"] - self.od_x) * math.sin(yr) + (b["Y"] - self.od_y) * math.cos(yr)

    def _to_ref(self, X, Y):
        x0, y0, yr = self.av_ref
        dx, dy = X - x0, Y - y0
        return dx * math.cos(yr) + dy * math.sin(yr), -dx * math.sin(yr) + dy * math.cos(yr)

    # ═════════════════════════════════════════════════════════
    # 主钩子：返回 True = 本周期已经出指令
    # ═════════════════════════════════════════════════════════
    def mission_step(self, now, msg, v_max):
        g = self.get_parameter
        self._ingest_barriers()
        # 丢线计数（自己的，和原逻辑互不影响）
        if msg.is_lost:
            self._lost_streak += 1
            self._ok_streak = 0
        else:
            self._ok_streak += 1
            self._lost_streak = 0
            if self._ok_streak >= 3 and self.mode == "LANE":
                self._lane_yaw = self._lane_yaw_now()
        if (not msg.is_lost) and msg.quality > float(g("av2_straight_q").value) \
                and abs(msg.curvature) < float(g("av2_straight_kappa").value) \
                and abs(self.od_gz) < float(g("av2_straight_gz").value):
            self._straight_n = getattr(self, "_straight_n", 0) + 1
            ly = self._lane_yaw_now()
            if self._straight_n == 1:
                self._ly_c, self._ly_s = math.cos(ly), math.sin(ly)
            else:
                self._ly_c = 0.85 * self._ly_c + 0.15 * math.cos(ly)
                self._ly_s = 0.85 * self._ly_s + 0.15 * math.sin(ly)
        else:
            self._straight_n = 0
        self._msg = msg
        # 分段限速（brain 里 v_target 会取 min(v_target, m_vcap)）
        vc = None
        if self.mode in ("RLINE", "OFFSET") or self.od_s < getattr(self, "av_post_until", -1.0):
            vc = float(g("av4_speed").value)
        self.m_vcap = vc
        s_cw = (self.od_s - self.s_cw_last) if self.s_cw_last is not None else -1.0
        handled = False

        # ── 终点 ──
        if self.mode == "FINISH":
            self.v_cmd = 0.0
            self.publish(0.0, 0.0)
            handled = True
        elif bool(g("finish_enable").value) and self.n_cw >= int(g("laps").value) \
                and len(self.s_cw) >= 2 and s_cw >= 0.0:
            lap_len = self.s_cw[1] - self.s_cw[0]
            to_finish = lap_len - self.s_cw[0] + float(g("finish_extra_m").value)
            if s_cw >= to_finish:
                self.mode = "FINISH"
                self.get_logger().warn(f"[andy] 完成 {self.n_cw} 圈，停车（圈长 {lap_len:.2f} m）")
                self.v_cmd = 0.0
                self.publish(0.0, 0.0)
                handled = True

        # ── 避障 v4（贴右线）──
        if not handled and bool(g("av4_enable").value):
            self._step_rline(now, msg)
            if self.o_on:
                self.mode = "RLINE"
                self._log(now, msg, s_cw)
                return False

        # ── 避障 v3（偏置巡线）──
        if not handled and bool(g("av3_enable").value):
            self._step_offset(now, msg)
            if self.o_on:
                self.mode = "OFFSET"
                self._log(now, msg, s_cw)
                return False

        # ── 避障 v2 ──
        if not handled and bool(g("av2_enable").value):
            handled = self._step_avoid(now)

        # ── 路口直行 ──
        if not handled:
            handled = self._step_xing(s_cw)

        # ── 直道丢线 → 航向保持 ──
        if not handled:
            handled = self._step_lostyaw(msg)

        if not handled:
            self.mode = "LANE"
        self._log(now, msg, s_cw)
        return handled

    # ─────────────────────────────────────────────────────────
    #  避障 v4：贴右线行驶（2026-10-29 用真实采集图回放后重写）
    #   回放结论：避障区只有右侧一条黄线；挡板蓝色会被巡线当成车道边，巡线 e 在
    #   「挡板当左边界 / 只看右线」两种解之间跳。所以这段不用巡线 e，改成：
    #     · 单独拟合右黄线 → 车到右线距离 dR、线方向 θ（vision_node 发 /lane_right）
    #     · 挡板框底左右角按同一 IPM 投到地面，算「左沿/右沿离右线多远」（同帧同尺度）
    #     · 每块板：右沿离线够宽 → 从板右侧过（D≈正常车道半宽）；否则从板左侧过（D=左沿+半车宽+余量）
    #     · 控制仍交给原巡线控制器：lateral_error = dR − D，heading_error = θ线
    #   回放实测：第一块 右沿离线 0.43~0.47 → 右侧过、D=0.22；第二块 右沿离线 ≈0.03、左沿 0.36~0.44 → 左侧过、D≈0.58
    # ─────────────────────────────────────────────────────────
    def _on_rline(self, m):
        d = list(m.data)
        if len(d) >= 3:
            self.rl_a, self.rl_b, self.rl_n = float(d[0]), float(d[1]), int(d[2])
            self.rl_t = time.time()

    def _step_rline(self, now, msg):
        g = self.get_parameter
        if self.od_s < float(g("av3_min_s").value):
            return
        D0 = float(g("av4_d_normal").value)
        # 原巡线（双线）是否可信：在下面改 msg 之前看原始值
        tg = (not msg.is_lost) and msg.quality >= float(g("av4_ret_q").value) \
            and 0.36 <= msg.lane_width <= 0.56 and abs(msg.lateral_error) < float(g("av4_ret_e").value) \
            and abs(msg.heading_error) < math.radians(float(g("av4_ret_th_deg").value))
        self.r_tg = (getattr(self, "r_tg", 0) + 1) if tg else 0
        half = float(g("av4_car_half").value)
        mg = float(g("av4_margin").value)
        fresh = (time.time() - getattr(self, "rl_t", 0.0)) < 0.15
        a_ok = fresh and self.rl_n >= 6 and -0.05 < self.rl_a < 0.95
        dt = 0.02
        # 1) 右线状态：看得见就直接用，看不见（被板挡住）用陀螺仪+里程推算，最多 av4_dr_max_m
        if a_ok and self.o_on and getattr(self, "r_yl", None) is not None:
            yl_new = wrap(self.od_yaw - math.atan(self.rl_b))
            yaw_bad = abs(wrap(yl_new - self.r_yl)) > math.radians(float(g("av4_yaw_gate_deg").value))
            jump = abs(self.rl_a - self.r_d) > float(g("av4_jump_m").value)
            # 方向对、只是离线距离对不上：多半是推算漂了 → 连续几帧一致就重新对准
            self.r_reacq = (getattr(self, "r_reacq", 0) + 1) if (jump and not yaw_bad) else 0
            if jump and not yaw_bad and self.r_reacq >= int(g("av4_reacq_n").value):
                self.r_d = self.rl_a
                jump = False
                self.get_logger().warn(f"[andy][贴线] 右线重新对准 离线={self.rl_a:.2f}")
            bad = yaw_bad or jump
            if bad:
                a_ok = False   # 不再「推算太久就强行接受」：实跑强行接受了一条偏 65° 的线，车跟着冲出赛道                                      # 线被挡板挡住/带偏（实跑出现 θ线=-60°）
        if a_ok:
            yl_new = wrap(self.od_yaw - math.atan(self.rl_b))     # 线在世界里的方向
            if self.o_on and getattr(self, "r_yl", None) is not None and self.r_dr_s0 is None:
                self.r_yl = wrap(self.r_yl + 0.25 * wrap(yl_new - self.r_yl))
                self.r_d += 0.5 * (self.rl_a - self.r_d)
            else:
                self.r_yl = yl_new
                self.r_d = self.rl_a                              # 相机处到右线（右正）= 车离右线
            self.r_dr_s0 = None
        elif getattr(self, "r_yl", None) is not None and self.o_on:
            if self.r_dr_s0 is None:
                self.r_dr_s0 = self.od_s
            self.r_d += self.od_v * math.sin(wrap(self.od_yaw - self.r_yl)) * dt
        have_line = a_ok or (self.o_on and getattr(self, "r_dr_s0", None) is not None
                             and self.od_s - self.r_dr_s0 < float(g("av4_dr_max_m").value)
                             + (float(g("av4_dr_tail_m").value)
                                if self.od_s >= getattr(self, "o_hold_s", 1e9) else 0.0))

        # 2) 挡板入库：左右沿离右线的距离
        if a_ok and self.det_time > self.o_det_seen:
            self.o_det_seen = self.det_time
            for it in self.det_items:
                if it.class_id != 2:
                    continue
                vb = it.cy + it.h / 2.0
                if vb > 0.985:
                    continue
                pl = self._ipm(it.cx - it.w / 2.0, vb)
                pr = self._ipm(it.cx + it.w / 2.0, vb)
                if pl is None or pr is None:
                    continue
                d = 0.5 * (pl[0] + pr[0])
                if not (float(g("av4_meas_min_m").value) <= d <= float(g("av4_meas_max_m").value)):
                    continue
                line_lat = -(self.rl_a + self.rl_b * d)            # 右线在该距离处的横向（左正）
                sL = pl[1] - line_lat if (it.cx - it.w / 2.0) > 0.005 else None
                sR = pr[1] - line_lat if (it.cx + it.w / 2.0) < 0.995 else None
                if sR is not None and sR > float(g("av4_far_m").value):
                    continue                                       # 离右线太远，不是本车道的
                if sR is None and sL is not None and sL - 0.40 > float(g("av4_far_m").value):
                    continue
                sb = self.od_s + d
                best = None
                for b in self.o_bars:
                    if abs(b["s"] - sb) < float(g("av3_assoc_m").value) and \
                            (best is None or abs(b["s"] - sb) < abs(best["s"] - sb)):
                        best = b
                if best is None:
                    best = {"s": sb, "sL": sL, "sR": sR, "n": 0, "side": 0}
                    self.o_bars.append(best)
                else:
                    best["s"] += 0.3 * (sb - best["s"])
                    for k, v in (("sL", sL), ("sR", sR)):
                        if v is not None:
                            best[k] = v if best[k] is None else best[k] + 0.3 * (v - best[k])
                best["n"] += 1
        self.o_bars = [b for b in self.o_bars if b["s"] > self.od_s - 1.0]

        # 3) 目标离线距离 D
        pass_m = float(g("av4_pass_m").value)
        ahead = sorted([b for b in self.o_bars if b["n"] >= int(g("av3_min_hits").value)
                        and b["s"] + pass_m > self.od_s
                        and b["s"] - self.od_s < float(g("av3_trigger_m").value)], key=lambda b: b["s"])
        D = D0
        d_right_min = half + float(g("av4_line_margin").value)

        def edges(bb):
            sR = bb["sR"] if bb["sR"] is not None else (bb["sL"] - 0.40 if bb["sL"] is not None else 0.0)
            sL = bb["sL"] if bb["sL"] is not None else sR + 0.40
            return sL, sR

        # 每块都按最新估计决定走哪边：远处（弯道出口外推）测得不准，锁定距离外带 4 cm 滞回随时改，
        # 进入锁定距离后只在明显过不去时改
        for bb in ahead:
            sL, sR = edges(bb)
            d_right_max = sR - half - mg
            far = bb["s"] - self.od_s > float(g("av4_lock_m").value)
            if bb["side"] == 0:
                want = -1 if d_right_max >= d_right_min else 1
            elif bb["side"] < 0:
                want = 1 if d_right_max < d_right_min - 0.04 else -1
            else:
                want = -1 if (far and d_right_max >= d_right_min + 0.04) else 1
            if want != bb["side"]:
                self.get_logger().warn(
                    f"[andy][贴线] 挡板 前{bb['s'] - self.od_s:.2f} 左沿离线{sL:.2f} 右沿离线{sR:.2f}"
                    f" → 走{'右' if want < 0 else '左'}边")
                bb["side"] = want
        if ahead:
            b = ahead[0]
            sL, sR = edges(b)
            d_right_max = sR - half - mg
            if b["side"] < 0:
                D = clamp(D0, d_right_min, max(d_right_min, d_right_max))
                # 下一块要从左边绕 → 这一块尽量贴着它右沿过，少横移（阿克曼车 0.8 m 内只横移得了 ~0.3 m）
                nxt = ahead[1] if len(ahead) > 1 else None
                if nxt is not None and nxt["side"] > 0 and nxt["s"] - b["s"] < 1.4:
                    D = max(D, d_right_max - 0.02)
            else:
                D = min(sL + half + mg, float(g("av4_d_max").value))
        ds = max(0.0, self.od_s - getattr(self, "_o_s_prev", self.od_s))
        self._o_s_prev = self.od_s
        if not hasattr(self, "r_D"):
            self.r_D = D0
        step = float(g("av3_slew").value) * max(ds, 0.002)
        self.r_D += clamp(D - self.r_D, -step, step)

        was = self.o_on
        if ahead:
            self.o_hold_s = max(getattr(self, "o_hold_s", 0.0), ahead[-1]["s"] + pass_m)
        settling = was and abs(self.r_D - D0) > 0.02
        # 交还前：右线连续可信且车头和线夹角小，否则继续贴线（实跑交还时车头偏 30° → 巡线丢线绕圈）
        ok_now = a_ok and abs(wrap(self.r_yl - self.od_yaw)) < math.radians(float(g("av4_exit_deg").value)) \
            and abs(self.r_d - D0) < 0.06
        self.r_good = (getattr(self, "r_good", 0) + 1) if ok_now else 0
        hold = getattr(self, "o_hold_s", 0.0)
        if was and not ahead and self.od_s >= hold and not settling \
                and self.r_good < int(g("av4_exit_good").value) \
                and self.od_s < hold + float(g("av4_exit_max_m").value):
            settling = True
        self.o_on = have_line and (bool(ahead) or self.od_s < hold or settling)
        ret_lane = False
        if was:
            self.o_nseen = max(getattr(self, "o_nseen", 0),
                               sum(1 for b in self.o_bars if b.get("n", 0) >= int(g("av3_min_hits").value)))
        if was and bool(g("av4_ret_lane").value) and not ahead:
            hits = [b["s"] for b in self.o_bars if b.get("n", 0) >= int(g("av3_min_hits").value)]
            last_s = max(hits) if hits else None
            nb = getattr(self, "o_nseen", 0)
            after = float(g("av4_ret_after_m").value) if nb >= int(g("av4_ret_bars").value) \
                else float(g("av4_ret_far_m").value)
            if last_s is not None and self.od_s >= last_s + after \
                    and self.r_tg >= int(g("av4_ret_n").value):
                ret_lane = True
                self.o_on = False
        if not self.o_on:
            self.o_nseen = 0
            if was and ret_lane:
                self.get_logger().warn(
                    f"[andy][贴线] 双线车道已看清（e={msg.lateral_error:+.2f} θ={math.degrees(msg.heading_error):+.0f}° "
                    f"w={msg.lane_width:.2f}），直接交还巡线")
                self.r_D = D0
                self.av_post_until = self.od_s + float(g("av4_post_m").value)
                return
            if was:
                self.av_post_until = self.od_s + float(g("av4_post_m").value)
                self.get_logger().warn("[andy][贴线] 结束，交还巡线")
                # 交还时车头若还没摆正 / 线还没看清 → 先按线方向锁航向（LOSTYAW），巡线连续可信后再退出
                if getattr(self, "r_yl", None) is not None and \
                        (self.r_good < int(g("av4_exit_good").value) or not a_ok):
                    self._lane_yaw = self.r_yl
                    self.mode = "LOSTYAW"
                    self.ly_ref = self.r_yl
                    self.ly_s0 = self.od_s
                    self.ly_from_av = True
                    self._ly_good = 0
                    self._ok_streak = 0
                    self.get_logger().warn(
                        f"[andy][贴线] 车头偏线 {math.degrees(wrap(self.od_yaw - self.r_yl)):+.0f}°，先锁航向")
            self.r_D = D0
            return
        if not was:
            self.get_logger().warn("[andy][贴线] 进入避障区")
        msg.is_lost = False
        msg.quality = 1.0
        msg.curvature = 0.0
        th = wrap(self.r_yl - self.od_yaw)
        amax = math.radians(float(g("av4_max_ang_deg").value))
        try:
            k = float(g("stanley_k").value)
        except Exception:
            k = 0.7
        # Stanley 稳态：车头相对线的角 = atan(k·e/v)，把 e 限幅让横移角 ≤ amax
        e_cap = math.tan(amax) * max(self.od_v, 0.25) / max(k, 0.1)
        e_cmd = clamp(self.r_d - self.r_D, -e_cap, e_cap)
        if abs(th) > amax + math.radians(6.0):
            e_cmd = 0.0                                           # 已经斜得太多：先把车头摆回来
        msg.lateral_error = float(e_cmd)
        msg.heading_error = float(th)
        self.state = "RLINE"
        self.get_logger().info(
            f"[andy][贴线] 离线{self.r_d:.2f}{'' if a_ok else '(推算)'} 目标{self.r_D:.2f}→{D:.2f}"
            f" θ线={math.degrees(wrap(self.r_yl - self.od_yaw)):+.0f}° 挡板="
            + str(["前%.2f 左%s 右%s %s" % (b["s"] - self.od_s,
                   "%.2f" % b["sL"] if b["sL"] is not None else "?",
                   "%.2f" % b["sR"] if b["sR"] is not None else "?",
                   {0: "?", 1: "左", -1: "右"}[b["side"]]) for b in self.o_bars]),
            throttle_duration_sec=0.3)

    # ─────────────────────────────────────────────────────────
    def _ipm(self, u_n, v_n):
        """归一化像素 → 地面 (前, 左)，和 lane_tracker 同一套针孔+俯仰+偏航。"""
        g = self.get_parameter
        W, H = float(g("av3_img_w").value), float(g("av3_img_h").value)
        f = float(g("av3_cam_f_px").value)
        x, y = (u_n * W - W / 2.0) / f, (v_n * H - H / 2.0) / f
        p = math.radians(float(g("av3_cam_pitch_deg").value))
        y2 = y * math.cos(p) + math.sin(p)
        z2 = -y * math.sin(p) + math.cos(p)
        if y2 <= 1e-3:
            return None
        t = float(g("av3_cam_h").value) / y2
        X, Z = x * t, z2 * t
        yw = math.radians(float(g("av3_cam_yaw_deg").value))
        fwd = Z * math.cos(yw) - X * math.sin(yw)
        left = -Z * math.sin(yw) - X * math.cos(yw)
        return fwd, left

    def _step_offset(self, now, msg):
        g = self.get_parameter
        # 不再依赖斑马线计数（斑马线没触发时整段避障都不会开）；场边杂物靠「相对车道」门限过滤
        if self.od_s < float(g("av3_min_s").value):
            return
        ok = (not msg.is_lost) and msg.quality > 0.4
        e_v, th_v, k_v = float(msg.lateral_error), float(msg.heading_error), float(msg.curvature)
        # 0) 车道坐标系滤波：车道航向 f_yl = 陀螺仪航向 + 视觉θ（这个和很稳），横向 f_e 用里程计推算 + 视觉校正。
        #    避障区是宽车道，视觉 e 会在「锁左线/锁右线」两种解之间跳 ±0.2~0.5 m（实测），
        #    所以偏置中视觉 e 只在和推算值接近时小增益校正，跳变直接丢掉。
        dt = 0.02
        if getattr(self, "f_yl", None) is None:
            self.f_yl, self.f_e = wrap(self.od_yaw + th_v), e_v
        self.f_e += self.od_v * math.sin(wrap(self.od_yaw - self.f_yl)) * dt
        if ok:
            yl = wrap(self.od_yaw + th_v)
            dyl = wrap(yl - self.f_yl)
            if not self.o_on:
                self.f_yl = wrap(self.f_yl + 0.15 * dyl)
                self.f_e = e_v
            else:
                if abs(dyl) < math.radians(12):
                    self.f_yl = wrap(self.f_yl + float(g("av3_yl_gain").value) * dyl)
                if abs(e_v - self.f_e) < float(g("av3_e_gate").value):
                    self.f_e += float(g("av3_e_gain").value) * (e_v - self.f_e)
        e = self.f_e
        th = wrap(self.f_yl - self.od_yaw)
        k = k_v if not self.o_on else 0.0
        # 1) 挡板入库（相对车道）
        if (ok or self.o_on) and self.det_time > self.o_det_seen and abs(th) < math.radians(40):
            self.o_det_seen = self.det_time
            for it in self.det_items:
                if it.class_id != 2 or it.dist_m <= 0.0:
                    continue
                if bool(g("av3_use_ipm").value):
                    gp = self._ipm(it.cx, it.cy + it.h / 2.0)
                    if gp is None or (it.cy + it.h / 2.0) > 0.985:
                        continue
                    d, lat_b = gp
                else:
                    d = float(it.dist_m)
                    lat_b = d * math.tan(float(it.bearing_rad))
                if not (float(g("av3_meas_min_m").value) <= d <= float(g("av3_meas_max_m").value)):
                    continue
                if (it.cx - it.w / 2.0) < 0.01 or (it.cx + it.w / 2.0) > 0.99:
                    continue
                lat_c = -e + d * math.tan(th) + 0.5 * k * d * d
                rel = lat_b - lat_c
                if abs(rel) > float(g("av3_rel_gate_m").value):
                    continue
                sb = self.od_s + d
                best = None
                for b in self.o_bars:
                    if abs(b["s"] - sb) < float(g("av3_assoc_m").value) and \
                            (best is None or abs(b["s"] - sb) < abs(best["s"] - sb)):
                        best = b
                if best is None:
                    self.o_bars.append({"s": sb, "rel": rel, "n": 1, "side": 0})
                else:
                    a = 0.35
                    best["s"] += a * (sb - best["s"])
                    # 已经开始往一边偏了就冻结横向：车偏开后挡板在画面侧面，测距比例误差会直接变成横向误差
                    if not (best["side"] and abs(self.o_y) > 0.08):
                        best["rel"] += a * (rel - best["rel"])
                    best["n"] += 1
        self.o_bars = [b for b in self.o_bars if b["s"] > self.od_s - 1.0]

        # 2) 目标偏置
        pass_m = float(g("av3_pass_m").value)
        ahead = sorted([b for b in self.o_bars if b["n"] >= int(g("av3_min_hits").value)
                        and b["s"] + pass_m > self.od_s
                        and b["s"] - self.od_s < float(g("av3_trigger_m").value)], key=lambda b: b["s"])
        clear, ymax = float(g("av3_clear_m").value), float(g("av3_y_max").value)
        if ahead:
            b = ahead[0]
            nxt = ahead[1] if len(ahead) > 1 else None
            def shortfall(bb, sd):
                yy = clamp(bb["rel"] + sd * clear, -ymax, ymax)
                return max(0.0, clear - abs(yy - bb["rel"])), yy
            sf_cur = shortfall(b, b["side"])[0] if b["side"] else 9.0
            if b["side"] == 0 or sf_cur > 0.06:
                best = None
                for s1 in (1, -1):
                    sf1, y1 = shortfall(b, s1)
                    cost = 6.0 * sf1 + 0.4 * abs(y1 - self.o_y)
                    if nxt is not None:
                        c2 = min(6.0 * shortfall(nxt, s2)[0] + 0.4 * abs(shortfall(nxt, s2)[1] - y1)
                                 / max(nxt["s"] - b["s"], 0.4) for s2 in (1, -1))
                        cost += c2
                    if best is None or cost < best[0]:
                        best = (cost, s1)
                if b["side"] != best[1]:
                    self.get_logger().warn(f"[andy][偏置] 挡板 前{b['s'] - self.od_s:.2f} 相对车道{b['rel']:+.2f} → 走{'左' if best[1] > 0 else '右'}边")
                b["side"] = best[1]
            y_t = clamp(b["rel"] + b["side"] * clear, -ymax, ymax)
        else:
            y_t = 0.0
        ds = max(0.0, self.od_s - getattr(self, "_o_s_prev", self.od_s))
        self._o_s_prev = self.od_s
        step = float(g("av3_slew").value) * max(ds, 0.002)
        self.o_y += clamp(y_t - self.o_y, -step, step)
        was = self.o_on
        if ahead:
            self.o_hold_s = max(getattr(self, "o_hold_s", 0.0),
                                ahead[-1]["s"] + pass_m + float(g("av3_hold_m").value))
        # 交还视觉：要视觉 e 连续若干帧和推算一致（说明已经出了宽车道、视觉又靠谱了），最多再推算 av3_dr_max_m
        if ok and abs(e_v - self.f_e) < float(g("av3_e_gate").value):
            self._o_cons = getattr(self, "_o_cons", 0) + 1
        else:
            self._o_cons = 0
        hold = getattr(self, "o_hold_s", 0.0)
        waiting = was and self.od_s < hold + float(g("av3_dr_max_m").value) and self._o_cons < 12
        self.o_on = bool(ahead) or abs(self.o_y) > 0.015 or self.od_s < hold or waiting
        if not self.o_on:
            if was:
                self.get_logger().warn("[andy][偏置] 回到车道中心，结束避障")
            self.o_y = 0.0
            self.o_dr_s0 = None
            return
        if not was:
            self.get_logger().warn("[andy][偏置] 开始避障")

        # 3) 整段偏置都用滤波后的车道坐标喂给原巡线控制器
        msg.is_lost = False
        msg.quality = 1.0
        msg.curvature = 0.0
        msg.heading_error = float(th)
        msg.lateral_error = float(e - self.o_y)
        self.state = "OFFSET"
        self.get_logger().info(
            f"[andy][偏置] y={self.o_y:+.2f}→{y_t:+.2f} e滤={e:+.2f} e视={e_v:+.2f}{'' if ok else '(丢)'} θ={math.degrees(th):+.0f}° 挡板="
            + str(["前%.2f 相对%+.2f %s" % (b["s"] - self.od_s, b["rel"], "左右"[b["side"] < 0] if b["side"] else "?")
                   for b in self.o_bars]),
            throttle_duration_sec=0.3)

    # ─────────────────────────────────────────────────────────
    def _step_avoid(self, now):
        g = self.get_parameter
        trig = float(g("av2_trigger_m").value)
        if self.mode != "AVOID":
            if self.mode not in ("LANE", "LOSTYAW"):
                return False
            if bool(g("av2_require_cw").value) and self.n_cw < 1:
                return False
            if self.barriers and now - getattr(self, "_bar_dbg_t", 0.0) > 0.3:
                self._bar_dbg_t = now
                yr0 = self._lane_yaw if self._lane_yaw is not None else self.od_yaw
                self.get_logger().info(
                    f"[andy][看见挡板] 直道帧={getattr(self, '_straight_n', 0)} s={self.od_s:.2f} "
                    + " ".join(f"(前{self._fwd_of(b, yr0):.2f},左{self._lat_of(b, yr0):+.2f})" for b in self.barriers))
            if getattr(self, "_straight_n", 0) < int(g("av2_straight_frames").value):
                return False
            yr = self._lane_yaw if self._lane_yaw is not None else self.od_yaw
            lat_gate = float(g("av2_lat_gate_m").value)
            ahead = [b for b in self.barriers if 0.0 < self._fwd_of(b, yr) < trig
                     and abs(self._lat_of(b, yr)) < lat_gate]
            if not ahead:
                return False
            yr = math.atan2(self._ly_s, self._ly_c)   # 直道上平均出来的车道方向
            self.mode = "AVOID"
            self.av_ref = (self.od_x, self.od_y, yr)
            self.av_s0 = self.od_s
            for b in self.barriers:
                b["side"] = 0
            self.get_logger().warn(f"[andy] 进入避障，前方 {len(ahead)} 块")

        m = getattr(self, "_msg", None)
        if m is not None and not m.is_lost and m.quality > float(g("av2_straight_q").value) \
                and abs(m.curvature) < float(g("av2_straight_kappa").value):
            x0, y0, yr0 = self.av_ref
            d = wrap(self._lane_yaw_now() - yr0)
            if abs(d) < math.radians(float(g("av2_ref_max_dev_deg").value)):
                self.av_ref = (x0, y0, wrap(yr0 + float(g("av2_ref_alpha").value) * d))
        Xc, Yc = self._to_ref(self.od_x, self.od_y)
        psi = wrap(self.od_yaw - self.av_ref[2])
        pass_m = float(g("av2_pass_m").value)
        bl = []
        for b in self.barriers:
            Xb, Yb = self._to_ref(b["X"], b["Y"])
            if Xc < Xb + pass_m and Xb - Xc < trig + 1.5 \
                    and abs(Yb) < float(g("av2_lat_gate_m").value):
                bl.append((Xb, Yb, b))
        bl.sort(key=lambda z: z[0])

        clear = (float(g("av2_barrier_half_w").value) + float(g("av2_car_half_w").value)
                 + float(g("av2_margin_m").value))
        ymax = float(g("av2_y_max").value)
        if bl and bl[0][0] - Xc < trig:
            Xb, Yb, b = bl[0]
            y_cur = clamp(Yb + b["side"] * clear, -ymax, ymax) if b["side"] else 0.0
            infeasible = b["side"] != 0 and abs(y_cur - Yb) < clear - 0.12
            if infeasible:
                self.get_logger().warn(f"[andy] 挡板位置更新后原来那边过不去了（间隙 {abs(y_cur - Yb):.2f}），重新选边")
            if b["side"] == 0 or infeasible or Xb - Xc > float(g("av2_side_lock_m").value):
                # 和下一块一起规划：枚举 (本块走哪边, 下一块走哪边)，
                # 代价 = −最小间隙 + 两块之间的横移量（间距越近越贵）+ 现在要横移多少
                nxt = bl[1] if len(bl) > 1 else None
                best = None
                for s1 in (1, -1):              # 1 = 从挡板左边过, -1 = 右边
                    y1 = clamp(Yb + s1 * clear, -ymax, ymax)
                    g1 = abs(y1 - Yb)
                    for s2 in ((1, -1) if nxt else (0,)):
                        if nxt:
                            y2 = clamp(nxt[1] + s2 * clear, -ymax, ymax)
                            g2 = abs(y2 - nxt[1])
                            gap_x = max(nxt[0] - Xb, 0.3)
                            hop = abs(y2 - y1) * (0.6 / gap_x)
                        else:
                            g2, hop = 9.0, 0.0
                        cost = -min(g1, g2) + hop + abs(y1 - Yc) * (0.6 / max(Xb - Xc, 0.3))
                        if s1 == b["side"]:
                            cost -= float(g("av2_flip_hyst").value)
                        if best is None or cost < best[0]:
                            best = (cost, s1)
                b["side"] = best[1]
            y_t = clamp(Yb + b["side"] * clear, -ymax, ymax)
            phase = "PASS%+d" % b["side"]
        else:
            y_t = 0.0
            phase = "RETURN"
            if abs(Yc) < float(g("av2_return_tol_m").value) and abs(psi) < math.radians(12.0):
                self.get_logger().warn("[andy] 避障完成，交还巡线")
                self.mode = "LANE"
                self.e_prev = None
                self._lane_yaw = None
                return False
        if abs(psi) > math.radians(float(g("av2_abort_psi_deg").value)):
            self.get_logger().error(f"[andy] 避障中车头偏 {math.degrees(psi):.0f}°，放弃避障交还巡线")
            self.mode = "LANE"
            self.e_prev = None
            self.barriers = []
            return False
        if self.od_s - self.av_s0 > float(g("av2_max_len_m").value):
            self.get_logger().warn("[andy] 避障超长，强制交还巡线")
            self.mode = "LANE"
            self.e_prev = None
            self.barriers = []
            return False
        self.av_y_t = y_t
        Ld = float(g("av2_lookahead_m").value)
        pmax = math.radians(float(g("av2_max_psi_deg").value))
        psi_des = clamp(math.atan2(y_t - Yc, Ld), -pmax, pmax)
        alpha = clamp(wrap(psi_des - psi), -math.radians(60), math.radians(60))
        L = float(g("wheelbase_m").value)
        dw = math.atan2(2.0 * L * math.sin(alpha), Ld)
        self._emit(float(g("av2_speed").value), dw)
        self.state = "AVOID_" + phase
        self._state_msg.data = "AVOID %s Yc=%+.2f y_t=%+.2f n=%d" % (phase, Yc, y_t, len(bl))
        self.pub_state.publish(self._state_msg)
        self.get_logger().info(
            "[andy][避障] %s 车Y=%+.2f 目标Y=%+.2f ψ=%+.0f° 挡板=%s" % (
                phase, Yc, y_t, math.degrees(psi),
                ["(%.2f,%+.2f)" % (x - Xc, y) for x, y, _ in bl]),
            throttle_duration_sec=0.5)
        return True

    # ─────────────────────────────────────────────────────────
    def _step_xing(self, s_cw):
        g = self.get_parameter
        win = list(g("xing_windows").value)
        if s_cw < 0.0 or len(win) < 2:
            if self.mode == "XING":
                self.mode = "LANE"
            return False
        inside = None
        for i in range(0, len(win) - 1, 2):
            a, l = float(win[i]), float(win[i + 1])
            if a > 0.0 and a <= s_cw < a + l:
                inside = (a, l)
                break
        if inside is None:
            if self.mode == "XING":
                self.get_logger().warn(f"[andy] 出路口（s_cw={s_cw:.2f}）")
                self.mode = "LANE"
                self.e_prev = None
            return False
        if self.mode != "XING":
            self.mode = "XING"
            self.xing_ref = self._lane_yaw if self._lane_yaw is not None else self.od_yaw
            self.get_logger().warn(f"[andy] 进路口直行（s_cw={s_cw:.2f}，窗口 {inside}）")
        err = wrap(self.od_yaw - self.xing_ref)
        dw = -float(g("xing_k_yaw").value) * err
        self._emit(float(g("xing_speed").value), dw)
        self.state = "XING"
        return True

    # ─────────────────────────────────────────────────────────
    def _step_lostyaw(self, msg):
        g = self.get_parameter
        if not bool(g("lost_yaw_hold").value):
            return False
        if self.mode == "LOSTYAW":
            if getattr(self, "ly_from_av", False):
                # 避障后的航向锁：巡线要「真看清」才退出（实跑：3 帧非丢线就退出，巡线锁到假线 → 车头偏 40° 出赛道）
                good = (not msg.is_lost) and msg.quality >= 0.5 and abs(msg.lateral_error) < 0.15 \
                    and abs(msg.heading_error) < math.radians(12.0)
                self._ly_good = (getattr(self, "_ly_good", 0) + 1) if good else 0
                aligned = abs(wrap(self.od_yaw - self.ly_ref)) < math.radians(8.0)
                done = (self._ly_good >= int(g("lostyaw_av_good").value) and aligned) or \
                    self.od_s - self.ly_s0 > float(g("lostyaw_av_max_m").value)
                if done:
                    self.get_logger().warn(
                        f"[andy][贴线] 航向锁结束（{'巡线可信' if self._ly_good else '走满距离'}），交还巡线")
            else:
                done = self._ok_streak >= int(g("lost_exit_frames").value) or \
                    self.od_s - self.ly_s0 > float(g("lostyaw_max_m").value)
            if done:
                self.mode = "LANE"
                self.ly_from_av = False
                return False
        else:
            if self._lost_streak < int(g("lost_enter_frames").value) or self._lane_yaw is None:
                return False
            k = abs(self.k_s) if self.k_s is not None else 9.9
            rec = list(self.steer_hist)[-10:]
            st = max((abs(x) for x in rec), default=9.9)
            if k > float(g("lost_straight_kappa").value) or \
                    st > math.radians(float(g("lost_straight_steer_deg").value)):
                return False
            self.mode = "LOSTYAW"
            self.ly_ref = self._lane_yaw
            self.ly_s0 = self.od_s
            self.ly_from_av = False
        err = wrap(self.od_yaw - self.ly_ref)
        dw = -float(g("xing_k_yaw").value) * err
        self._emit(max(self.v_cmd, float(g("lost_speed").value)), dw)
        self.state = "LOSTYAW"
        return True

    # ─────────────────────────────────────────────────────────
    def _log(self, now, msg, s_cw):
        self.pub_mode.publish(String(data=f"{self.mode} lap_cw={self.n_cw} s={self.od_s:.2f} "
                                          f"s_cw={s_cw:.2f} yaw={math.degrees(self.od_yaw):+.0f}"
                                          + (f" D={self.r_D:.2f} dR={self.r_d:.2f}" if getattr(self, "o_on", False)
                                             and hasattr(self, "r_d") and self.mode == "RLINE" else "")))
        if self._csv is None or now - self._csv_t < 0.05:
            return
        self._csv_t = now
        try:
            n_bar = sum(1 for it in self.det_items if it.class_id == 2)
            r = (msg.reject_reason or "").replace(",", ";")[:60]
            self._csv.write(
                f"{now - self.t0:.2f},{self.od_s:.3f},{s_cw:.3f},{self.od_x:.3f},{self.od_y:.3f},"
                f"{math.degrees(self.od_yaw):.1f},{self.od_v:.3f},{self.od_gz:.3f},{self.mode},"
                f"{self.state},{msg.lateral_error:.3f},{math.degrees(msg.heading_error):.1f},"
                f"{msg.curvature:.2f},{msg.lane_width:.2f},{msg.quality:.2f},{int(msg.is_lost)},"
                f"{n_bar},{math.degrees(self.steer_prev):.1f},{self.cw:.1f},{r}\n")
            if now - getattr(self, "_csv_fs", 0.0) > 1.0:   # 防断电丢日志
                self._csv_fs = now
                os.fsync(self._csv.fileno())
        except Exception:
            pass
