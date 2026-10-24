# brain_node v2 设计（2026-10-24，方案 B）

## 目标
巡线更稳更快 + 路口直行；YOLO（斑马线/挡板/计圈）留接口。

## 数据流
camera → vision_node → /track (TrackInfo, header.stamp = 图像时间)
base_node → /imu (sensor_msgs/Imu, angular_velocity.z = 偏航角速度 rad/s, 左正)
          → /wheel_speed (geometry_msgs/TwistStamped, linear.x = 实测车速 m/s)
brain_node ← /track /imu /wheel_speed (/detections 预留) → /cmd_vel（约定不变）

## 模块（race_decision 包内，纯 Python，可单测；brain_node.py 只做 ROS 胶水）
- lane_estimator.py  LaneEstimator：状态 x=[e, θ, κ]（TrackInfo 符号，左正）
  预测：de/dt = −v·sinθ ；dθ/dt = v·κ − r ；κ 随机游走
  更新：视觉测量按图像时间戳回推到当前（延迟补偿），马氏距离门限剔除野值，R 随 quality 放大
  丢线/路口：只预测不更新（陀螺仪+车速推算）
- controller.py  Controller：
  外环 纯跟踪：目标点 y(s)= −(e−e_ref) + s·tanθ + κs²/2，κ_cmd = 2y/Ld²，Ld = ld0 + ld_k·v（自带曲率前馈）
  内环 偏航角速度：δ = steer_map(atan(L·κ_cmd)) + kr·(v·κ_cmd − r)（低速自动关），限幅 + 变化率
  速度：v = min(v_max, sqrt(a_lat/|κ_preview|))，按质量/丢线降速，加减速斜坡
- race_fsm.py  RaceFSM：WAIT → CRUISE ⇄ JUNCTION / LOST_HOLD → STOP
  JUNCTION：lane_width > junction_width_m 连续 N 帧触发；κ 置 0、忽略视觉，按估计器直行；
           走过 junction_min_m 后视觉连续 M 帧通过门限即退出；超过 junction_max_m 强制退出
  LOST_HOLD：估计器推算继续走 lost_hold_m，降速；超出则停车
  预留：CROSSWALK_STOP / AVOID（e_ref 平移）状态，/detections 回调空实现
- 无 IMU 时自动退化：r 用 v·tan(δ_wheel)/L 估计，内环关闭

## 验证
- 单测：估计器、控制器、状态机
- 闭环仿真 tools/sim_brain.py：自行车模型 + 赛道（直道/R0.5 弯/S 弯/路口）+ 视觉噪声、延迟 60 ms、野值、路口宽度突变
