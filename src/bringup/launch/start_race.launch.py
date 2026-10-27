# -*- coding: utf-8 -*-
"""
start_race.launch.py —— 一键发车

默认只起三样：摄像头 + 视觉 + 大脑。
底盘 base_driver 由 launch 参数 base_driver 控制（默认 false，因为它是队友写的老包，
话题名万一不一样会白跑，先单独启动确认后再接进来）。

用法：
    ros2 launch bringup start_race.launch.py
    ros2 launch bringup start_race.launch.py dry_run:=true          # 只算不发
    ros2 launch bringup start_race.launch.py max_speed:=0.2 speed_scale:=0.6
    ros2 launch bringup start_race.launch.py camera_device:=/dev/video1
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    args = [
        DeclareLaunchArgument("camera_device", default_value="/dev/video0",
                              description="摄像头设备节点"),
        DeclareLaunchArgument("width", default_value="1280"),
        DeclareLaunchArgument("height", default_value="720"),
        DeclareLaunchArgument("dry_run", default_value="false",
                              description="true = 只算不发 /cmd_vel"),
        DeclareLaunchArgument("max_speed", default_value="0.60",
                              description="最高速，米/秒。每 1% ≈ 0.0134 m/s：0.21=16%，0.30=22%"),
        DeclareLaunchArgument("speed_scale", default_value="1.0"),
        DeclareLaunchArgument("publish_debug", default_value="true"),
        DeclareLaunchArgument("enable_detect", default_value="true",
                              description="是否起 NPU 挡板检测节点"),
        DeclareLaunchArgument("publish_crosswalk", default_value="true",
                              description="视觉是否发 /crosswalk（测试时可关掉，自己灌假值）"),
        DeclareLaunchArgument("base_driver", default_value="true",
                              description="是否一起起底盘节点。只想看视觉时传 false"),
    ]

    # ① 摄像头。用 usb_cam 而不是 v4l2_camera ——
    #    实测这台相机在 v4l2_camera 下只要设 pixel_format:=MJPG 就会崩
    #    （"Unrecognized image encoding []"），不设又只能用 YUYV 640x480。
    #    usb_cam 的 mjpeg2rgb 能正确出 1280x720。
    camera = Node(
        package="usb_cam",
        executable="usb_cam_node_exe",
        name="camera",
        output="screen",
        parameters=[{
            "video_device": LaunchConfiguration("camera_device"),
            "image_width": LaunchConfiguration("width"),
            "image_height": LaunchConfiguration("height"),
            "pixel_format": "mjpeg2rgb",
            "framerate": 30.0,
            # ★ 必须显式指定：usb_cam 默认会写 brightness=50，
            #   把画面整体提亮到 202、地面饱和 233 —— 黄线冲白、白条分不出来，
            #   循迹和斑马线检测都跟着废。实测 brightness=0 才是正常曝光。
            "brightness": 0,
            "autoexposure": False,  # False = 手动曝光（这个参数是 bool，不是 int！）
            "exposure": 60,         # 对应 v4l2 的 exposure_time_absolute
            "gain": 0,
        }],
        remappings=[("image_raw", "/camera/image_raw")],
        # 摄像头掉线自动重启（USB 掉线是常事）
        respawn=True, respawn_delay=2.0,
    )

    # ② 视觉：订阅图像，发 /track
    #   ⚠ publish_debug 必须显式转成 bool：LaunchConfiguration 返回的是字符串，
    #     直接传进去 bool("false") 还是 True，参数根本关不掉。
    #     调试图 visualize() 一次 10~15 ms，跑车时开着会明显拖慢帧率。
    vision = Node(
        package="vision_perception",
        executable="vision_node",
        name="vision_node",
        output="screen",
        parameters=[{
            "publish_debug": ParameterValue(
                LaunchConfiguration("publish_debug"), value_type=bool),
            "publish_crosswalk": ParameterValue(
                LaunchConfiguration("publish_crosswalk"), value_type=bool),
        }],
        respawn=True, respawn_delay=1.0,
    )

    # ②' NPU 挡板检测：订阅同一路图像，发 /detections
    #    单独一个进程，不和 vision_node 抢 CPU（推理 11.7ms + 循迹 13ms 串起来会掉帧）
    detect = Node(
        package="vision_perception",
        executable="detect_node",
        name="detect_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_detect")),
        parameters=[{
            "detect_hz": 15.0,
            # 复用 publish_debug：发 /debug/detect_image（画了框的图，5 Hz）
            "publish_debug": ParameterValue(
                LaunchConfiguration("publish_debug"), value_type=bool),
        }],
        respawn=True, respawn_delay=2.0,
    )

    # ③ 大脑：订阅 /track，发 /cmd_vel
    brain = Node(
        package="race_decision",
        executable="brain_node",
        name="brain_node",
        output="screen",
        # ⚠ LaunchConfiguration 传出来的是**字符串**，直接塞给 double 参数会类型不匹配、
        #   覆盖被静默忽略（2026-10-23 出现过 max_speed 改了不生效）。
        #   必须用 ParameterValue(..., value_type=float) 显式转换类型。
        parameters=[{
            "dry_run": ParameterValue(LaunchConfiguration("dry_run"), value_type=bool),
            "max_speed": ParameterValue(LaunchConfiguration("max_speed"), value_type=float),
            "speed_scale": ParameterValue(LaunchConfiguration("speed_scale"), value_type=float),
        }],
        respawn=True, respawn_delay=1.0,
    )

    # ④ 底盘（默认一起起，否则 /cmd_vel 没人听，车不动）
    #    只想单独看视觉时：ros2 launch ... base_driver:=false
    base = Node(
        package="base_driver",
        executable="base_node",
        name="base_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("base_driver")),
    )

    return LaunchDescription(args + [
        LogInfo(msg="=== 任务二发车 ==="),
        camera, vision, detect, brain, base,
    ])
