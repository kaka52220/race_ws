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


def generate_launch_description():
    args = [
        DeclareLaunchArgument("camera_device", default_value="/dev/video0",
                              description="摄像头设备节点"),
        DeclareLaunchArgument("width", default_value="1280"),
        DeclareLaunchArgument("height", default_value="720"),
        DeclareLaunchArgument("dry_run", default_value="false",
                              description="true = 只算不发 /cmd_vel"),
        DeclareLaunchArgument("max_speed", default_value="0.30"),
        DeclareLaunchArgument("speed_scale", default_value="1.0"),
        DeclareLaunchArgument("publish_debug", default_value="true"),
        DeclareLaunchArgument("base_driver", default_value="false",
                              description="是否一起起底盘节点"),
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
        }],
        remappings=[("image_raw", "/camera/image_raw")],
        # 摄像头掉线自动重启（USB 掉线是常事）
        respawn=True, respawn_delay=2.0,
    )

    # ② 视觉：订阅图像，发 /track
    vision = Node(
        package="vision_perception",
        executable="vision_node",
        name="vision_node",
        output="screen",
        parameters=[{"publish_debug": LaunchConfiguration("publish_debug")}],
        respawn=True, respawn_delay=1.0,
    )

    # ③ 大脑：订阅 /track，发 /cmd_vel
    brain = Node(
        package="race_decision",
        executable="brain_node",
        name="brain_node",
        output="screen",
        parameters=[{
            "dry_run": LaunchConfiguration("dry_run"),
            "max_speed": LaunchConfiguration("max_speed"),
            "speed_scale": LaunchConfiguration("speed_scale"),
        }],
        respawn=True, respawn_delay=1.0,
    )

    # ④ 底盘（可选）
    base = Node(
        package="base_driver",
        executable="base_node",
        name="base_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("base_driver")),
    )

    return LaunchDescription(args + [
        LogInfo(msg="=== 任务二发车 ==="),
        camera, vision, brain, base,
    ])
