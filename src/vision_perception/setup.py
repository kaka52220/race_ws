from glob import glob
import os

from setuptools import setup

package_name = "vision_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="team",
    maintainer_email="team@example.com",
    description="任务二视觉感知",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "vision_node = vision_perception.vision_node:main",
            "detect_node = vision_perception.detect_node:main",
        ],
    },
)
