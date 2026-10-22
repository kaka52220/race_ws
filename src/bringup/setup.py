from glob import glob
import os

from setuptools import setup

package_name = "bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="team",
    maintainer_email="team@example.com",
    description="任务二一键发车",
    license="Apache-2.0",
    entry_points={"console_scripts": []},
)
