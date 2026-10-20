"""把 semphyspc 注册成 ament_python 包，只暴露 launch/params 供 ros2 launch 使用。

semphyspc 本身是训练代码目录（mamba/models/data/…），这里不打包任何 Python 模块，
只把 launch/*.launch.py 与 params/*.yaml 复制到 install/share/semphyspc，
使 `ros2 launch semphyspc <file>` 与 `FindPackageShare('semphyspc')` 可用。
"""
from glob import glob

from setuptools import setup

package_name = 'semphyspc'

setup(
    name=package_name,
    version='0.0.1',
    packages=[],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/params', glob('params/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='saw',
    maintainer_email='saw@todo.todo',
    description='SemPhysPC semantic-physical navigation (launch/params only)',
    license='MIT',
)
