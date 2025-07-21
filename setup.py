# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import os

from setuptools import setup, find_packages

package_name = 'gazefollower'
build_file = 'build_number.txt'


def get_build_number():
    if os.path.exists(build_file):
        with open(build_file, 'r') as f:
            build_number = int(f.read().strip())
    else:
        build_number = 0
    build_number += 1
    # 将新的 build 号写入文件
    with open(build_file, 'w') as f:
        f.write(str(build_number))
    return build_number


build_number = get_build_number()

from gazefollower import version

major_version, minor_version, patch_version = version.__version__.split(".")

setup(
    name=package_name,
    version=f"{major_version}.{minor_version}.{patch_version}",
    author=version.__author__,
    author_email=version.__email__,
    description=version.__description__,
    url=version.__url__,
    packages=find_packages(),
    long_description=open('README.md').read(),  # 或者使用其他文档文件
    long_description_content_type='text/markdown',  # 如果使用 Markdown 格式
    package_data={
        package_name: ['res/audio/*', 'res/image/*', 'res/model_weights/base.mnn'],
    },

    install_requires=[
        'mediapipe',
        'MNN',
        'numpy',
        'opencv-python',
        'pandas',
        'pygame',
        'screeninfo',
    ],

    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
    ],
    python_requires='>=3.7',  # Specify the required Python version

)
