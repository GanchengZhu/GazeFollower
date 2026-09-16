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

version_dict = {}
version_file = os.path.join(os.path.dirname(__file__), 'gazefollower', 'version.py')
with open(version_file, 'r', encoding='utf-8') as f:
    exec(f.read(), version_dict)

setup(
    name=package_name,
    version=version_dict['__version__'],
    author=version_dict['__author__'],
    author_email=version_dict['__email__'],
    description=version_dict['__description__'],
    url=version_dict['__url__'],
    packages=find_packages(),
    long_description=open('README.md', encoding='utf-8').read(),  # 或者使用其他文档文件
    long_description_content_type='text/markdown',  # 如果使用 Markdown 格式
    package_data={
        package_name: ['res/audio/*', 'res/image/*', 'res/model_weights/*'],
    },
    include_package_data=True,

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
