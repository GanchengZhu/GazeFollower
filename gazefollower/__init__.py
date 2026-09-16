# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com
import os

# Disable GPU initialization for headless / CPU environments
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

from .GazeFollower import GazeFollower
from .multiprocess import MultiprocessGazeFollower

__all__ = ["GazeFollower", "MultiprocessGazeFollower"]
