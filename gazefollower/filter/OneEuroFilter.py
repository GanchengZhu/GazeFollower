# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import numpy as np
from .Filter import Filter


class LowPassFilter:
    """
    Exponential moving average low-pass filter supporting both scalars and NumPy arrays.
    """

    def __init__(self, alpha: float = 1.0, initval=0.0):
        self.a = float(alpha)
        self.initval = initval
        self.y = None
        self.s = None
        self.initialized = False

    def has_last_raw_value(self) -> bool:
        return self.initialized

    def last_raw_value(self):
        return self.y

    def filter(self, value, alpha=None):
        a = self.a if alpha is None else alpha
        val_arr = np.asarray(value, dtype=np.float64)
        if self.initialized:
            result = a * val_arr + (1.0 - a) * self.s
        else:
            result = val_arr.copy()
            self.initialized = True
        self.y = val_arr.copy()
        self.s = result.copy()
        if np.isscalar(value) or isinstance(value, (int, float)):
            return float(result)
        return result

    def reset(self):
        self.initialized = False
        self.y = None
        self.s = None


class OneEuroFilter(Filter):
    """
    1-Euro Filter (Casiez et al., CHI 2012) with adaptive cutoff frequency.
    Supports scalars, bounding boxes [x, y, w, h], and multidimensional landmark coordinates.

    Parameters:
        freq (float): Sampling frequency in Hz (default 30.0).
        min_cutoff (float): Minimum cutoff frequency in Hz (default 1.0).
        beta (float): Speed coefficient for dynamic cutoff (default 0.01).
        d_cutoff (float): Cutoff frequency for derivative filtering (default 1.0).
        beta_ (float, optional): Alias for beta parameter for backwards compatibility.
    """

    def __init__(self, freq: float = 30.0, min_cutoff: float = 1.0, beta: float = 0.01, d_cutoff: float = 1.0, beta_: float = None):
        super().__init__()
        if beta_ is not None:
            beta = beta_
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_time = None

    def _alpha(self, cutoff):
        te = 1.0 / max(self.freq, 1e-5)
        tau = 1.0 / (2.0 * np.pi * np.maximum(cutoff, 1e-5))
        return 1.0 / (1.0 + tau / te)

    def filter(self, x, timestamp=None):
        """
        Filters input x using adaptive cutoff frequency.

        Args:
            x (float | list | np.ndarray): Input value, vector, or landmark matrix.
            timestamp (float | int, optional): Timestamp in s, ms, or ns.

        Returns:
            np.ndarray | float: Smoothed output of the same shape/type as input.
        """
        is_scalar = np.isscalar(x)
        val_arr = np.asarray(x, dtype=np.float64)

        if timestamp is not None and self.last_time is not None:
            dt = float(timestamp - self.last_time)
            # Automatic timestamp unit detection
            if dt > 1e11:  # nanoseconds
                dt /= 1e9
            elif dt > 1e5:  # microseconds
                dt /= 1e6
            elif dt > 50:  # milliseconds
                dt /= 1e3
            if 0 < dt < 10:  # valid dt range
                self.freq = 1.0 / dt
        self.last_time = timestamp

        if not self.x_filter.initialized:
            dx = np.zeros_like(val_arr)
        else:
            dx = (val_arr - self.x_filter.y) * self.freq

        edx = self.dx_filter.filter(dx, alpha=self._alpha(self.d_cutoff))
        cutoff = self.min_cutoff + self.beta * np.abs(edx)
        alpha = self._alpha(cutoff)
        filtered = self.x_filter.filter(val_arr, alpha=alpha)

        if is_scalar:
            return float(filtered)
        return filtered

    def filter_values(self, values, timestamp=-1):
        """
        Filter a list or array of values (Filter base class interface).
        """
        ts = None if timestamp == -1 else timestamp
        res = self.filter(values, timestamp=ts)
        if isinstance(values, list):
            return res.tolist()
        return res

    def reset(self):
        """
        Resets internal filter states. Call this when tracking is lost or restarted.
        """
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_time = None

    def release(self):
        self.reset()
