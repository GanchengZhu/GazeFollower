# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

from .Filter import Filter


class LowPassFilter:
    """
    A simple low-pass filter that smooths input values over time.

    This filter uses an exponential moving average to filter input values.
    """

    def __init__(self, alpha, initval=0.0):
        """
        Initialize the LowPassFilter with a given alpha and initial value.

        :param alpha: The smoothing factor (should be in (0.0, 1.0]).
        :param initval: The initial value for the filter (default is 0.0).
        """
        self.y = self.s = initval  # Current and smoothed values
        self.a = alpha  # Smoothing factor
        self.initialized = False  # Track if the filter is initialized

    def set_alpha(self, alpha):
        """
        Set the smoothing factor alpha.

        :param alpha: The new smoothing factor (should be in (0.0, 1.0]).
        :raises Exception: If alpha is not in the valid range.
        """
        if not (0.0 < alpha <= 1.0):
            raise Exception("alpha should be in (0.0, 1.0]")
        self.a = alpha

    def filter(self, value):
        """
        Apply the low-pass filter to a new value.

        :param value: The new value to filter.
        :return: The filtered result.
        """
        if self.initialized:
            result = self.a * value + (1.0 - self.a) * self.s
        else:
            result = value
            self.initialized = True

        self.y = value  # Update the last raw value
        self.s = result  # Update the smoothed value
        return result

    def filter_with_alpha(self, value, alpha):
        """
        Apply the low-pass filter with a specified alpha.

        :param value: The new value to filter.
        :param alpha: The smoothing factor for this operation.
        :return: The filtered result.
        """
        self.set_alpha(alpha)
        return self.filter(value)

    def has_last_raw_value(self):
        """
        Check if there is a valid last raw value.

        :return: True if the filter has been initialized, False otherwise.
        """
        return self.initialized

    def last_raw_value(self):
        """
        Get the last raw value processed by the filter.

        :return: The last raw value.
        """
        return self.y


class OneEuroFilter(Filter):
    """
    One Euro Filter for smoothing noisy signals with adaptive cutoff frequency.

    This filter is particularly useful for tracking applications where
    the signal may have high-frequency noise.
    """

    def __init__(self, freq, min_cutoff=1.0, beta_=0.007, d_cutoff=1.0):
        """
        Initialize the OneEuroFilter with frequency and cutoff parameters.

        :param freq: The sampling frequency of the input signal.
        :param min_cutoff: The minimum cutoff frequency for the filter.
        :param beta_: The parameter to adjust the dynamic cutoff.
        :param d_cutoff: The cutoff frequency for the derivative filter.
        """
        super().__init__()
        self.freq = freq  # Sampling frequency
        self.min_cutoff = min_cutoff  # Minimum cutoff frequency
        self.beta = beta_  # Beta parameter
        self.d_cutoff = d_cutoff  # Derivative cutoff frequency
        self.x = LowPassFilter(self.alpha(min_cutoff))  # Low-pass filter for the signal
        self.dx = LowPassFilter(self.alpha(d_cutoff))  # Low-pass filter for the derivative
        self.last_time = -1  # Track the last timestamp

    def alpha(self, cutoff):
        """
        Calculate the alpha value for the low-pass filter based on the cutoff frequency.

        :param cutoff: The cutoff frequency for the filter.
        :return: The calculated alpha value.
        """
        te = 1.0 / self.freq  # Sampling period
        tau = 1.0 / (6.2831855 * cutoff)  # Time constant
        return 1.0 / (1.0 + tau / te)

    def set_frequency(self, f):
        """
        Set the sampling frequency for the filter.

        :param f: The new frequency (should be > 0).
        :raises Exception: If frequency is not positive.
        """
        if f <= 0.0:
            raise Exception("freq should be >0")
        self.freq = f

    def set_min_cutoff(self, mc):
        """
        Set the minimum cutoff frequency for the filter.

        :param mc: The new minimum cutoff frequency (should be > 0).
        :raises Exception: If minCutoff is not positive.
        """
        if mc <= 0.0:
            raise Exception("minCutoff should be >0")
        self.min_cutoff = mc

    def set_beta(self, b):
        """
        Set the beta parameter for adjusting dynamic cutoff.

        :param b: The new beta value.
        """
        self.beta = b

    def set_derivative_cutoff(self, dc):
        """
        Set the cutoff frequency for the derivative filter.

        :param dc: The new derivative cutoff frequency (should be > 0).
        :raises Exception: If dCutoff is not positive.
        """
        if dc <= 0.0:
            raise Exception("dCutoff should be >0")
        self.d_cutoff = dc

    def filter(self, value, timestamp=-1):
        """
        Apply the One Euro Filter to a new value.

        :param value: The new value to filter.
        :param timestamp: The timestamp for the new value (default is -1).
        :return: The filtered result.
        """
        # Update the frequency based on the elapsed time
        if self.last_time != -1 and timestamp != -1 and (timestamp - self.last_time) > 0:
            self.freq = 1000.0 / (timestamp - self.last_time)

        self.last_time = timestamp  # Update the last timestamp
        d_value = (value - self.x.last_raw_value()) * self.freq if self.x.has_last_raw_value() else 0.0
        ed_value = self.dx.filter_with_alpha(d_value, self.alpha(self.d_cutoff))  # Filter the derivative
        cutoff = self.min_cutoff + self.beta * abs(ed_value)  # Calculate adaptive cutoff
        return self.x.filter_with_alpha(value, self.alpha(cutoff))  # Filter the current value

    def filter_values(self, values, timestamp=-1):
        """
        Filter a list of values using the One Euro Filter.

        :param values: List of values to be filtered.
        :param timestamp: The timestamp for the values (default is -1).
        :return: A list of filtered results.
        """
        if len(values) == 1:
            return self.filter(values[0], timestamp)  # Filter a single value
        return [self.filter(val, timestamp) for val in values]  # Filter multiple values

    def release(self):
        pass
