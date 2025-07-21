# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import numpy as np

from .Filter import Filter


class HeuristicFilter(Filter):
    """
    A heuristic filter that smooths two input values based on a look-ahead strategy.

    This filter maintains a history of raw values and applies a smoothing heuristic
    based on comparisons with neighboring values.
    """

    def __init__(self, look_ahead=3):
        """
        Initialize the HeuristicFilter with a specified look-ahead value.

        :param look_ahead: The number of values to look ahead for smoothing.
        """
        super().__init__()
        self.raw_x = []  # List to store raw x values
        self.raw_y = []  # List to store raw y values
        self.dummy_x = np.nan  # Smoothed x value
        self.dummy_y = np.nan  # Smoothed y value
        self.look_ahead = look_ahead  # Number of values to look ahead

    def filter_values(self, values, timestamp=-1):
        """
        Filter a pair of values using the heuristic method.

        :param values: A list containing two values (x and y).
        :param timestamp: Optional timestamp (default is -1, not used in filtering).
        :raises ValueError: If the input values list does not contain exactly two values.
        :return: The filtered x and y values.
        """
        if len(values) != 2:
            raise ValueError("Heuristic filter requires two values")

        # Apply filtering to x and y values separately
        self.raw_x = self.do_filter(True, values[0])
        self.raw_y = self.do_filter(False, values[1])

        # If we have enough history, return the smoothed values
        if len(self.raw_x) == self.look_ahead * 2:
            if self.dummy_x is np.nan and self.dummy_y is np.nan:
                # If no smoothed values available, return original values
                return values
            else:
                return [self.dummy_x, self.dummy_y]  # Return the smoothed values
        else:
            return values  # Return original values if not enough history

    def do_filter(self, is_x, element):
        """
        Apply the filtering logic to the raw value.

        :param is_x: Boolean indicating if the value is for x (True) or y (False).
        :param element: The new value to filter.
        :return: The updated list of raw values.
        """
        raw = self.raw_x if is_x else self.raw_y  # Select the appropriate raw value list
        raw.append(element)  # Append the new element to the raw values

        # If we have enough raw values to apply the look-ahead smoothing
        if len(raw) == self.look_ahead * 2 + 1:
            # Apply heuristic smoothing
            for next_val in range(1, self.look_ahead + 1):
                condition_one = (
                        raw[self.look_ahead - next_val] < raw[self.look_ahead] and
                        raw[self.look_ahead] > raw[self.look_ahead + next_val]
                )
                condition_two = (
                        raw[self.look_ahead - next_val] > raw[self.look_ahead] and
                        raw[self.look_ahead] < raw[self.look_ahead + next_val]
                )
                if condition_one or condition_two:
                    # Calculate distances to the neighboring values
                    prev_dist = abs(raw[self.look_ahead - next_val] - raw[self.look_ahead])
                    next_dist = abs(raw[self.look_ahead + next_val] - raw[self.look_ahead])

                    # Choose the closer neighbor
                    if prev_dist < next_dist:
                        raw[self.look_ahead] = raw[self.look_ahead - next_val]
                    else:
                        raw[self.look_ahead] = raw[self.look_ahead + next_val]

            # Update the dummy smoothed values
            if is_x:
                self.dummy_x = raw[self.look_ahead]
            else:
                self.dummy_y = raw[self.look_ahead]
            raw.pop(0)  # Remove the oldest value to maintain the history length

        return raw  # Return the updated raw values

    def release(self):
        pass
