# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

class Filter:
    """
    Abstract base class for implementing various filter types.

    This class defines a template for filter objects that will process
    input values based on specific filtering criteria. Subclasses must
    implement the `filter_values` method.
    """

    def __init__(self):
        """
        Initialize the Filter object.

        This constructor can be extended by subclasses to initialize
        any necessary attributes.
        """
        pass

    def filter_values(self, values, timestamp=-1):
        """
        Filter the input values based on specific criteria.

        :param values: List or array of values to be filtered.
        :param timestamp: Optional timestamp for filtering context (default is -1).
                          Subclasses can use this parameter as needed for filtering logic.
        :raises NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def release(self):
        raise NotImplementedError("Subclasses must implement this method.")
