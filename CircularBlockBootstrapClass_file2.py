import numpy as np
from arch.bootstrap import CircularBlockBootstrap
import warnings
warnings.simplefilter('ignore')


class CircularBlockBootstrapClass:
    """the class is designed to adapt Circular Block Bootstrap method

    Input Parameters:
        time_series: pd.Series, index is a time period
        block_size: number
        bootstrap_sampling_times: number
    Returns:
        bootstrapped_time_series_arrays: np.array
                            in bootstrap_sampling_times dimensions;
                            each bootstrapped time series takes up an array"""

    def __init__(self, time_series, block_size, bootstrap_sampling_times):
        self.time_series = time_series
        self.block_size = block_size
        self.bootstrap_sampling_times = bootstrap_sampling_times
        self.bootstrapped_time_series_arrays = []

    def circular_block_bootstrap_function(self):
        bootstrap = CircularBlockBootstrap(self.block_size, self.time_series)
        self.bootstrapped_time_series_arrays = \
            np.array([data[0][0] for data in bootstrap.bootstrap(self.bootstrap_sampling_times)])
        # reshape the result in the array form
        self.bootstrapped_time_series_arrays = \
            np.reshape(self.bootstrapped_time_series_arrays,
                       (self.bootstrap_sampling_times,
                        len(self.bootstrapped_time_series_arrays[0])))

    def get_bootstrapped_time_series_arrays_function(self):
        self.circular_block_bootstrap_function()
        return self.bootstrapped_time_series_arrays


# testing code
# block_size = 5
# bootstrapped_sampling_times = 200
# test_d = CircularBlockBootstrapClass(time_series=ar_time_series, block_size=block_size,
#                                      bootstrap_sampling_times=bootstrapped_sampling_times)
# bootstrapped_time_series_arrays = test_d.get_bootstrapped_time_series_arrays_function()


# Potential Problems / improvements:
# 1. frequently recalling the self.bootstrapped_time_series_arrays,
#        which may create confusion for understanding


