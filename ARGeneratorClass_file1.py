import pandas as pd
import numpy as np
from statsmodels.tsa.arima_process import arma_generate_sample
import warnings
warnings.simplefilter('ignore')


class ARGeneratorClass:
    """class designed to generate AR(p) model

    Input Parameters:
        parameters_sampling_times: number
        ar_p: number
    Return:
        ar_parameters: np.array
        ar_time_series: pd.Series, index is a time period"""

    def __init__(self, ar_p, time_series_size, parameters_sampling_times, start_date, freq):
        self.ar_p = ar_p
        self.time_series_size = time_series_size
        self.parameters_sampling_times = parameters_sampling_times
        self.ar_parameters = []
        self.ma_parameters = []
        self.date_period = pd.date_range(start=start_date,
                                         periods=self.time_series_size, freq=freq)
        self.ar_time_series = []

    def parameters_generator_function(self):
        """generate parameters with 3 constraints:
            1. No. of parameters is ar_p;
            2. sum of parameters is smaller than 1
            3. each parameter is in [-0.5, 0.5]

        For 2.: the explanatory power(sum of parameters) should not exceed 100%
        For 3.: limit the explanatory power of single variable"""
        np.random.seed(self.parameters_sampling_times)
        self.ar_parameters.append(np.random.rand(self.ar_p) - 0.5)
        if np.sum(self.ar_parameters) > 1:
            self.ar_parameters = []
            self.parameters_generator_function()

    def ar_p_time_series_generator_function(self):
        # add zero-lag and negate
        # np.array(self.ar_parameters)) is [[]], thus [0] is used here
        self.ar_parameters = np.r_[1, -np.array(self.ar_parameters)[0]]
        self.ma_parameters = np.r_[1, np.array([0.0] * self.ar_p)]
        self.ar_time_series = pd.Series(
            arma_generate_sample(self.ar_parameters, self.ma_parameters,
                                 self.time_series_size), index=self.date_period)

    def get_ar_parameters_and_time_series_function(self):
        self.parameters_generator_function()
        self.ar_p_time_series_generator_function()
        self.ar_parameters = - self.ar_parameters
        return self.ar_parameters[1:], self.ar_time_series



# testing code
# ar_p = 2
# time_series_size = 200
# parameters_sampling_times = 2
# start_date = "2000-1-1"
# freq = "D"
# test_a = ARGeneratorClass(ar_p=ar_p, time_series_size=time_series_size,
#                           parameters_sampling_times=parameters_sampling_times,
#                           start_date=start_date, freq=freq)
# ar_parameters, ar_time_series = test_a.get_ar_parameters_and_time_series_function()

# potential problems/ improvement:
# 1.  0.5 is a number, could change it into an input or an option?
# [solved] 2.  time period, frequency: D
