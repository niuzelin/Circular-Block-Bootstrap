import pandas as pd
import numpy as np
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.stattools import adfuller, kpss
from scipy import stats
from arch.bootstrap import CircularBlockBootstrap
from statsmodels.tsa.arima_model import ARMA
from statsmodels.stats.diagnostic import acorr_ljungbox
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


class ARModelFitClass:
    """fit the AR(p) model, given ar_p

    Input Parameters:
        ar_time_series: pd.Series, index is a time period
        ar_p: number
    Return:
        ar_model_fit_params: pd.Series, index is ar.Lx.y
        ar_model_residual: pd.Series, index is a time period"""

    def __init__(self, ar_time_series, ar_p):
        self.ar_time_series = ar_time_series
        self.ar_p = ar_p
        self.ar_model = []
        self.ar_model_fit = []
        self.ar_model_residual = []
        self.ar_model_fit_parameters = []

    def ar_model_fit_function(self):
        # method: maximum likelihood estimation
        # model judgment: AIC;  non constant
        # transform parameters to make sure stationary
        # no convergence information is displayed
        self.ar_model = ARMA(endog=self.ar_time_series, order=(self.ar_p, 0))
        # print(self.ar_model)
        self.ar_model_fit = self.ar_model.fit(method='mle', ic='aic', trend='nc',
                                              transparams=True, disp=0)
        # print(self.ar_model_fit)
        self.ar_model_fit_parameters = self.ar_model_fit.params
        self.ar_model_residual = self.ar_model_fit.resid

    def get_model_fit_parameters_function(self):
        return self.ar_model_fit_parameters

    def get_model_residual_function(self):
        return self.ar_model_residual


# testing code
# ar_time_series is generated by ARGeneratorClass
# test_b = ARModelFitClass(ar_time_series=ar_time_series, ar_p=ar_p)
# test_b.ar_model_fit_function()
# ar_model_fit_parameters = test_b.get_model_fit_parameters_function()
# ar_model_residual = test_b.get_model_residual_function()

# potential problems / improvements:
# 1. too many instance empty variables are created before
# 2. ar_model.fit(transform=True).
# When No. parameters of AR increase to more than 10,
# some problems are increased

# 3. ar_model.fit(trend="nc").
# Make it as an option for user to choose,
# which suggests the corresponding change in ARGeneratorClass
# to make sure that user can generate AR(p) model with constant


class ModelRightTestClass:
    """test whether the residual of fitted model is white noise or not
        1. mean of residual vs 0,  1-sample t test
        2. stationary or not,  AD Fuller test & KPSS test
        3. auto-correlation among time series or not,  ljung box test

    Input Parameters:
        model_residual: pd.Series, index is a time period
    Return:
        model_right_test_result: number, 0(wrong) 1(right)"""

    def __init__(self, model_residual):
        self.model_residual = model_residual
        self.model_right_test_result = 0
        self.mean_equals_zero_test_result = 0
        self.stationary_test_result = " "
        self.auto_correlation_test_result = 1
        self.adfuller_test_result = []
        self.kpss_test_result = []

    def mean_equals_zero_function(self):
        if stats.ttest_1samp(self.model_residual, 0.0)[1] > 0.25:
            self.mean_equals_zero_test_result = 1
        return self.mean_equals_zero_test_result

    def stationary_test_function(self):
        # to ensure the application
        self.adfuller_test_result = adfuller(self.model_residual,
                                             maxlag=int(len(self.model_residual) / 4))
        self.kpss_test_result = kpss(self.model_residual)
        # print(self.adfuller_test_result[1])
        if self.adfuller_test_result[1] < 0.05:
            if self.kpss_test_result[1] > 0.05:
                self.stationary_test_result = "stationary"
            else:
                self.stationary_test_result = "difference stationary"
        else:
            if self.kpss_test_result[1] > 0.05:
                self.stationary_test_result = "trend stationary"
            else:
                self.stationary_test_result = "not stationary"
        return self.stationary_test_result

    def auto_correlation_test_function(self):
        """default maximum lag is min((nobs // 2 - 2), 40) """
        if np.sum(acorr_ljungbox(self.model_residual)[1] <= 0.05) > 0:
            self.auto_correlation_test_result = 0
        return self.auto_correlation_test_result

    def get_model_right_test_result_function(self):
        if self.mean_equals_zero_test_result == 1 and \
                self.stationary_test_result == "stationary" and \
                self.auto_correlation_test_result == 1:
            self.model_right_test_result = 1
        return self.model_right_test_result


# testing code
# model_residual is calculated in ARModelFitClass
# test_c = ModelRightTestClass(model_residual=ar_model_residual)
# print(test_c.mean_equals_zero_function())
# print(test_c.stationary_test_function())
# print(test_c.auto_correlation_test_function())
# model_right_test_result = test_c.get_model_right_test_result_function()

# potential problems / improvements:
# 1. kpss_test_result[1] suggests the unsafe use of bracket
#


class RecallModelFitAndModelRightClass:
    """main purpose: fit the bootstrapped time series and conduct the Model Right Test
            by recalling the classes built up before

    Input Parameters:
        bootstrapped_time_series_arrays: np.array
                in bootstrap_sampling_times dimensions;
                each bootstrapped time series takes up an array
        ar_p: number

        for AR Model Fit Class
            input parameters:
                ar_time_series: pd.Series, index is a time period
                ar_p: number
            return:
                ar_model_fit_params: pd.Series, index is ar.Lx.y
                ar_model_residual: pd.Series, index is a time period

        for Model Right Test Class
            input parameters:
                ar_model_residual: pd.Series, index is a time period
            return:
                model_right_test_result: number, 0(wrong) 1(right)

    Return:
        bootstrapped_ar_model_fit_parameters_df: pd.DataFrame
        model_right_test_results_series: pd.Series, consist of 0(wrong) 1(right)"""

    def __init__(self, bootstrapped_time_series_arrays, ar_p):
        self.bootstrapped_time_series_arrays = bootstrapped_time_series_arrays
        self.bootstrapped_time_series_arrays_len = len(bootstrapped_time_series_arrays)
        self.bootstrapped_time_series_arrays_index = 0
        self.ar_p =ar_p
        self.bootstrapped_ar_model_fit_results = []
        self.bootstrapped_ar_model_fit_parameters_df = pd.DataFrame()
        self.model_right_test = []
        # self.model_right_test_fail_list = []
        self.model_right_test_results_series = pd.Series()

    def recall_model_fit_and_model_right_class_function(self):
        from find_block_size_files import ARModelFitClass_file3
        from find_block_size_files import ModelRightTestClass_file4

        for self.bootstrapped_time_series_arrays_index in np.arange(self.bootstrapped_time_series_arrays_len):

            self.bootstrapped_ar_model_fit_results = ARModelFitClass_file3.ARModelFitClass(
                ar_time_series=self.bootstrapped_time_series_arrays[self.bootstrapped_time_series_arrays_index], ar_p=self.ar_p)
            self.bootstrapped_ar_model_fit_results.ar_model_fit_function()
            # + 1 refers to start from time 1 not time 0
            self.bootstrapped_ar_model_fit_parameters_df[self.bootstrapped_time_series_arrays_index + 1] \
                = self.bootstrapped_ar_model_fit_results.get_model_fit_parameters_function()

            self.model_right_test = ModelRightTestClass_file4.ModelRightTestClass(
                model_residual=self.bootstrapped_ar_model_fit_results.get_model_residual_function())
            self.model_right_test.mean_equals_zero_function()
            self.model_right_test.stationary_test_function()
            self.model_right_test.auto_correlation_test_function()
            self.model_right_test_results_series.loc[self.bootstrapped_time_series_arrays_index + 1] = \
                self.model_right_test.get_model_right_test_result_function()

    def get_bootstrapped_ar_model_fit_parameters_df_function(self):
        self.recall_model_fit_and_model_right_class_function()
        self.bootstrapped_ar_model_fit_parameters_df = self.bootstrapped_ar_model_fit_parameters_df.transpose()
        return self.bootstrapped_ar_model_fit_parameters_df

    def get_model_right_test_results_series_function(self):
        return self.model_right_test_results_series


# testing code
# test_e = RecallModelFitAndModelRightClass(
#     bootstrapped_time_series_arrays=bootstrapped_time_series_arrays, ar_p=ar_p)
# bootstrapped_ar_model_fit_parameters_df = test_e.get_bootstrapped_ar_model_fit_parameters_df_function()
# model_right_test_results_series = test_e.get_model_right_test_results_series_function()


class DistributionTestClass:
    """test whether the given parameters series follow t, norm or other distribution

    Input Parameters:
        input_series: pd.Series
    Return:
        distribution_test_result: str; possible elements: "t", "norm", "other" """

    def __init__(self, input_series):
        self.input_series = input_series
        self.shapiro_results = []
        self.ks_test_results = []
        self.distribution_test_result = "other"

    def normality_test_function(self):
        if len(self.input_series) <= 3:
            print("\ncannot conduct Shapiro-Wilk test because len of given input_series is less than 3")
        else:
            self.shapiro_results = stats.shapiro(self.input_series)
            if self.shapiro_results[1] > 0.05:
                self.distribution_test_result = "norm"

    def t_distribution_test_function(self):
        self.ks_test_results = stats.kstest(self.input_series, "t", args=(len(self.input_series), ))
        if self.ks_test_results[1] > 0.025:
            self.distribution_test_result = 't'

    def get_distribution_test_result_function(self):
        self.normality_test_function()
        self.t_distribution_test_function()
        return self.distribution_test_result


# testing code
# test_f = DistributionTestClass(input_series=bootstrapped_ar_model_fit_parameters_df[0])
# distribution_test_result = test_f.get_distribution_test_result_function()


class CalculateConfidenceIntervalClass:
    """calculate confidence interval given distribution type and data

    Default setting: 0.05 significance level, double-tailed

    Input Parameters:
        null_value: number, original value
        input_series: pd.Series
        distribution_test_result: str; possible elements: "t", "norm", "other"
    Return:
        interval_left: number
        interval_right: number
        """

    def __init__(self, null_value, input_series, distribution_test_result):
        self.null_value = null_value
        # from find_block_size_files import updated_DistributionTestClass_file6
        self.input_series = input_series
        self.distribution_test_result = distribution_test_result
        self.interval_left = 0
        self.interval_right = 0
        self.input_series_std = np.std(input_series)

    def calculate_confidence_interval_function(self, statistic):
        self.interval_left = -statistic * self.input_series_std + self.null_value
        self.interval_right = statistic * self.input_series_std + self.null_value

    def calculate_statistic_function(self):
        if self.distribution_test_result == "norm":
            statistic = 1.96
            self.calculate_confidence_interval_function(statistic)
        elif self.distribution_test_result == "t":
            statistic = stats.t.ppf(1 - 0.025, len(self.input_series))
            self.calculate_confidence_interval_function(statistic)
        else:
            quantile_left, quantile_right = np.quantile(self.input_series, [0.025, 0.975])
            self.interval_left = quantile_left - np.mean(self.input_series) + self.null_value
            self.interval_right = quantile_right - np.mean(self.input_series) + self.null_value

    def get_confidence_interval_function(self):
        self.calculate_statistic_function()
        return self.interval_left, self.interval_right


class ConfidenceIntervalTestClass:
    """test whether the value is in the confidence interval

    Input Parameters:
        interval_left: number
        interval_right: number
        input_series: pd.Series
    Returns:
        parameter_same_test_series: pd.Series, consist of boolean number 0/1"""

    def __init__(self, interval_left, interval_right, input_series):
        self.interval_left = interval_left
        self.interval_right = interval_right
        self.input_series = input_series
        self.parameter_same_test_series = pd.Series(data=[0]*len(input_series))

    def parameter_same_test_function(self):
        for i in np.arange(len(self.input_series)):
            if self.interval_left < self.input_series.iloc[i] < self.interval_right:
                self.parameter_same_test_series.iloc[i] = 1

    def get_parameter_same_test_series_function(self):
        self.parameter_same_test_function()
        return self.parameter_same_test_series

# test the function
# df_after_drop_rows = df_drop_rows_given_zero_function(df=bootstrapped_ar_model_fit_parameters_df,
                                                      # indicator_series=model_right_test_results_series)


class ModelSameTestClass:
    """recall all the steps of Model Right test

    Input Parameters:
        df: pd.DataFrame; columns: number of parameter
        indicator_series: pd.Series, consist of 0(wrong) 1(right) or just the nonzero index
        null_values: pd.Series, require len(null_series) = len(df.columns)
    Return:
        """
    def __init__(self, df, indicator_series, null_values):
        self.df = df
        self.indicator_series = indicator_series
        self.null_values = null_values
        self.parameters_same_test_result_df = pd.DataFrame()

    def df_drop_rows_with_constraint_function(self):
        """df: pd.DataFrame
        indicator_series: pd.Series, consist of 0(wrong) 1(right) or just the nonzero index"""
        if len(self.df) != len(self.indicator_series):
            print("inputted df and indicator_series have different length\n")
        self.df = self.df.iloc[self.indicator_series.to_numpy().nonzero()[0], :]

    def recall_parameters_same_steps_function(self):
        self.df_drop_rows_with_constraint_function()
        for i in np.arange(len(self.df.columns)):
            test_a = DistributionTestClass(input_series=self.df[i])
            distribution_test_result = test_a.get_distribution_test_result_function()
            test_b = CalculateConfidenceIntervalClass(null_value=self.null_values[i], input_series=self.df[i],
                                                      distribution_test_result=distribution_test_result)
            interval_left, interval_right = test_b.get_confidence_interval_function()
            test_c = ConfidenceIntervalTestClass(interval_left=interval_left, interval_right=interval_right,
                                                 input_series=self.df[i])
            self.parameters_same_test_result_df[i] = test_c.get_parameter_same_test_series_function()

    def get_parameters_same_test_result_df_function(self):
        self.recall_parameters_same_steps_function()
        return self.parameters_same_test_result_df


# testing code
# test_f = ModelSameTestClass(df=bootstrapped_ar_model_fit_parameters_df,
#                              indicator_series=model_right_test_results_series,
#                             null_values=ar_parameters)
# parameters_same_test_result_df = test_f.get_parameters_same_test_result_df_function()


def judgement_rates_function(model_right_test_results_series, parameters_same_test_result_df):
    """calculate benchmark judgement rates

    Input Parameters:
        model_right_test_results_series: pd.Series, consist of 0 and 1
        parameters_same_test_result_df: pd.DataFrame, consist of 0 and 1

    Return:
        model_right_test_pass_rate, average_parameters_same_rate,
        model_same_pass_rate, totally_pass_rate
    """
    model_right_test_pass_rate = np.mean(model_right_test_results_series)
    average_parameters_same_rate = np.mean(parameters_same_test_result_df.values)
    model_same_pass_counts = parameters_same_test_result_df[0][
        parameters_same_test_result_df.sum(axis=1)
        == len(parameters_same_test_result_df.columns)].count()
    model_same_pass_rate = model_same_pass_counts / len(parameters_same_test_result_df)

    # because parameters that fail to pass Model Right Test have been dropped
    # thus, totally_pass_rate needs to add up them
    totally_pass_rate = model_same_pass_counts / len(model_right_test_results_series)
    return model_right_test_pass_rate, average_parameters_same_rate, model_same_pass_rate, totally_pass_rate


# Testing code
# model_right_test_pass_rate, average_parameters_same_rate, model_same_pass_rate,\
# totally_pass_rate = judgement_rates_function(model_right_test_results_series,
#                                              parameters_same_test_result_df)

# potential improvement
# calculate the


def calling_multiprocessing_function(ar_p, time_series_size,
                                     parameters_sampling_times_single,
                                     start_date, freq, block_size,
                                     bootstrapped_sampling_times):
    """calling all the former steps"""

    from find_block_size_files import ARGeneratorClass_file1
    ar_generator = ARGeneratorClass_file1.ARGeneratorClass(
        ar_p=ar_p, time_series_size=time_series_size,
        parameters_sampling_times=parameters_sampling_times_single,
        start_date=start_date, freq=freq)
    ar_parameters, ar_time_series = \
        ar_generator.get_ar_parameters_and_time_series_function()

    from find_block_size_files import CircularBlockBootstrapClass_file2
    cbb_bootstrap = CircularBlockBootstrapClass_file2.CircularBlockBootstrapClass(
        time_series=ar_time_series, block_size=block_size,
        bootstrap_sampling_times=bootstrapped_sampling_times)
    bootstrapped_time_series_arrays = \
        cbb_bootstrap.get_bootstrapped_time_series_arrays_function()

    from find_block_size_files import RecallModelFitAndModelRightClass_file5
    model_right_test = \
        RecallModelFitAndModelRightClass_file5.RecallModelFitAndModelRightClass(
            bootstrapped_time_series_arrays=bootstrapped_time_series_arrays, ar_p=ar_p)
    bootstrapped_ar_model_fit_parameters_df = \
        model_right_test.get_bootstrapped_ar_model_fit_parameters_df_function()
    model_right_test_results_series = \
        model_right_test.get_model_right_test_results_series_function()

    from find_block_size_files import ModelSameTestClass_file6
    model_same = ModelSameTestClass_file6.ModelSameTestClass(
        df=bootstrapped_ar_model_fit_parameters_df,
        indicator_series=model_right_test_results_series, null_values=ar_parameters)
    parameters_same_test_result_df = \
        model_same.get_parameters_same_test_result_df_function()

    name = "AR({}), parameters resampling times:{}, block size:{}".format(
        ar_p, parameters_sampling_times_single, block_size)
    judgement_rates_series = ModelSameTestClass_file6.judgement_rates_function(
        model_right_test_results_series, parameters_same_test_result_df)
    judgement_rates_series = pd.Series(data=judgement_rates_series,
                                       index=name_list, name=name)
    return judgement_rates_series


# testing code
# judgement_rates_series = calling_multiprocessing_function(ar_p, time_series_size,
#                                                           parameters_sampling_times,
#                                                           start_date, freq, block_size,
#                                                           bootstrapped_sampling_times)


