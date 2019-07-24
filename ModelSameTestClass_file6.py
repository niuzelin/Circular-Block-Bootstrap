import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.simplefilter('ignore')


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
