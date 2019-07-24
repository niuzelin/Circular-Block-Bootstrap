import pandas as pd
import numpy as np
import warnings
warnings.simplefilter('ignore')


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