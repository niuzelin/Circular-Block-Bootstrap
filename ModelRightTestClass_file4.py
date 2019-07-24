import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.simplefilter('ignore')


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

