import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
from arch.bootstrap import CircularBlockBootstrap
from statsmodels.tsa.arima_model import ARMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from time import sleep
import time
import os
import warnings
warnings.simplefilter('ignore')


# def ts_graph(ts, ts_name):
#     """input: pd.Series or pd.DataFrame"""
#     ts.plot()
#     plt.title('{}'.format(ts_name))
#     plt.show()


def stationary_test(ts, ts_name):
    """stationary test for time series
        using AD Fuller Method and KPSS method"""
    adf = adfuller(ts, maxlag=int(len(ts)/4))
    result_kpss = kpss(ts)

    if adf[1] < 0.05:
        if result_kpss[1] > 0.05:
            stationary_test_outcome = 's'
            # print('\nstationary test for {}: stationary'.format(ts_name))
        else:
            stationary_test_outcome = 'ds'
            # print('\nstationary test for {}: difference stationary'.format(ts_name))
    else:
        if result_kpss[1] > 0.05:
            stationary_test_outcome = 'ts'
            # print('\nstationary test for {}: trend stationary'.format(ts_name))
        else:
            stationary_test_outcome = 'ns'
            # print('\nstationary test for {}: not stationary'.format(ts_name))
    return stationary_test_outcome


# def acf_pacf_analysis(ts, ts_name):
#     plot_acf(ts, alpha=0.05, lags=int(len(ts)/5))
#     plt.title('{}: acf starts from {}'.format(ts_name, int(len(ts)/5)))
#     plt.show()
#     plot_pacf(ts, alpha=0.05, lags=int(len(ts)/5))
#     plt.title('{}: pacf starts from {}'.format(ts_name, int(len(ts)/5)))
#     plt.show()


# def distribution_graph(ts, ts_name):
#     ax = plt.subplot(211)
#     kde = stats.gaussian_kde(ts)
#     x = np.linspace(np.min(ts), np.max(ts), 1000)
#     plt.plot(x, kde(x))
#     plt.grid(True)
#     plt.title("simulated distribution for {}".format(ts_name))
#
#     plt.subplot(212, sharex=ax)
#     box_plot_keys = plt.boxplot(ts, notch=True, vert=False)
#     plt.grid(True)
#     plt.show()
#     return box_plot_keys
#
#
# def distribution_analysis(ts, ts_name):
#     """Time series analysis based on frequency"""
#     print('\n{} description:'.format(ts_name))
#     print(ts.describe())
#     distribution_graph(ts, ts_name)


def get_parameters(p, parameters_resampling_times):
    """generate AR parameters with constraints:
        1. NO. of parameters equals p
        2. sum of parameters is smaller than 1
        3. single parameter value should be in the range [-0.5, 0.5]"""
    np.random.seed(parameters_resampling_times)
    a = np.array(np.random.rand(p)) - 0.5
    if np.abs(np.sum(a)) < 1:
        parameters = a
        # print(parameters)
    else:
        parameters = get_parameters(p, parameters_resampling_times*21)
    return parameters


def generate_ar(p, size, parameters_resampling_times):
    ar_params = get_parameters(p, parameters_resampling_times)
    ma_params = np.array([0.0]*p)
    ar = np.r_[1, -ar_params]
    # add zero-lag and negate
    ma = np.r_[1, ma_params]
    # add zero-lag
    dates = pd.date_range(start='2000-1-1', periods=size, freq='D')
    ar = pd.Series(arma_generate_sample(ar, ma, size), index=dates)
    str1 = 'generate AR({})'.format(p)
    # ts_graph(ar, str1)
    stationary_test(ar, str1)
    # acf_pacf_analysis(ar, str1)
    # distribution_analysis(ar, str1)
    return ar, ar_params


# Part1: Basic analysis of generated AR(2) time series
# Part2: AR(2) model fitting and check whether the AR(2) is suitable for Bootstrapped data
def ljung_box_test(ts):
    """ Ljung-Box chi-square statistic for testing for white noise residuals
        return: n-array-[test statistic, p-value]
        H0: residuals are white noise
        default maximum of lag is 40"""
    ljungbox_test = acorr_ljungbox(ts)
    ljungbox_p_value = ljungbox_test[1]
    # print('\nNo. unsuitable AR(2) model: ', np.sum(ljungbox_p_value <= 0.05))
    if np.sum(ljungbox_p_value <= 0.05) > 0:
        # print('AR(2) model is not suitable due to the existing auto-correlation of residuals')
        ljungbox_test_result = 0
    else:
        ljungbox_test_result = 1
        # print('outcome: there is no auto-correlation of residuals')
    return ljungbox_test_result


def model_residual_test(ts_residual):
    """check whether model residuals are white noise or not
    stationary, mean = 0, no auto-correlation
    AD Fuller & KPSS test; one-sample t test; Ljung box test"""
    t_test_statistic, t_test_p_values = stats.ttest_1samp(ts_residual, 0.0)
    # print('p-values: ', t_test_p_values)
    if t_test_p_values < 0.25:
        model_residual_test_result = 0
    else:
        stationary_test_result = stationary_test(ts_residual, 'model residuals')
        if stationary_test_result == 's':
            ljungbox_test_result = ljung_box_test(ts_residual)
            if ljungbox_test_result == 1:
                model_residual_test_result = 1
                # print('AR(2) model is suitable')
                # print('--------------------------\n')
            else:
                model_residual_test_result = 0
        else:
            model_residual_test_result = 0
    return model_residual_test_result


def ar_model_fit(ts, p):
    """divide the dataset into training and test set and fit AR(2) model
        return: model parameters; result of white noise test of model residuals"""
    model = ARMA(endog=ts, order=(p, 0))
    model_fit = model.fit(method='mle', ic='aic', trend='nc', transparams=True, disp=0)
    # print('Coefficients: {}'.format(model_fit.params))
    model_residual = model_fit.resid
    model_residual_test_results = model_residual_test(model_residual)
    return model_fit.params, model_residual_test_results


# Part3: Circular Block Bootstrap to do re-sampling
def confidence_interval(params, indicator, ar_parameters_original):
    """To build up the confidence level for parameters
        based on the outcome of distribution test: t, norm, or nothing
        params: pd.Series;    indicator: 't', 'norm'  """
    # print('\n params inputted for confidence interval: \n', params.head())
    sample_std = np.std(params) / np.sqrt(len(params))
    params_similarity_test = [0]*len(params)

    norm_ci1 = -1.96*sample_std + ar_parameters_original
    norm_ci2 = ar_parameters_original + 1.96*sample_std
    t_statistic = stats.t.ppf(1 - 0.025, len(params))
    t_ci1 = -t_statistic*sample_std + ar_parameters_original
    t_ci2 = ar_parameters_original + t_statistic*sample_std

    other_distribution_quantile1, other_distribution_quantile2 = np.quantile(params, [0.025, 0.975])
    other_distribution_mean = np.mean(params)
    other_distribution_statistic1 = (other_distribution_quantile1 - other_distribution_mean)/sample_std
    other_distribution_statistic2 = (other_distribution_quantile2 - other_distribution_mean)/sample_std
    other_distribution_confidence_interval1 = other_distribution_statistic1*sample_std + ar_parameters_original
    other_distribution_confidence_interval2 = other_distribution_statistic2*sample_std + ar_parameters_original

    # if indicator == 'norm':
    #     print("confidence interval: {}".format((norm_ci1, norm_ci2)))
    # elif indicator == 't':
    #     print("confidence interval: {}".format((t_ci1, t_ci2)))
    # else:
    #     print("confidence interval: {}".
    #           format((other_distribution_confidence_interval1, other_distribution_confidence_interval2)))

    if indicator == 'norm':
        for i in np.arange(len(params)):
            if norm_ci1 < params.iloc[i] < norm_ci2:
                params_similarity_test[i] = 1
    elif indicator == 't':
        for i in np.arange(len(params)):
            if t_ci1 < params.iloc[i] < t_ci2:
                params_similarity_test[i] = 1
    else:
        for i in np.arange(len(params)):
            if other_distribution_confidence_interval1 < params.iloc[i] < other_distribution_confidence_interval2:
                params_similarity_test[i] = 1
    return params_similarity_test


def t_distribution_test(params):
    """test for whether the simulated model parameters
        follow t distribution"""
    kstest_results = stats.kstest(params, 't', args=(len(params), ))
    if kstest_results[1] > 0.025:
        params_distribution_result = 't'
    else:
        params_distribution_result = 'neither t nor normal distribution'
    return params_distribution_result


def normality_distribution(params, ar_parameters_original):
    """test for whether the simulated model parameters
        follow normal distribution
        tests adapted: K-S test, shapiro–wilk test"""
    # print("----------------------------------------")
    # print("inputted parameters for normality test: \n", params.head())
    shapiro_results = stats.shapiro(params)
    # returns: (w statistics, p value);  w vs 1, p vs 0.05; smaller, more likely to reject
    # kstest_results = stats.kstest(params, 'norm')
    # return: (K-S test statistics, 2-tailed p value)
    # print("shapiro_results[1]: ", shapiro_results[1])
    if shapiro_results[1] > 0.05:
        params_distribution_result = 'norm'
    else:
        params_distribution_result = t_distribution_test(params)
    # print("\nparams_distribution_result: ", params_distribution_result)
    params_similarity_test = confidence_interval(params, params_distribution_result, ar_parameters_original)
    sleep(0.05)
    return params_similarity_test


def model_parameters_comparison(params_df, ar_parameters_original, model_residual_test_results_series):
    """1. for loop Parameter DataFrame to get the Parameters Similarity test
       2. also, consider the ones that fail to pass the Model Right"""
    not_pass_model_right = []
    for i in np.arange(len(model_residual_test_results_series)):
        if model_residual_test_results_series.iloc[i] == 0:
            not_pass_model_right.append(i+1)
    params_df_exclude = params_df.drop(not_pass_model_right, axis=0)

    parameters_similarity_test = pd.DataFrame()
    for j in np.arange(len(params_df_exclude.columns)):
        parameters_similarity_test[params_df.columns[j]] = normality_distribution(
            params_df_exclude[params_df_exclude.columns[j]], ar_parameters_original[j])
    # print("\nparameters_similarity_test: \n", parameters_similarity_test.head())
    sleep(0.05)
    return parameters_similarity_test


def circular_block_bootstrap(block_size, dataset, bootstrap_resampling_times, p, ar_parameters_original, parameters_resampling_times):
    """Circular Block Bootstrap is adapted
        re-sampling 100 times for given block size
        1. store the parameters in the DataFrame form
        2. store the Model Right result in the pd.Series form
        3. recall the Model Parameter Comparison test
        4. store the results of Model Parameter

    # parameters_similarity_test_rate: mean of all 1-or-0 matrix
    # parameters_accuracy_vs_model_right: take the Model Right Test into consideration
    # parameters_same_test_pass_rate: all parameters pass the Parameters Same Test
    # pass_2_tests_rate: Pass Model Right Test & Model Same Test"""

    # 1. re-sample time series
    bootstrap = CircularBlockBootstrap(block_size, dataset)
    re_sample = np.array([k[0][0] for k in bootstrap.bootstrap(bootstrap_resampling_times)])
    # re_sample = np.reshape(re_sample, -1)
    # print('first change for re_sample:\n', re_sample)
    len_simulation = len(re_sample[0])
    re_sample = np.reshape(re_sample, (bootstrap_resampling_times, len_simulation))
    sleep(0.05)

    # 2. store fitted parameters & Model Right results
    model_residual_test_results_series = pd.Series()
    model_parameters = pd.DataFrame()
    for l in np.arange(len(re_sample)):
        results = ar_model_fit(re_sample[l], p)
        model_residual_test_results_series.loc[l + 1] = results[1]
        model_parameters[l+1] = results[0]
    model_parameters = model_parameters.transpose()
    # print("\nbootstrapped model parameters | re-sampling times: \n", model_parameters.head())
    # print('\nar model residuals test result: \n', model_residual_test_results_series.head())
    sleep(0.05)

    # 3. recall the Parameters Comparison Test & store the results
    parameters_similarity_test = \
        model_parameters_comparison(model_parameters, ar_parameters_original, model_residual_test_results_series)
    parameters_similarity_test_individual_rate_list = np.mean(parameters_similarity_test, axis=1)
    parameters_similarity_test_rate = np.mean(parameters_similarity_test_individual_rate_list)

    model_right_test_pass_rate = np.mean(model_residual_test_results_series)
    # print('\nModel Right Test Pass Rate: ', model_right_test_pass_rate)
    sleep(0.05)

    counts = 0
    for i in np.arange(len(parameters_similarity_test)):
        if np.sum(parameters_similarity_test.iloc[i, :]) == len(parameters_similarity_test.columns):
            counts = counts + 1
    parameters_same_test_pass_rate = counts / len(parameters_similarity_test)
    parameters_accuracy_vs_model_right = \
        np.sum(parameters_similarity_test_individual_rate_list) / len(model_residual_test_results_series)
    pass_2_tests_rate = counts / len(model_residual_test_results_series)
    half_test1_half_test2 = 0.7*model_right_test_pass_rate + 0.3*parameters_same_test_pass_rate
    sleep(0.05)
    return pass_2_tests_rate, half_test1_half_test2, model_right_test_pass_rate, parameters_same_test_pass_rate, \
        parameters_similarity_test_rate, parameters_accuracy_vs_model_right, block_size, parameters_resampling_times, p


def calling_function(ar_p, ar_size, parameters_resampling_times, block_size, bootstrap_resampling_times):
    generated_ar_results = generate_ar(ar_p, ar_size, parameters_resampling_times)
    ar_generated = generated_ar_results[0]
    ar_parameters_original = generated_ar_results[1]
    # print('\nAR({}), parameters resampling:{} times,\noriginally given parameters:{}'.
    #       format(ar_p, parameters_resampling_times, ar_parameters_original))

    bootstrap_results = circular_block_bootstrap(block_size, ar_generated, bootstrap_resampling_times,
                                                 ar_p, ar_parameters_original, parameters_resampling_times)
    str1 = "AR({}), parameters resampling times:{}, block size:{}".format(ar_p, parameters_resampling_times, block_size)
    outcome_series = pd.Series(data=bootstrap_results, index=name_list, name=str1)
    print("\n outcomes: \n{}".format(outcome_series[:-3]))
    return outcome_series


start_time = time.time()

ar_size = 504
# block_size = np.arange(2, 10, 2)
block_size = np.concatenate([np.arange(2, 20, 2), np.arange(20, 65, 5)])
bootstrap_resampling_times = 500

df = pd.DataFrame()
name_list = ["pass_2_tests_rate", "half_test1_half_test2", "model_right_test_pass_rate",
             "parameters_same_test_pass_rate", "parameters_similarity_test_rate",
             "parameters_accuracy_vs_model_right", "block_size", "params_resmapling_times", "ar_p"]

pool = mp.Pool(processes=10)
results = [pool.apply_async(calling_function, args=(ar_p, ar_size, parameters_resampling_times, block_size_single,
                                                    bootstrap_resampling_times))
           for ar_p in np.arange(4, 6)
           for parameters_resampling_times in np.arange(1, 11)
           for block_size_single in block_size]

for p in results:
    output = p.get()
    df[p] = output

df = df.transpose()
df = df.set_index("block_size")
print("\n-------------------------------------------------------")
print("\ndf:\n", df)

for l in np.arange(len(df)/len(block_size)):
    optimal_block_size = df.iloc[int(len(block_size)*l):int(len(block_size)*(l+1)), 0][
        df.iloc[int(len(block_size)*l):int(len(block_size)*(l+1)), 0] ==
        np.max(df.iloc[int(len(block_size)*l):int(len(block_size)*(l+1)), 0])].index.tolist()

    df.iloc[int(len(block_size)*l): int(len(block_size)*(l+1))].plot(y=name_list[:2])
    plt.scatter(x=optimal_block_size, y=[0]*len(optimal_block_size), c="r", marker="o")
    plt.text(x=block_size[int(len(block_size)/2)], y=0,
             s="optimal block size is:\n{}".format(list(map(int, optimal_block_size))))
    plt.title("AR({}) params sampling:{}".format(int(df.iloc[int(l*len(block_size)), -1]),
                                                 int(df.iloc[int(l*len(block_size)), -2])))
    plt.xlabel("block size")
    plt.show()

    df.iloc[int(len(block_size)*l): int(len(block_size)*(l+1))].plot(y=name_list[2:-3])
    plt.xlabel("block size")
    plt.title("AR({}) params sampling:{}".format(int(df.iloc[int(l*len(block_size)), -1]),
                                                 int(df.iloc[int(l*len(block_size)), -2])))
    plt.show()

end_time = time.time()
print("running time: ", end_time - start_time)
