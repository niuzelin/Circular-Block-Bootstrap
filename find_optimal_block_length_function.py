import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import warnings
warnings.simplefilter('ignore')


def calling_multiprocessing_function(ar_p, time_series_size,
                                     parameters_sampling_times_single,
                                     start_date, freq, block_size,
                                     bootstrapped_sampling_times, name_list):
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
# ar_p = 2
# time_series_size = 200
# parameters_sampling_times = 2
# start_date = "2000-1-1"
# freq = "D"
# block_size = 5
# bootstrapped_sampling_times = 200
# judgement_rates_series = calling_multiprocessing_function(ar_p, time_series_size,
#                                                           parameters_sampling_times,
#                                                           start_date, freq, block_size,
#                                                           bootstrapped_sampling_times)


def find_optimal_block_length_function(ar_p_array, ar_size,
    parameters_sampling_times_array, start_date, freq, block_size_array,
    bootstrap_sampling_times, idle_core):
    """conduct the function to find the optimal block length
        call the multiprocessing processing

    Input Parameters:
    ar_p_array: array; array of AR(p)
    ar_size: number; length of time series
    parameters_sampling_times_array: array; sampling times for AR parameters
    start_date: "2018-01-01"; time period index for generated time series
    freq: 'D','M','Y';set the frequency of time period index
    block_size_array: np.array/list/pd.Series; list of block size
    bootstrap_sampling_times: number; times of bootstrap sampling times
    idle_core: number of idle cores as processing multiprocessing

    Print:
    graphs
    running time in second

    Return:
        rates_df: DataFrame; column: judgement rates;
            index: np.tile(block_size_array, parameter_sampling_times*len(ar_p_array))
    """

    start_time = time.time()

    rates_df = pd.DataFrame()
    name_list = ["model_right_test_pass_rate", "average_parameters_same_rate",
                 "model_same_pass_rate", "totally_pass_rate"]

    pool = mp.Pool(processes=mp.cpu_count() - idle_core)
    results = [pool.apply_async(calling_multiprocessing_function,
                                args=(
                                ar_p, ar_size, parameters_sampling_times_single,
                                start_date, freq, block_size_single,
                                bootstrap_sampling_times, name_list))
               for ar_p in ar_p_array
               for parameters_sampling_times_single in parameters_sampling_times_array
               for block_size_single in block_size_array]

    for p in results:
        output = p.get()
        rates_df[p] = output

    rates_df = rates_df.transpose()
    block_size_index = np.tile(block_size_array, len(parameters_sampling_times_array) * len(
        ar_p_array))
    rates_df = rates_df.set_index(block_size_index, "block_size")

    # the totally_pass_rate is too low to compare
    # manually create a judgement rate
    standardized_model_right_test_rate = (rates_df[name_list[0]] - np.mean(
        rates_df[name_list[0]])) / np.std(rates_df[name_list[0]])
    standardized_average_parameters_same_rate = (rates_df[
                                                     name_list[1]] - np.mean(
        rates_df[name_list[1]])) / np.std(rates_df[name_list[1]])
    name_list.append("half_model_right_half_model_same_rate")
    rates_df[name_list[-1]] = 0.5 * standardized_model_right_test_rate + 0.5 * standardized_average_parameters_same_rate

    # plot
    # rates_df
    # columns: rates; index: block_size_array
    # rows: ar_p_array*parameter_sampling_times*block_size_array
    # because we wanna find the relation
    # between optimal block size and parameter, even ar_p
    # thus, for each given time series(ar_p=xx, parameters_sampling_time=xx),
    # plot a graph of rates

    optimal_block_size_list = []
    for l_b in np.arange(int(len(rates_df) / len(block_size_array))):
        # l_b = range(ar_p_array*parameter_sampling_times)
        interval_start = len(block_size_array) * l_b
        interval_end = len(block_size_array) * (l_b + 1)
        optimal_block_size = rates_df.iloc[interval_start:interval_end, -1][
            rates_df.iloc[interval_start:interval_end, -1]
            == np.max(
                rates_df.iloc[interval_start:interval_end, -1])].index.tolist()
        # due to potential several optimal block sizes, choosing first one
        optimal_block_size_list.append(optimal_block_size[0])

        rates_df.iloc[interval_start:interval_end, :].plot(subplots=True)
        plt.legend(loc="best")
        plt.title("AR({}), params sampling:{}".format(
            ar_p_array[int(l_b / len(parameters_sampling_times_array))],
            parameters_sampling_times_array[l_b]))
        plt.xlabel("block size")
        plt.show()

        rates_df.iloc[interval_start:interval_end, -1].plot()
        plt.scatter(x=optimal_block_size, y=np.max(
            rates_df.iloc[interval_start:interval_end, -1]), c="r", marker="o")
        plt.text(x=block_size_array[int(len(block_size_array) / 2)],
                 y=np.max(rates_df.iloc[interval_start:interval_end, -1]) / 2,
                 s="optimal block size is:\n{}".format(
                     list(map(int, optimal_block_size))))
        plt.title("AR({}), params sampling:{}".format(
            ar_p_array[int(l_b / len(parameters_sampling_times_array))],
            parameters_sampling_times_array[l_b]))
        plt.xlabel("block size")
        plt.show()

    end_time = time.time()
    print("running time: ", end_time - start_time)
    return rates_df


ar_p_array = np.arange(2, 6)
ar_size = 300
parameters_sampling_times_array = np.arange(1, 11)
block_size_array = np.concatenate([np.arange(2, 20, 2), np.arange(20, 65, 5)])
# block_size_array = np.arange(2, 6, 2)
start_date = "2000-1-1"
freq = "D"
bootstrap_sampling_times = 500
idle_core = 3
df = find_optimal_block_length_function(ar_p_array, ar_size,
                                        parameters_sampling_times_array,
                                        start_date, freq, block_size_array,
                                        bootstrap_sampling_times, idle_core)
