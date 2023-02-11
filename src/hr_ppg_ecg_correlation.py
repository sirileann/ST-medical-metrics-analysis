import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import *
import numpy as np
import matplotlib.mlab as mlab
from scipy.stats import pearsonr
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from numpy.linalg import norm


def cross_correlation_eval(x, y):
    r"""
    Inspired by matplotlib.pyplot.xcorr()
    Calculate the cross correlation between *x* and *y*.
    The input vectors are detrended and normalized to unit length.
    The correlation with lag k is defined as
    :math:`\sum_n x[n+k] \cdot y^*[n]`, where :math:`y^*` is the complex
    conjugate of :math:`y`.
    Parameters
    ----------
    x, y : array-like of length n
    Returns
    -------
    max_corr : scalar
        The maximum cross-correlation value.
    phase_lag : scalar
        The lag corresponding to the maximal cross-correlation value.
    Notes
    -----
    The cross correlation is performed with `numpy.correlate` with
    ``mode = "full"``.
    """

    detrend = mlab.detrend_none
    Nx = len(x)
    # if Nx != len(y):
    #     raise ValueError('x and y must be equal length')

    x = detrend(np.asarray(x))
    y = detrend(np.asarray(y))
    correls = np.correlate(x, y, mode="full")

    normed = True
    if normed:
        correls /= np.sqrt(np.dot(x, x) * np.dot(y, y))

    maxlags = Nx - 1
    if maxlags >= Nx or maxlags < 1:
        raise ValueError('maxlags must be None or strictly '
                         'positive < %d' % Nx)

    lags = np.arange(-maxlags, maxlags + 1)
    correls = correls[Nx - 1 - maxlags:Nx + maxlags]
    max_corr = correls.max()
    phase_lag = lags[correls.argmax()]
    return max_corr, phase_lag


def fit_hr(time_vector, heart_rate, heart_rate_quality, fit_fct=PchipInterpolator):
    """
    Takes a heart rate time-series and a boolean quality vector
    that states whether the quality of a datapoint is good enough
    to be used. Calculates the full heart-rate time-series using a PchipInterpolator (default).
    Parameters
    ----------
    heart_rate :
        array of length n, HR of patient including faulty values
    heart_rate_quality :
        boolean of length n, Quality rating that tells whether datapoint should be used
    fit_fct :
        (optional) to choose the Interpolation Method
    Returns
    -------
    fitted_heart_rate :
        array of length n, HR of patient with interpolated values instead of faulty values
    heart_rate_rmse : scalar
        RMSE value between raw HR and fitted HR for whole time-series
    heart_rate_rmse_predicted:
        RMSE value between raw HR and fitted HR for fitted HR datapoints
    Notes
    -----
    The default Interpolator is a PChipInterpolator
    """
    # Mask the time and heart rate vectors based on HR signal quality
    quality_false_indices = np.where(heart_rate_quality == False)

    time_masked = np.delete(time_vector, quality_false_indices)
    heart_rate_masked = np.delete(heart_rate, quality_false_indices)

    # Fit the selected spine function
    cs = fit_fct(time_masked, heart_rate_masked)

    # Compute the fitted values
    fitted_heart_rate = cs(time_vector)

    # Compute the RMSE of the whole hr and only the fitted part
    heart_rate_rmse = np.sqrt(((heart_rate - fitted_heart_rate) ** 2).mean())

    predicted_hr_values = np.take(fitted_heart_rate, quality_false_indices)
    true_hr_values = np.take(heart_rate, quality_false_indices)
    heart_rate_rmse_predicted = np.sqrt(((true_hr_values[0] - predicted_hr_values[0]) ** 2).mean())
    return fitted_heart_rate, heart_rate_rmse, heart_rate_rmse_predicted


def warp_time_signals(x, y, time_x, time_y):
    """
    Applies DTW and performs time warping transformation according to warping path.
    Parameters
    ----------
    x, y:
        Arrays of length n, signals
    time_x, time_y:
        Arrays of length n, corresponding time-stamps in s starting at 0
    Returns
    -------
    x_warped, y_warped:
        Warped time-series, time is discrete here with meaningless, constant frequency
    dtw_distance:
        Distance measure from DTW cost matrix.
    warp_path:
        DTW-derived warp path which is needed for time warping transformation.
    Notes
    -----
    x and y can have different lengths.
    """
    x_temporal = np.vstack((time_x, x)).T
    y_temporal = np.vstack((time_y, y)).T
    dtw_distance, warp_path = fastdtw(x_temporal, y_temporal, dist=euclidean)

    # Transform time-series into new domain:
    warp_path_array = np.array(warp_path).T
    x_warped = np.zeros(warp_path_array.shape[1])
    y_warped = np.zeros(warp_path_array.shape[1])

    for i in range(x_warped.shape[0]):
        x_warped[i] = x[warp_path_array[0, i]]
        y_warped[i] = y[warp_path_array[1, i]]
    return x_warped, y_warped, dtw_distance, warp_path


user = 642
hr_ppg = pd.read_csv('/home/siri/PycharmProjects/medical-metrics-analysis/data/output/ppg_hr_user_sensei_' + str(user) + '.csv').drop(['Unnamed: 0'], axis=1)
hr_ecg = pd.read_csv('/home/siri/PycharmProjects/medical-metrics-analysis/data/output/ecg_hr_user_sensei_' + str(user) + '.csv').drop(['Unnamed: 0'], axis=1)

####################################
#      Select random samples       #
####################################

p = 0.1  # 10% of samples are assumed to be faulty
N_ppg = hr_ppg['HR'].values.shape[0]
N_ecg = hr_ecg['HR'].values.shape[0]
boolean_array_ppg = hr_ppg['final'].values
boolean_array_ecg = hr_ecg['final'].values

###########################################
#      Choose the best Interpolator       #
###########################################

choose_best_interpolator = False

if choose_best_interpolator:
    interpolation_methods = {
        'CubicSpline: ': CubicSpline,
        'interp1d': interp1d,
        'PchipInterpolator': PchipInterpolator,
        'Akima1DInterpolator': Akima1DInterpolator,
    }

    repetitions = np.arange(100)
    method_rmses = {}
    for name, method in interpolation_methods.items():
        try:
            total_list = []
            predicted_list = []
            for rep in repetitions:
                yf_ppg, hr_ppg_rmse_total, hr_ppg_rmse_predicted = fit_hr(hr_ppg['time'].values, hr_ppg['HR'].values, boolean_array_ppg,
                                                                          method)
                total_list.append(hr_ppg_rmse_total)
                predicted_list.append(hr_ppg_rmse_predicted)
            method_rmses[name] = np.mean(predicted_list)
        except:
            print(str(name) + 'failed')
            method_rmses[name] = np.mean(-1)

    fig, ax = plt.subplots()
    ax.bar(method_rmses.keys(), method_rmses.values(), color='maroon', width=0.4)
    ax.set_xlabel('Method')
    ax.set_ylabel('HR fitting RMSE')
    ax.set_title('HR fitting - comparison of methods (n=100)')
    plt.tight_layout()
    plt.show()

###########################################
#      PchipInterpolator HR fitting       #
###########################################

yf_ppg, hr_ppg_rmse_total, hr_ppg_rmse_predicted = fit_hr(hr_ppg['time'].values, hr_ppg['HR'].values, boolean_array_ppg)
yf_ecg, hr_ecg_rmse_total, hr_ecg_rmse_predicted = fit_hr(hr_ecg['time'].values, hr_ecg['HR'].values, boolean_array_ecg)
hr_ppg['HR_fitted'] = yf_ppg
hr_ecg['HR_fitted'] = yf_ecg

print(f'{hr_ppg_rmse_predicted=}')
print(f'{hr_ecg_rmse_predicted=}')

###########################################
#        Plotting the fitted series       #
###########################################

fig, ax = plt.subplots(4, figsize=(10, 10))
fig.suptitle('PPG and ECG series after Spline-based Interpolation')
ax[0].plot(hr_ppg['time'], hr_ppg['HR_fitted'], label='PPG predicted', linewidth=0.4)
ax[0].plot(hr_ppg['time'], hr_ppg['HR'], label='PPG true', linewidth=0.4)
ax[0].legend()

ax[1].plot(hr_ppg['time'], hr_ppg['label'], label='PPG Label')
ax[1].legend()
ax[1].set_yticks([-1, 0, 1, 2, 3])

ax[2].plot(hr_ecg['time'], hr_ecg['HR_fitted'], label='ECG predicted', linewidth=0.4)
ax[2].plot(hr_ecg['time'], hr_ecg['HR'], label='ECG true', linewidth=0.4)
ax[2].legend()

ax[3].plot(hr_ecg['time'], hr_ecg['label'], label='ECG Label')
ax[3].set_yticks([-1, 0, 1, 2, 3])
ax[3].legend()
fig.tight_layout()
plt.show()

plt.figure()
fig, ax = plt.subplots(figsize=(20, 8))

ax.plot(hr_ppg.time, hr_ppg.HR_fitted, color="tab:blue")
ax.plot(hr_ecg.time, hr_ecg.HR_fitted, color="tab:orange")
ax.set_xlabel("Time [s]", fontsize = 14)
ax.set_ylabel("Heart/Pulse Rate", fontsize = 14)

ax2=ax.twinx()
ax2.plot(hr_ppg.time, hr_ppg.label, color="tab:red", linewidth=3)
ax2.set_ylabel("Label", fontsize = 14)
ax.legend(['Pulse Rate [PPG]', 'Heart Rate [ECG]'], loc='upper left')
ax2.legend(['Label'], loc='upper right')
plt.title("ECG Signal over time with corresponding Label")

start_time = 750
end_time = 1250
interval_indices_ppg = hr_ppg.index[(hr_ppg['time'] > start_time) & (hr_ppg['time'] < end_time)].tolist()
interval_indices_ecg = hr_ecg.index[(hr_ecg['time'] > start_time) & (hr_ecg['time'] < end_time)].tolist()
intersecting_values = list(set(interval_indices_ppg) & set(interval_indices_ecg))

section_hr_ppg = hr_ppg[(hr_ppg['time'] > start_time) & (hr_ppg['time'] < end_time)]
section_hr_ecg = hr_ecg[(hr_ecg['time'] > start_time) & (hr_ecg['time'] < end_time)]

linewidth = 0.4
fig, ax = plt.subplots(5, figsize=(10, 10))
ax[0].plot(hr_ppg['time'], hr_ppg['HR'], label='PPG', linewidth=linewidth)
ax[0].plot(hr_ecg['time'], hr_ecg['HR'], label='ECG', linewidth=linewidth)
ax[0].legend()
ax[0].set_ylabel('HR in BPS')

ax[1].plot(section_hr_ppg['time'], section_hr_ppg['HR'], label='PPG', linewidth=linewidth)
ax[1].plot(section_hr_ecg['time'], section_hr_ecg['HR'], label='ECG', linewidth=linewidth)
ax[1].legend()
ax[1].set_ylabel('HR in BPS')

ax[2].plot(section_hr_ppg['time'], section_hr_ppg['HR'], label='PPG HR true', linewidth=linewidth)
ax[2].plot(section_hr_ppg['time'], section_hr_ppg['HR_fitted'], label='PPG HR fit', linewidth=linewidth)
ax[2].legend()
ax[2].set_ylabel('HR in BPS')

ax[3].plot(interval_indices_ppg, hr_ppg['HR'].values[interval_indices_ppg], label='PPG', linewidth=linewidth)
ax[3].plot(interval_indices_ecg, hr_ecg['HR'].values[interval_indices_ecg], label='ECG', linewidth=linewidth)
ax[3].legend()
ax[3].set_ylabel('HR in BPS')

ax[4].plot(intersecting_values, hr_ppg['HR'].values[intersecting_values], label='PPG', linewidth=linewidth)
ax[4].plot(intersecting_values, hr_ecg['HR'].values[intersecting_values], label='ECG', linewidth=linewidth)
ax[4].legend()
ax[4].set_ylabel('HR in BPS')

fig.tight_layout()
plt.show()

corr, _ = pearsonr(hr_ppg['HR'].values[intersecting_values], hr_ecg['HR'].values[intersecting_values])
print("CORR: " + str(corr))


####################################################
#        Sectioning data according to labels       #
####################################################


def define_sections(df_input):
    """
    Sections HR time-series according to labels.
    Parameters
    ----------
    df_input:
        HR time-series dataframe including labels.
    Returns
    -------
    section_df:
        Dataframe containing all sections of the HR dataframe.
    """
    df_input = df_input[df_input['label'].notna()]
    label_change_indices = df_input['label'][df_input['label'].diff() != 0].index.tolist()  # The value is always the first value
    sections = np.zeros((len(label_change_indices), 2))

    # creating the dataframe
    df = pd.DataFrame(data=sections, columns=['start', 'end'])
    for index, row in df.iterrows():
        if index == len(label_change_indices) - 1:
            df.loc[index, 'start'] = label_change_indices[int(index)]
            df.loc[index, 'end'] = len(df_input) - 1
        elif index < len(label_change_indices) - 1:
            df.loc[index, 'start'] = label_change_indices[int(index)]
            df.loc[index, 'end'] = label_change_indices[int(index) + 1] - 1

    section_df = df_input.copy()
    for index, row in df.iterrows():
        section_df.loc[row['start']:row['end'], 'section_id'] = int(index)
    return section_df


def detect_consecutive_bad_quality_labels(bool_array, bad_qual_threshold=8):
    """
    Detection of consecutive sections of bad quality hr estimation datapoints.
    A threshold can be set. All sections with a sequence of 'bad_qual_threshold' or more are divided into two sub-sections.
    The sequence of False values is discarded.
    Parameters
    ----------
    bool_array:
        The boolean array which specifies the quality of HR estimation datapoints.
    bad_qual_threshold:
        All sections with a sequence of 'bad_qual_threshold' or more False-values are divided into two sub-sections.
    Returns
    -------
    long_section_start_arr, long_section_end_arr:
        Arrays of same length that specify the start and end indices of the boolean array where the threshold sequence length is observed.
    Notes
    -----
    The threshold should be set based on a RMSE analysis of the signal.
    Ideally, it would have different values for ecg and ppg.
    """
    prev_value = True
    long_section_start = []
    long_section_end = []
    for i, val in enumerate(bool_array):
        if val == False and prev_value == True:
            long_section_start.append(i)
        if val == True and prev_value == False:
            long_section_end.append(i)
        prev_value = val
    if bool_array[-1] == False:
        long_section_end.append(len(bool_array))
    bad_qual_section_length = np.array(long_section_end) - np.array(long_section_start)
    long_bad_qual_idx = np.where(bad_qual_section_length >= bad_qual_threshold)
    long_section_start_arr = np.array(long_section_start)[long_bad_qual_idx]
    long_section_end_arr = np.array(long_section_end)[long_bad_qual_idx]
    assert len(long_section_start) == len(long_section_end)

    if long_bad_qual_idx[0].shape[0] > 0:
        subsection_start = np.zeros(long_section_end_arr.shape[0] + 1)
        subsection_start[1:] = long_section_end_arr
        subsection_end = np.zeros(long_section_end_arr.shape[0] + 1)
        subsection_end[0:-1] = long_section_start_arr - 1
        subsection_end[-1] = len(bool_array) - 1
        return list(long_section_start_arr), list(long_section_end_arr)
    return [], []


start_idx_array_ppg, end_idx_array_ppg = detect_consecutive_bad_quality_labels(boolean_array_ppg)
start_idx_array_ecg, end_idx_array_ecg = detect_consecutive_bad_quality_labels(boolean_array_ecg)


def faulty_sections(hr_df, hr_df2, start_idx_array_ppg, end_idx_array_ppg, start_idx_array_ecg, end_idx_array_ecg):
    """
    Function description.
    Parameters
    ----------
    hr_df, hr_df2:
        Dataframes containing either ecg- or ppg-derived HR time-series data.
    start_idx_array_ppg, end_idx_array_ppg, start_idx_array_ecg, end_idx_array_ecg:
        Lists of indices with the start and end values for editing the dataframe
        such that no long faulty signal quality sequences are contained anymore.
    Returns
    -------
    hr_df, hr_df2:
        Dataframes without long faulty HR quality sequences.
    """
    start_list = []
    end_list = []
    for start, end in zip(start_idx_array_ppg, end_idx_array_ppg):
        start_time = hr_df.loc[start, 'time']
        start_list.append(start_time)
        try:
            end_time = hr_df.loc[end, 'time']
        except:
            end_time = hr_df.loc[hr_df.last_valid_index(), 'time']
        end_list.append(end_time)
    for start, end in zip(start_idx_array_ecg, end_idx_array_ecg):
        start_time = hr_df2.loc[start, 'time']
        start_list.append(start_time)
        try:
            end_time = hr_df2.loc[end, 'time']
        except:
            end_time = hr_df2.loc[hr_df2.last_valid_index(), 'time']
        end_list.append(end_time)
    for start, end in zip(start_list, end_list):
        faulty_indices_hr_df = hr_df.loc[(start < hr_df['time']) & (end > hr_df['time'])].index.tolist()
        faulty_indices_hr_df2 = hr_df2.loc[(start < hr_df2['time']) & (end > hr_df2['time'])].index.tolist()
        if len(faulty_indices_hr_df2) > 1 and len(faulty_indices_hr_df2) > 1:
            hr_df.loc[faulty_indices_hr_df, 'label'] = -2
            hr_df2.loc[faulty_indices_hr_df2, 'label'] = -2
    return hr_df, hr_df2


hr_ppg, hr_ecg = faulty_sections(hr_ppg, hr_ecg, start_idx_array_ppg, end_idx_array_ppg, start_idx_array_ecg, end_idx_array_ecg)
hr_ppg_sectioned = define_sections(hr_ppg)
hr_ecg_sectioned = define_sections(hr_ecg)

section_idx_list = hr_ppg_sectioned['section_id'].unique()
section_dict = {
    -1: [],
    0: [],
    1: [],
    2: [],
    3: []
}

section_time_warp_cost = {
    -1: [],
    0: [],
    1: [],
    2: [],
    3: []
}

section_pearson_corr = {
    -1: [],
    0: [],
    1: [],
    2: [],
    3: []
}

section_l2_dist = {
    -1: [],
    0: [],
    1: [],
    2: [],
    3: []
}

section_counter = {
    -1: 0,
    0: 0,
    1: 0,
    2: 0,
    3: 0
}

for section in section_idx_list:
    hr_ppg_section = hr_ppg_sectioned.loc[hr_ppg_sectioned['section_id'] == section].reset_index()
    start_time = hr_ppg_section.loc[0, 'time']
    end_time = hr_ppg_section.loc[hr_ppg_section.last_valid_index(), 'time']
    if end_time - start_time < 5:
        continue

    hr_ecg_section_indices = hr_ecg.loc[(hr_ecg['time'] > start_time) & (hr_ecg['time'] < end_time)].index.tolist()
    hr_ecg_section = hr_ecg.loc[hr_ecg_section_indices].reset_index()
    section_label = int(hr_ppg_section['label'].values[0])

    # Very short sections (mostly label = -2 (section splitting) are discarded
    if len(hr_ecg_section) < 2 or len(hr_ppg_section) < 2:
        continue

    x_warping = hr_ppg_section['HR_fitted']
    y_warping = hr_ecg_section['HR_fitted']
    time_x = hr_ppg_section['time'].values - hr_ppg_section['time'].values[0]
    time_y = hr_ecg_section['time'].values - hr_ecg_section['time'].values[0]

    x_warped, y_warped, dtw_distance, warp_path = warp_time_signals(x_warping, y_warping, time_x, time_y)
    dtw_dist_normalized = dtw_distance / len(warp_path)

    if section_label == 0:
        fig, ax = plt.subplots()
        ax.plot(hr_ppg_section['time'], hr_ppg_section['HR_fitted'], label='ppg')
        ax.plot(hr_ecg_section['time'], hr_ecg_section['HR_fitted'], label='ecg')
        ax.set_title('Label: ' + str(section_label) + ', Section#: ' + str(section_counter[section_label]) + ', DTW Dist.: ' + str(
            round(dtw_dist_normalized, 3)) + ', User: ' + str(user))
        ax.set_ylabel('HR')
        ax.set_xlabel('Time in s')
        ax.legend()
        plt.show()

    # Implement L2 distance as similarity measure
    # Shift such that one signal starts at time Zero
    min_time = min(hr_ppg_section.loc[0, 'time'], hr_ecg_section.loc[0, 'time'])
    hr_ppg_section['time'] = hr_ppg_section['time'] - min_time
    hr_ecg_section['time'] = hr_ecg_section['time'] - min_time

    # Convert to datetime, resample, interpolate and reindex
    hr_ppg_section_datetime = hr_ppg_section.set_index('time')
    hr_ppg_section_datetime.index = pd.to_timedelta(hr_ppg_section_datetime.index, unit='s')
    hr_before = hr_ppg_section_datetime['HR_fitted'].values
    hr_ppg_section_res = hr_ppg_section_datetime.resample('1s').mean().interpolate('linear')
    hr_ecg_section_datetime = hr_ecg_section.set_index('time')
    hr_ecg_section_datetime.index = pd.to_timedelta(hr_ecg_section_datetime.index, unit='s')
    hr_before_ecg = hr_ecg_section_datetime['HR_fitted'].values
    hr_ecg_section_res = hr_ecg_section_datetime.reindex(hr_ecg_section_datetime.index.union(hr_ppg_section_res.index)).interpolate(
        method='time').reindex(hr_ppg_section_res.index)

    # Drop first value (this would have to be extrapolated)
    l2_norm = norm(hr_ppg_section_res['HR_fitted'].values[1::] - hr_ecg_section_res['HR_fitted'].values[1::]) / len(
        hr_ppg_section_res['HR_fitted'].values[1::])
    try:
        pearson_corr, _ = pearsonr(hr_ppg_section_res['HR_fitted'].values[1::], hr_ecg_section_res['HR_fitted'].values[1::])
    except:
        print('NaN values found')
    # Dependent on the sectioning some values at the beginning might be NaNs
    if np.isnan(l2_norm):
        l2_norm = norm(hr_ppg_section_res['HR_fitted'].values[3::] - hr_ecg_section_res['HR_fitted'].values[3::]) / len(
            hr_ppg_section_res['HR_fitted'].values[3::])
        pearson_corr, _ = pearsonr(hr_ppg_section_res['HR_fitted'].values[3::], hr_ecg_section_res['HR_fitted'].values[3::])

    show_dtw_figures = False
    if show_dtw_figures or section_label == 0:
        fig, ax = plt.subplots(4, figsize=(12, 16))
        ax[0].plot(x_warping, label='x', color='blue', marker='o', markersize=10, linewidth=5)
        ax[0].plot(y_warping, label='y', color='red', marker='o', markersize=10, linewidth=5)
        ax[0].legend()
        ax[0].set_title('Initial signal')
        for [map_x, map_y] in warp_path:
            ax[1].plot([map_x, map_y], [x_warping[map_x], y_warping[map_y]], '-k')
        ax[1].plot(x_warping, label='x', color='blue', marker='o', markersize=10, linewidth=5)
        ax[1].plot(y_warping, label='y', color='red', marker='o', markersize=10, linewidth=5)
        ax[1].set_title('Warping')
        ax[1].legend()
        ax[2].plot(x_warped, label='x', color='blue', marker='o', markersize=10, linewidth=5)
        ax[2].plot(y_warped, label='y', color='red', marker='o', markersize=10, linewidth=5)
        ax[2].set_title('Warped signal')
        ax[2].legend()
        ax[3].plot(x_warped, label='x')
        ax[3].plot(y_warped, label='y')
        ax[3].legend()
        plt.show()
        fig.tight_layout()
        fig.savefig('time_warping.png')

    # pearson_corr, _ = pearsonr(x_warped, y_warped)
    if section_label != -2 and not np.isnan(pearson_corr):
        section_time_warp_cost[section_label].append(dtw_dist_normalized)
        section_pearson_corr[section_label].append(pearson_corr)
        section_l2_dist[section_label].append(l2_norm)

    print("PPG section len: " + str(len(hr_ppg_section)))
    print("ECG section len: " + str(len(hr_ecg_section)))

    min_len = min(len(hr_ppg_section), len(hr_ecg_section))
    max_len = max(len(hr_ppg_section), len(hr_ecg_section))

    # Append with zeros:
    longer_series = np.zeros(max_len)
    if len(hr_ppg_section) > len(hr_ecg_section):
        longer_series[0:min_len] = hr_ecg_section['HR_fitted'].values
        x = hr_ppg_section['HR_fitted'].values
        y = longer_series
    else:
        longer_series[0:min_len] = hr_ppg_section['HR_fitted'].values
        x = longer_series
        y = hr_ecg_section['HR_fitted'].values

    corr, p = pearsonr(x, y)
    print("Pearson Corr: " + str(corr))
    max_corr, phase_lag = cross_correlation_eval(x, y)
    print("MAX CORR: " + str(max_corr))
    print("PHASE LAG: " + str(phase_lag) + '\n\n\n')

    section_label = int(hr_ppg_section['label'].values[0])
    if section_label != -2:
        section_dict[section_label].append(max_corr)

    show_figures = False
    if show_figures:
        linewidth = 1.0
        fig, ax = plt.subplots(3, figsize=(10, 10))
        fig.suptitle('Label: ' + str(section_label) + ' Lag: ' + str(phase_lag) + ' MaxXCorr: ' + str(round(max_corr, 3)))
        ax[0].plot(x, label='PPG predicted', linewidth=linewidth)
        ax[0].plot(hr_ppg_section['HR'], label='PPG true', linewidth=linewidth)
        ax[0].legend()
        ax[0].set_xlim([0, max_len])

        ax[1].plot(y, label='ECG predicted', linewidth=linewidth)
        ax[1].plot(hr_ecg_section['HR'], label='ECG true', linewidth=linewidth)
        ax[1].legend()
        ax[1].set_xlim([0, max_len])

        ax[2].xcorr(x, y, usevlines=True, normed=True, lw=2)
        ax[2].grid(True)

        fig.tight_layout()
        plt.show()
    if section_label != -2:
        section_counter[section_label] += 1

####################################################################
#       Boxplot of Cross-Correlation results per label class       #
####################################################################


labels, data = section_dict.keys(), section_dict.values()
fig, ax = plt.subplots()
ax.set_title('Section-wise Cross Correlation per label')
ax.boxplot(data)
ax.set_xlabel('Label')
ax.set_ylabel('Cross Correlation')
ax.set_xticks(range(1, len(labels) + 1), labels)
fig.tight_layout()
plt.show()

labels, data = section_time_warp_cost.keys(), section_time_warp_cost.values()
fig, ax = plt.subplots()
ax.set_title('Section-wise Time Warping Cost per label, User: ' + str(user))
ax.boxplot(data)
ax.set_xlabel('Label')
ax.set_ylabel('Time Warping Cost')
ax.set_xticks(range(1, len(labels) + 1), labels)
fig.tight_layout()
plt.show()

labels, data = section_pearson_corr.keys(), section_pearson_corr.values()
fig, ax = plt.subplots()
ax.set_title('Section-wise Pearson Correlation, User: ' + str(user))
ax.boxplot(data)
ax.set_xlabel('Label')
ax.set_ylabel('Pearson correlation')
ax.set_xticks(range(1, len(labels) + 1), labels)
fig.tight_layout()
plt.show()

labels, data = section_l2_dist.keys(), section_l2_dist.values()
fig, ax = plt.subplots()
ax.set_title('Section-wise L2 Norm, User: ' + str(user))
ax.boxplot(data)
ax.set_xlabel('Label')
ax.set_ylabel('L2 Norm')
ax.set_xticks(range(1, len(labels) + 1), labels)
fig.tight_layout()
plt.show()