import gzip
import os
import shutil
import zipfile
import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sps
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from biosppy.signals import ecg
from scipy.spatial import distance


def euc_dist_quality(templates, mean_waveform, sampling_Frequency, show=False, plt_title="Waveforms centered around Peak"):
    """
    Function to calculate the euclidian distance between each template in an array and a given waveform
    :param templates: array with templates
    :param mean_waveform: given waveform with which each template in the array will be compared
    :param show: set True for showing the plot with each template and the given waveform in red
    :return: returns the sum of the euclidian distance for each template in the array
    """
    euc_dist = []
    for signal in templates:
        if all(math.isfinite(x) for x in signal):
            dist = distance.euclidean(mean_waveform, signal)
            euc_dist.append(dist)
        else:
            euc_dist.append(np.nan)
    if show:
        plt.figure(figsize=(12, 8))
        for wave in templates:
            x_ax = np.arange(0, len(wave) * 1 / sampling_Frequency, 1 / sampling_Frequency)
            plt.plot(x_ax, wave)
        plt.plot(x_ax, mean_waveform, color='r', linewidth=4)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude (uV)')
        plt.title(plt_title)

    return euc_dist


def calculate_sqi(templates, sample_frequency):
    """
    Calculates skewness, kurtosis and signal-to-noise ratio for each template in an array
    :param templates: array with templates
    :param sample_frequency: sample frequency of given signal
    :return: returns the value for skewness, kurtosis and signal-to-noise ratio for each template
    """
    skewness = []
    kurtosis = []
    snr = []

    for signal in templates:
        skewness_value = ecg.sSQI(signal)
        skewness.append(skewness_value)

        kurtosis_value = ecg.kSQI(signal)
        kurtosis.append(kurtosis_value)

        snr_value = calculate_snr(signal, sample_frequency)
        snr.append(snr_value)

    skewness = np.array(skewness)
    kurtosis = np.array(kurtosis)

    return skewness, kurtosis, snr


def calculate_sampling_rate(time_series):
    """
    Calculates the sampling rate of a time series based on a histogram
    :param time_series: time-series for calculating the sampling frequency
    :return: returns sampling frequency
    """
    ts1 = np.histogram(np.diff(time_series))[1][0]
    ts1 = np.round(np.mean(ts1), 7)
    fs1 = 1 / ts1
    return fs1


def calculate_snr(signal, sampling_frequency):
    """
    Function to calculate the signal-to-noise ratio of a signal with a frequency band of 2-40Hz
    :param signal: input signal
    :param sampling_frequency: sampling frequency of input signal
    :return: returns the signal-to-noise ratio of a signal
    """
    signal_freq_band = [2, 40]  # from .. to .. in Hz
    f, pxx_den = sps.periodogram(signal, fs=sampling_frequency, scaling="spectrum")
    if sum(pxx_den):
        signal_power = sum(pxx_den[(signal_freq_band[0] * 10):(signal_freq_band[1] * 10)])
        return signal_power / (sum(pxx_den) - signal_power)


def unzip_all(directory):
    """
    Helper function to unzip all folders, subfolders and files in a given directory
    The function works both for g-zip and zip
    """
    extension1 = ".gz"
    extension2 = ".zip"
    os.chdir(directory)
    for subdir, dirs, files in os.walk(directory):
        for item in files:  # loop through items in dir
            if item.endswith(extension1):  # check for ".gz" extension
                gz_name = os.path.join(subdir, item)  # get full path of files
                file_name = gz_name.rsplit('.', 1)[0]  # get file name for file within (os.path.basename(gz_name))
                with gzip.open(gz_name, "rb") as f_in, open(file_name, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(gz_name)  # delete zipped file
            if item.endswith(extension2):
                gz_name = os.path.join(subdir, item)  # get full path of files
                with zipfile.ZipFile(gz_name, 'r') as zip_ref:
                    zip_ref.extractall(subdir)
                os.remove(gz_name)  # delete zipped file


def align_labels_to_hr(df_hr, df_label, feat_label: str = "label", feat_time_label: str = "time_reset"):
    """
    Function to align the labels from one dataframe that are already in the correct order with another dataframe
    :param df_hr: dataframe that contains the heartrate
    :param df_label: dataframe that contains the labels
    :param feat_label: column name of the label column
    :param feat_time_label: column name of the column that contains the time column in the label dataframe
    :return: returns the df_hr with the label column added
    """
    temp_label = np.nan
    temp_idx = 0
    break_label = -1

    for i in range(df_label.shape[0]):
        if (df_label[feat_label][i] != temp_label) & (np.isfinite(df_label[feat_label][i])):
            temp_time = df_label[feat_time_label][i]
            next_idx = np.argmax(df_hr["time"] > temp_time)

            df_hr.loc[temp_idx:next_idx, ['label']] = [temp_label]

            temp_label = df_label[feat_label][i]
            temp_idx = next_idx

        if i == df_label.shape[0] - 1:
            last_idx = df_hr.last_valid_index()
            df_hr.loc[temp_idx:last_idx, ['label']] = [break_label]

    return df_hr


def align_labels(df_signal, df_label, feat_label: str = "gt_label"):
    """
    Function to align the sensomative labels to the ECG/PPG signal.
    The Function can handle the breaks and multiple labels.
    :param df_signal: dataframe with signal
    :param df_label: dataframe with sensomative labels
    :param feat_label: name of label column in label dataframe
    :return: returns dataframe of signal with added label column
    """
    temp_label = np.nan
    temp_idx = 0
    given_label = np.nan

    temp_time = 0

    resting_period = True
    start = False
    break_label = -1

    for i in range(df_label.shape[0]):
        if start and (df_label["time"][i] - temp_time > 120):
            temp_time = df_label["time"][i]
            next_idx = np.argmax(df_signal["time"] > temp_time)
            df_signal.loc[temp_idx:next_idx, ['label']] = [temp_label]
            temp_idx = next_idx
            start = False
        if (df_label[feat_label][i] != temp_label) & (np.isfinite(df_label[feat_label][i])):

            temp_time = df_label["time"][i]
            next_idx = np.argmax(df_signal["time"] > temp_time)

            if resting_period:
                if temp_label == 1.0:
                    resting_period = False
                    df_signal.loc[temp_idx:next_idx, ['label']] = [break_label]
                else:
                    df_signal.loc[temp_idx:next_idx, ['label']] = [given_label]
                    given_label = 2.0
                    start = True
            else:
                if temp_label == 1.0:
                    if given_label == 18:
                        df_signal.loc[temp_idx:temp_idx + 5000, ['label']] = [break_label]
                        break
                    df_signal.loc[temp_idx:next_idx, ['label']] = [break_label]
                    given_label = given_label + 1
                else:
                    df_signal.loc[temp_idx:next_idx, ['label']] = [given_label]

            temp_label = df_label[feat_label][i]
            temp_idx = next_idx

        if i == df_label.shape[0] - 1:
            df_signal.loc[temp_idx:temp_idx + 5000, ['label']] = [break_label]
            break

    return df_signal


def normalize_to_userBaseline(df, first_sqi_column=2):
    """
    Function to normalize the data based on the RobustScaler.
    This Scaler removes the median and scales the data according to the quantile range.
    The IQR (inter-quantile-range) is the range between the 1st quartile (25th quantile) and the 3rd quartile (75th quantile).
    :param df: dataframe with SQI values
    :param first_sqi_column: Column with first SQI value, in case the first column has user-name etc. in it
    :return: returns the normalized dataframe
    """
    nrm = RobustScaler()  # can also choose StandardScaler or MinMaxScaler, check sklearn documentation for more infos
    nrm.fit(df[df.label == 0].iloc[:, first_sqi_column:])
    df_normalized = pd.DataFrame(nrm.transform(df[df.label >= -1].iloc[:, first_sqi_column:]),
                                 columns=df[df.label >= -1].iloc[:, first_sqi_column:].columns[:],
                                 index=df[df.label >= -1].iloc[:, first_sqi_column:].index)
    df_normalized.insert(0, "label", df[df.label >= -1].label, True)
    df_normalized.insert(0, "user", df[df.label >= -1].user, True)

    return df_normalized


def reclassify_labels(df):  # reclassification for sensei v2
    """
    Function to reclassify the labels of the activities into the desired classes.
    This case:
    Class 0: Resting Period
    Class 1: Static Tasks
    Class 2: Low-Effort Tasks
    Class 3: High-Effort Tasks
    :param df: dataframe with a label column
    :return: returns dataframe with reclassified labels
    """
    df_reclassified = df.replace(
        {'label': {1.0: 0, 2.0: 3, 3.0: 3, 4.0: 3, 5.0: 3, 16: 3, 17: 3, 18: 3, 6.0: 1, 7.0: 1, 8.0: 1, 12.0: 1,
                   13.0: 1, 14.0: 1, 15.0: 1, 9.0: 2, 10.0: 2, 11.0: 2}})
    df_reclassified = df_reclassified.drop(df_reclassified[df_reclassified.label > 18].index)
    return df_reclassified


def combine_csv(directory):
    """
    Helper function to combine all csv files of a folder into 1 file in case they are split
    :param directory: main directory
    :return: returns df with concatenated csv files
    """
    first = True
    for subdir, dirs, files in os.walk(directory):
        for file in sorted(files):
            file_path = os.path.join(subdir, file)
            df_file = pd.read_csv(file_path)
            if first:
                df = df_file.copy()
                first = False
            else:
                df = pd.concat([df, df_file], ignore_index=True)
    return df


def find_paths(directory, keyword, endword):
    """
    Helper function to find all subdirs of folders containing a keyword and ending with a certain string
    :param directory: main directory
    :param keyword: keyword that must be contained in the folder path
    :param endword: word on which the folder path must end
    :return:
    """
    path_list = []
    for subdir, dirs, files in sorted(os.walk(directory)):
        if (keyword in subdir) and (subdir.endswith(endword) == False):
            path_list.append(subdir)
    return path_list


def find_csvs(directory, keyword, endword):
    """
    Helper function to find all csvs in a directory that contain a certain keyword and end on a certain string
    :param directory: main directory
    :param keyword: keyword that must be contained in the csv name
    :param endword: string on which the csv must end (in this case .csv)
    :return:
    """
    path_list = []
    for subdir, dirs, files in sorted(os.walk(directory)):
        for file in sorted(files):
            if (keyword in subdir) and (file.endswith(endword)):
                file_path = os.path.join(subdir, file)
                path_list.append(file_path)
    return path_list


def threshold_analysis(df_normalized, show=True):
    """
    Function to calculate the thresholds for excluding the 3% of the worst data
    :param df_normalized: input dataframe, data should be normalized
    :param show: set show=True to display the histograms of the resting period
    :return: returns a dataframe where each SQI is checked to lie within the threshold or not
    """
    df_threshold = df_normalized.copy()

    skewness_low = np.percentile(df_normalized[df_normalized.label == 0].skewness, 1.5)
    skewness_high = np.percentile(df_normalized[df_normalized.label == 0].skewness, 98.5)
    kurtosis_low = np.percentile(df_normalized[df_normalized.label == 0].sqi_kurtosis, 1.5)
    kurtosis_high = np.percentile(df_normalized[df_normalized.label == 0].sqi_kurtosis, 98.5)
    euc_threshold = np.percentile(df_normalized[df_normalized.label == 0].euc_distance, 97)
    snr_threshold = np.percentile(df_normalized[df_normalized.label == 0].snr, 3)

    df_threshold.skewness = ((df_threshold.skewness > skewness_low) & (df_threshold.skewness < skewness_high)).astype(
        bool)
    df_threshold.sqi_kurtosis = (
            (df_threshold.sqi_kurtosis > kurtosis_low) & (df_threshold.sqi_kurtosis < kurtosis_high)).astype(bool)
    df_threshold.snr = (df_threshold.snr > snr_threshold).astype(bool)
    df_threshold.euc_distance = (df_threshold.euc_distance < euc_threshold).astype(bool)

    df_threshold['final'] = (
            df_threshold.skewness & df_threshold.sqi_kurtosis & df_threshold.snr & df_threshold.euc_distance)

    passing_rate = {}
    class_names = ["Resting Period", "Static Tasks", "Low-Effort Tasks", "High-Effort Tasks"]
    df_renamed = df_normalized.rename(
        columns={"skewness": "Skewness", "sqi_kurtosis": "Kurtosis", "snr": "Signal-to-Noise Ratio",
                 "euc_distance": "L2-Norm"}, errors="raise")
    for i in range(4):
        percentage_pass = len(df_threshold[(df_threshold.label == i) & (df_threshold.final == True)]) / len(
            df_threshold[df_threshold.label == i]) * 100
        passing_rate[i] = percentage_pass

        if show:
            df_renamed[df_renamed.label == i].iloc[:, 1:].hist(bins=20);
            plt.suptitle("Histograms for %s" % class_names[i], fontsize=20)

    return df_threshold, passing_rate
