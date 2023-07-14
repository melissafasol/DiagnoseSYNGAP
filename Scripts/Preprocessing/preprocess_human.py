import sys
import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
import scipy 
import seaborn as sns
import signal 


def load_filtered_data(file_path, file_name):
    os.chdir(file_path)
    raw_edf = mne.io.read_raw_edf(str(file_name), eog = [0],
            exclude = ['F3:M2', 'C3:M2', 'O1:M2', 'Position', 'PLM1', 'PLM2',
                       'Snore', 'Flow', 'Effort', 'Thorax', 'Abdomen', 'SpO2', 'Pleth', 'Pulse',
                      'E1:M2', 'E2:M2'], preload=True)

    filtered_data = raw_edf.filter(l_freq = 0.3, h_freq = 35)

    #extract array object to save
    data, times = filtered_data[:, :]

    return data

#calculate divisible length 
#sampling_rate = 256, seconds per epoch = 30
def split_into_epochs(data, sampling_rate, num_seconds):
    
    data_points = sampling_rate*num_seconds #256*30
    new_len = len(data[1]) - len(data[1])%data_points
    split_data = data[:, 0:new_len]
    number_epochs = new_len/sampling_rate/num_seconds
    epochs = np.split(split_data, int(number_epochs), axis = 1)
    
    return int(number_epochs), epochs

#epochs = split_into_epochs(data = data, sampling_rate = 256, num_seconds = 30)



def identify_noisy_epochs(split_epochs, num_channels, num_epochs):
    channels_idx = list(np.arange(0, num_channels))
    epochs_idx = list(np.arange(0, num_epochs))
    
    intercept_noise = []
    slope_noise = []
    for chan in channels_idx:
        for epoch in epochs_idx:
            freq, power = scipy.signal.welch(split_epochs[epoch][chan], window='hann', fs=256, nperseg=7680)
            slope, intercept = np.polyfit(freq, power, 1)
            if intercept < 1e-13:
                int_noise_dict = {'Intercept': [intercept], 'Epoch_IDX': [epoch], 'Channel': [chan]}
                int_noise_df = pd.DataFrame(data = int_noise_dict)
                intercept_noise.append(int_noise_df)
            elif slope > -1e-13:
                slope_noise_dict = {'Slope': [slope], 'Epoch_IDX': [epoch], 'Channel': [chan]}
                slope_noise_df = pd.DataFrame(data = slope_noise_dict)
                slope_noise.append(slope_noise_df)
            
    intercept_noise_concat = pd.concat(intercept_noise)
    slope_noise_concat = pd.concat(slope_noise)
    
    return channels_idx, epochs_idx, intercept_noise_concat, slope_noise_concat


def remove_duplicate_idx(slope_indices_df, intercept_indices_df):
    slope_noisy_indices = slope_indices_df['Epoch_IDX'].to_list()
    intercept_noisy_indices = intercept_indices_df['Epoch_IDX'].to_list()
    rm_dup = list(set(slope_noisy_indices + intercept_noisy_indices))
    
    return rm_dup


def thresholding_algo( y, lag, threshold, influence):
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
            else:
                signals[i] = -1
                filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
                avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
                stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])

    signal_calc = np.asarray(signals)
    harmonic_1 = np.mean(signal_calc[25:50])
    harmonic_2 = np.mean(signal_calc[60:85])
    return harmonic_1, harmonic_2

def harmonic_filter(channels_idx, epochs_idx, split_epochs, noise_indices):
    
    harmonic_noise_df_ls = []

    for chan in channels_idx:
        for epoch in epochs_idx:
            if epoch in noise_indices:
                pass
            else:
                power_calculations = signal.welch(split_epochs[epoch][chan], window = 'hann', fs = 256, nperseg = 7680)
                harmonic_1, harmonic_2 = thresholding_algo(y = power_calculations[1], lag = 30, threshold = 5, influence = 0)
                if harmonic_1 or harmonic_2 > 0:
                    harmonic_indices.append(epoch)
                    harmonic_noise_dict = {'Epoch_IDX': [epoch], 'Channel': [chan]}
                    harmonic_noise_df = pd.DataFrame(data = harmonic_noise_dict)
                    harmonic_noise_df_ls.append(harmonic_noise_df)
                else:
                    pass
    
    if len(harmonic_noise_df_ls)>0:
        harmonic_noise_df_concat = pd.concat(harmonic_noise_df_ls)
        harmonic_noisy_indices = harmonic_noise_df_concat['Epoch_IDX'].to_list()
        harmonic_indices = sorted(list(set(harmonic_noisy_indices)))
        return harmonic_indices
    else:
        return 'no harmonic artefacts'


def plot_individual_indices(test_epoch, channel, num_to_plot):
    
    for i in list(np.arange(0, 200)):
        freq, power = scipy.signal.welch(test_epoch[num_to_plot][channel], window = 'hann', fs = 256, nperseg = 7680)
        slope, intercept = np.polyfit(freq, power, 1)
        plt.plot(freq, power)
        plt.yscale("log")
        plt.show()
        plt.clf()

def calculate_psd(epochs, rm_dup, channels_idx, epochs_idx):
    power_df_ls = []
    for chan in channels_idx:
        for epoch in epochs_idx:
            if epoch in rm_dup:
                pass
            else:
                freq, power = scipy.signal.welch(epochs[epoch][chan], window='hann', fs=256, nperseg=7680)
                power_dict = {'Power': power, 'Frequency': freq,'Epoch_IDX': [epoch]*len(power), 'Channel': [chan]*len(power)}
                power_df = pd.DataFrame(data = power_dict)
                print(power_df)
                power_df_ls.append(power_df)
    
    power_concat = pd.concat(power_df_ls)
    return power_concat
        
def plot_psd(power_concat, save_directory, patient):
    sns.set_style("white")
    fig, axs = plt.subplots(1, 1, figsize=(20,15), sharex = True, sharey = True)
    chan_palette = ['black', 'coral', 'orangered', 'skyblue', 'teal', 'pink', 'darkseagreen']
    hue_order = [0, 1, 2, 3, 4, 5, 6]

    #input colour palette parameters
    sns.lineplot(data = power_concat, x = 'Frequency', y = 'Power', 
    hue = 'Channel', hue_order = hue_order, palette = chan_palette, errorbar = ('se'),linewidth = 2, ax = axs)


    plt.suptitle(str(patient), fontsize = 30, fontweight = 'bold')

    #remove outer border
    sns.despine()
    axs.set_yscale('log')
    sns.despine()
    axs.set_xlim(1, 35)
    #axs.set_ylim(10**-2, 10**3)
    axs.set_xlabel("Frequency (Hz)", fontsize = 25)
    axs.set_ylabel("log Power", fontsize = 25)

    plt.savefig(save_directory + str(patient) + '_all_channels.jpg')