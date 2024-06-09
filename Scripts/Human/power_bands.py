# Standard library imports
import numpy as np
import scipy.signal
import sys

# Third-party imports
import mne
from mne_features.univariate import compute_higuchi_fd, compute_hurst_exp

from preprocess_human import load_filtered_data, split_into_epochs, select_clean_indices
sys.path.insert(0, '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Scripts/Preprocessing')
from constants import patient_list


# Constants
SAMPLING_RATE = 256
WINDOW = 'hann'
NPERSEG = 7680
FREQUENCY_BANDS = {'delta': (15,61), #1-4Hz
                    'theta': (120,211), #4-8Hz
                    'alpha': (140, 361), #8-12Hz
                    'beta': (450, 901)} #13-30Hz

#indices for frequency bands
# delta [15:61] [1:4]
# theta [120: 211] [4:8]
# alpha [140: 361] [8:12]
# beta [450: 901] [13:30]

# Base directories
PROJECT_DIR = '/home/melissa'
DATA_DIR = f'{PROJECT_DIR}/PREPROCESSING/SYNGAP1/SYNGAP1_Human_Data'
RESULTS_DIR = f'{PROJECT_DIR}/RESULTS/XGBoost/Human_SYNGAP1/Theta_Power'
NOISE_DIR = f'{PROJECT_DIR}/PREPROCESSING/SYNGAP1/human_npy/harmonic_idx'

def calculate_power_for_patient(patient_id, frequency_band):
    file_name = f'{patient_id}_(1).edf'
    filtered_data = load_filtered_data(file_path=DATA_DIR, file_name=file_name)
    number_epochs, epochs = split_into_epochs(filtered_data, sampling_rate=SAMPLING_RATE, num_seconds=30)
    clean_indices = select_clean_indices(noise_directory=NOISE_DIR, patient_id=patient_id, total_num_epochs=number_epochs)

    for channel_idx in range(6):  # Assuming 6 channels as per original script
        power_ls = []
        for clean_idx in clean_indices:
            freqs, power = scipy.signal.welch(epochs[clean_idx][channel_idx], fs=SAMPLING_RATE, window=WINDOW, nperseg=NPERSEG)
            avg_power = np.mean(power[FREQUENCY_BANDS[frequency_band][0]:FREQUENCY_BANDS[frequency_band][1]])
            power_ls.append(avg_power)

        power_array = np.array(power_ls)
        np.save(f'{RESULTS_DIR}/{patient_id}_chan_{channel_idx}.npy', power_array)

def main():
    for patient in patient_list:
        print(patient, 'delta')  # Example for one frequency band, expand as needed
        calculate_power_for_patient(patient, 'theta')

if __name__ == "__main__":
    main()
