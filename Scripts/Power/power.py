import os
import sys
import artifactdetection as ad

sys.path.insert(0, '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Scripts/Preprocessing')
from constants import SYNGAP_baseline_start, SYNGAP_baseline_end, analysis_ls, SYNGAP_1_ls, SYNGAP_2_ls

channel_indices  = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15]
directory_path = '/home/melissa/PREPROCESSING/SYNGAP1/numpyformat_baseline/'

analysis_ls = [ 'S7063', 'S7064', 
               'S7069', 'S7070', 'S7072', 'S7086', 'S7091', 'S7101', 'S7094', 'S7096', 'S7098',
               'S7068', 'S7074', 'S7075', 'S7071', 'S7076']

for animal in analysis_ls:
    print(animal)
    for channel in channel_indices:
        print(channel)
        save_folder = f'/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Results/Preprocessing/channel_{channel}/'
        load_files = ad.LoadFiles(directory_path = directory_path, animal_id = animal)
        if animal in SYNGAP_2_ls:
            num_epochs = 34560
            ad.two_files(load_files = load_files, animal = animal, num_epochs = num_epochs, chan_idx = channel,
                  save_folder = save_folder, start_times_dict = SYNGAP_baseline_start, end_times_dict = SYNGAP_baseline_end)
        if animal in SYNGAP_1_ls:
            num_epochs = 17280
            ad.one_file(load_files = load_files, animal = animal, num_epochs = num_epochs, chan_idx = channel,
                save_folder = save_folder, start_times_dict = SYNGAP_baseline_start, end_times_dict = SYNGAP_baseline_end)