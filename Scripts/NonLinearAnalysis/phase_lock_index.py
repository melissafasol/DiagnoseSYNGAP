import os 
import numpy as np
import pandas as pd
import scipy


def channel_pairs(channels):
    pairs = []
    for i in range(len(channels)):
        for j in range(i + 1, len(channels)):
            pairs.append([channels[i], channels[j]])
    
    return pairs

channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12,13]
channel_combinations = channel_pairs(channels)



def phase_lock_index(filtered_data, epoch, channel_combinations):
    
    for channels in channel_combinations:
        phase1 = np.angle(scipy.signal.hilbert(filtered_data[channels[0], epoch]))
        phase2 = np.angle(scipy.signal.hilbert(filtered_data[channels[1], epoch]))
        
    def phase_locking_index(phase1, phase2):
        phase_freq = np.angle(np.exp(1j * (phase1 - phase2)))
        pli = np.abs(np.mean(np.sign(phase_freq)))
        return pli

    pli_value = phase_locking_index(phase1, phase2)
    print("Phase Locking Index (PLI):", pli_value)