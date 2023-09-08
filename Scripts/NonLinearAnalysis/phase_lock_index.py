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

    
#frequency range 1 - 48
frequencies = np.arange(1,48,1)


def wpli(filtered_data, channel_combinations):
    
    wpli_ls = []
    
    #hilbert transform to get complex signal 
    analytic_signal = scipy.signal.hilbert(filtered_data)
    for combination in channel_combinations:
        signal_1 = analytic_signal(filtered_data[combination[0]])
        signal_2 = analytic_signal(filtered_data[combination[1]])
        #real part corresponds to the amplitude and the imaginary part can be used to calculate the phase
        phase_diff = np.angle(signal_1) - np.angle(signal_2)
        # Convert phase difference to complex part i(phase1-phase2)
        phase_diff_im = 0*phase_diff+1j*phase_diff
        #wPLI tries to correct for small fluctations in noise by weighting the PLI
        #calc magnitude of phase diff 
        phase_diff_mag = np.abs(np.sin(phase_diff))
        #calculate the signed phase difference (PLI)
        sign_phase_diff = np.sign(np.sin(phase_diff))
        #calculate the nominator (abs and average across time)
        WPLI_nominator = np.abs(np.mean(phase_diff_mag*sign_phase_diff))
        #calculate denominator for normalisation 
        WPLI_denom = np.mean(phase_diff_mag)
        #calculate WPLI
        wpli = WPLI_nominator/WPLI_denom
        #save to array
        wpli_ls.append(wpli)
        