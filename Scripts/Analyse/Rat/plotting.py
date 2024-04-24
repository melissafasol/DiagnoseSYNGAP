'''Class for plotting results'''

import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

def plot_animals_by_channels(animal_id, avg_numpy_array, channels, palette, sleepstage, save_directory):
    
    frequency = np.arange(0.2, 125.6, 0.2)
    fig, axs = plt.subplots(1,1, figsize=(15,10), sharex = True, sharey=True)
    for chan_idx, channel_power in zip(channels, avg_numpy_array):
        sns.lineplot(x = frequency, y = channel_power, linewidth =3, ax = axs, color = palette[chan_idx])
        sns.despine() 
        axs.set_yscale('log')
        axs.set_xlim(1, 48)
        axs.set_ylim(10**-2, 10**3)
        axs.set_xlabel("Frequency (Hz)", fontsize = 15)
        axs.set_ylabel("log Power (\u03bc$\\mathregular{V^{2}}$)", fontsize = 15)

        for axis in ['bottom', 'left']:
            axs.spines[axis].set_linewidth(3)

        tick_values = list(range(1, 54, 6))
        label_list = ['1', '6', '12', '18', '24', '30', '36', '42', '48']
        axs.set_xticks(ticks = tick_values, labels = label_list)
            
        plt.rc(['xtick', 'ytick'], labelsize=16)
    
    custom_lines = [Line2D([0], [0], color= palette[0], lw=2),
                Line2D([0], [0], color= palette[2], lw=2),
                Line2D([0], [0], color= palette[3], lw=2),
                Line2D([0], [0], color= palette[4], lw=2),
                Line2D([0], [0], color= palette[5], lw=2),
                Line2D([0], [0], color= palette[6], lw=2),
                Line2D([0], [0], color= palette[7], lw=2),
                Line2D([0], [0], color= palette[8], lw=2),
                Line2D([0], [0], color= palette[9], lw=2),
                Line2D([0], [0], color= palette[10], lw=2),
                Line2D([0], [0], color= palette[11], lw=2),
                Line2D([0], [0], color= palette[12], lw=2),
                Line2D([0], [0], color= palette[13], lw=2),
                Line2D([0], [0], color= palette[15], lw=2)]
    axs.legend(custom_lines, channels)
    fig.suptitle(str(animal_id) + ' ' + str(sleepstage), y = 0.96, fontsize = 20, fontweight = 'bold') 
    
    plt.savefig(save_directory + str(animal_id) + '_' + str(sleepstage) + '.pdf')