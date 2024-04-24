import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.lines import Line2D
import numpy as np

def split_into_epochs(data, sampling_rate, num_seconds):
    
    data_points = sampling_rate*num_seconds #256*30
    new_len = len(data[1]) - len(data[1])%data_points
    split_data = data[:, 0:new_len]
    number_epochs = new_len/sampling_rate/num_seconds
    epochs = np.split(split_data, int(number_epochs), axis = 1)
    
    return int(number_epochs), epochs

def process_data_for_genotype_frequency(data, genotype_human, genotype_label, frequency):
    """
    Process data for a given genotype and frequency.
    
    Args:
        data (DataFrame): The dataframe containing the power measurements.
        genotype_human (list): List of Patient IDs for the specific genotype.
        genotype_label (str): The label of the genotype ('WT' or 'GAP').
        frequency (str): The frequency label ('Delta', 'Alpha', 'Beta', 'Gamma').
    
    Returns:
        DataFrame: A dataframe containing the means for each patient.
    """
    ls = []
    for idx in genotype_human:
        patient_mean = data.loc[data['Patient_ID'] == idx]["Power"].mean()
        mean_dict = {'Patient_ID': [idx], 'Genotype': genotype_label, 'Frequency': [frequency],
                     'Power': [patient_mean]}
        mean_df = pd.DataFrame(data=mean_dict)
        ls.append(mean_df)
    return pd.concat(ls)


def plot_frequency_data(ax, data, frequency, hue_order_palette, palette_stats, pointplot_palette):
    """
    Plot both bar and strip plots for a given frequency on the specified Axes object.
    
    Args:
        ax: The Axes object to plot on.
        data: The DataFrame containing the data to plot.
        frequency: The name of the frequency to set as the xticklabel.
        hue_order_palette: The order of hues for plotting.
        palette_stats: The color palette for the bar plots.
        pointplot_palette: The color palette for the strip plots.
    """
    sns.barplot(x='Frequency', y='Power', hue='Genotype', errorbar=("se"), data=data, width=1.0,
                hue_order=hue_order_palette, palette=palette_stats, ax=ax)
    data_mean = pd.DataFrame(data.groupby(['Patient_ID', 'Genotype', 'Frequency'])['Power'].mean().reset_index())
    sns.stripplot(x='Frequency', y='Power', hue='Genotype', data=data_mean, hue_order=hue_order_palette,
                  palette=pointplot_palette, edgecolor='k', sizes=(50, 50), dodge=True, linewidth=1, ax=ax)
    ax.legend([], [], frameon=False)
    ax.set_yscale('log')
    ax.set(xlabel=None, xticklabels=[frequency], ylabel="")

def bar_and_strip_plots(delta, alpha, beta, gamma):
    f, axs = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    sns.set_style("white")
    hue_order_palette = ['WT', 'GAP']
    palette_stats = ['darkblue', 'orangered']
    pointplot_palette = ['white', 'white']
    
    frequencies = ['Delta', 'Alpha', 'Beta', 'Gamma']
    data_frames = [delta, alpha, beta, gamma]

    for ax, data, freq in zip(axs, data_frames, frequencies):
        plot_frequency_data(ax, data, freq, hue_order_palette, palette_stats, pointplot_palette)
        ax.set_ylabel("log Power (\u03bc$\\mathregular{V^{2}}$)" if freq == 'Delta' else "")
        
    # Hide spines for all subplots
    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    # Adding a shared legend
    custom_lines = [Line2D([0], [0], color='darkblue', lw=4),
                    Line2D([0], [0], color='orangered', lw=4)]
    labels = ['WT', 'GAP']
    f.legend(custom_lines, labels, loc='upper right', frameon=False)

    plt.tight_layout()
    return f