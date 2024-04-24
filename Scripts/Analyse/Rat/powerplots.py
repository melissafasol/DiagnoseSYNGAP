'''code for log power plots and bar & strip plots for individual frequency bands '''

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.lines import Line2D

def prepare_mean_data(data):
    """Calculate the mean of power for each genotype and frequency."""
    return pd.DataFrame(data.groupby(['Animal_ID', 'Genotype', 'Frequency'])['Power'].mean().reset_index())

def plot_bar_and_strip(ax, data, hue_order_palette, palette_stats, pointplot_palette):
    """Plot both bar and strip plots on the same Axes object."""
    sns.barplot(x='Frequency', y='Power', hue='Genotype', errorbar=("se"), data=data, width=1.0,
                hue_order=hue_order_palette, palette=palette_stats, ax=ax)
    data_mean = prepare_mean_data(data)
    sns.stripplot(x='Frequency', y='Power', hue='Genotype', data=data_mean, hue_order=hue_order_palette,
                  palette=pointplot_palette, edgecolor='k', dodge=True, linewidth=1, ax=ax)
    ax.legend([], [], frameon=False)
    ax.set_yscale('log')
    ax.set_ylabel("")  # Clear the default ylabel

def bar_and_strip_plots(delta, theta, sigma, beta, gamma, sleepstage):
    plt.figure(figsize=(10, 10))
    sns.set_style("white")
    hue_order_palette = ['WT', 'GAP']
    palette_stats = ['darkblue', 'orangered']
    pointplot_palette = ['white', 'white']
    freq_labels = ['Delta', 'Theta', 'Sigma', 'Beta', 'Gamma']

    f, axs = plt.subplots(1, 5, figsize=(20, 4), sharey=True)

    # Plot each dataset
    for ax, data, label in zip(axs, [delta, theta, sigma, beta, gamma], freq_labels):
        plot_bar_and_strip(ax, data, hue_order_palette, palette_stats, pointplot_palette)
        ax.set(xlabel=None, xticklabels=[label])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

    # Set log scale for y-axis and add a common ylabel
    axs[0].set_ylabel("log Power (\u03bc$\\mathregular{V^{2}}$)")

    # Add title
    plt.suptitle(sleepstage, fontsize=16, fontweight='bold')

    # Custom Legend
    custom_lines = [Line2D([0], [0], color='darkblue', lw=4),
                    Line2D([0], [0], color='orangered', lw=4)]
    labels = ['WT', 'GAP']
    f.legend(custom_lines, labels, loc='upper right', frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return f