import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from matplotlib import colormaps
from mne_connectivity.viz import plot_connectivity_circle

def analyse_files(folder_path, file_type, patient_list, wt_list, gap_list):
    all_patients_analysis = []
    genotype_ls = [] 
    for patient in patient_list:
        analysis_file = pd.read_csv(f'{folder_path}{patient}_{file_type}.csv')
        analysis_file['Patient'] = [patient]*len(analysis_file)
        all_patients_analysis.append(analysis_file)
        if patient in wt_list:
            genotype_df = pd.DataFrame(data = {'Genotype': [0]*len(analysis_file)})
            genotype_ls.append(genotype_df)
        elif patient in gap_list:
            genotype_df = pd.DataFrame(data = {'Genotype': [1]*len(analysis_file)})
            genotype_ls.append(genotype_df)
        else:
            print(f'{patient} not in wt or gap')
            
    patients_concat = pd.concat(all_patients_analysis)
    genotype_concat = pd.concat(genotype_ls)
    all_files_concat = pd.concat([genotype_concat, patients_concat], axis = 1)
    
    return all_files_concat


def plot_power(power_df, genotype, save_path, patient):
    fig, axs = plt.subplots(1,1, figsize=(15,10), sharex = True, sharey=True)
    if genotype == 0:
        power_palette = 'winter'
    elif genotype == 1:
        power_palette = 'spring'
    
    sns.lineplot(data= power_df, x='Frequency', y='Power',hue = 'Channel', errorbar = ("se"), linewidth = 2,
            palette = power_palette)
    sns.despine()
    plt.yscale('log')
    axs.set_xlim(1, 35)
    axs.set_ylim(10**-13, 10**-7)

    axs.set_xlabel("Frequency (Hz)", fontsize = 15)
    axs.set_ylabel("log Power (\u03bc$\\mathregular{V^{2}}$)", fontsize = 15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize = 15)

    plt.suptitle(f'{patient}', fontsize = 15, fontweight = 'bold')
    plt.savefig(f'{save_path}{patient}_power.png')
    plt.savefig(f'{save_path}{patient}_power.svg')
    
def barplot_prepare(concat_dataframe, variable_name='hfd', split_index = -1):
    df_melted = concat_dataframe.melt(id_vars=['Genotype', 'Patient'], var_name='Channel', value_name=variable_name)
    df_melted['Channel'] = df_melted['Channel'].apply(lambda x: x.split('_')[split_index])
    df_melted = df_melted.rename(columns={variable_name: f'{variable_name}', 'Channel': 'channel'})
    return df_melted

def separate_power_frequency(df):
    df_melted = df.melt(id_vars=['Patient', 'Epoch'], var_name='Power_Frequency_Channel', value_name='Power')
    df_melted[['Frequency', 'Channel']] = df_melted['Power_Frequency_Channel'].str.extract(r'Power_([a-z]+)_(\d+)')
    df_melted['Channel'] = df_melted['Channel'].astype(int)
    # drop original column
    df_melted = df_melted.drop(columns='Power_Frequency_Channel')
    return df_melted