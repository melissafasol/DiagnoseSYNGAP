import os
import numpy as np
import pandas as pd

class PowerAnalysis:
    def __init__(self, analysis_ls, clean_folder, channel_folder, channel_indices, output_path):
        self.analysis_ls = analysis_ls
        self.clean_folder = clean_folder
        self.channel_folder = channel_folder
        self.channel_indices = channel_indices
        self.output_path = output_path

    def analyze(self):
        for animal in self.analysis_ls:
            print(animal)
            clean_indices = self.get_clean_indices(animal)
            all_channels = []
            for channel in self.channel_indices:
                print(channel)
                all_frequencies = self.process_channel(animal, channel, clean_indices)
                all_channels.append(all_frequencies)

            all_frequencies_chan_2 = self.process_channel(animal, 2, clean_indices, clean_power=True)
            all_channels.append(all_frequencies_chan_2)
            
            self.save_results(animal, all_channels)
            print(f'{animal} saved')
            
    def analyze_slope(self):
        for animal in self.analysis_ls:
            print(f'{animal} slope')
            clean_indices = self.get_clean_indices(animal)
            all_channels_slope = []
            for channel in self.channel_indices:
                print(channel)
                channel_df = pd.read_csv(os.path.join(self.channel_folder, f'channel_{channel}/', f'{animal}_slope.csv'))
                filtered_slope_df = channel_df[channel_df['Epoch'].isin(clean_indices)]
                all_channels_slope.append(filtered_slope_df)
            channel_df = pd.read_csv(os.path.join(self.channel_folder, f'channel_2/', f'{animal}_slope.csv'))
            filtered_slope_df_2 = channel_df[channel_df['Epoch'].isin(clean_indices)]
            all_channels_slope.append(filtered_slope_df_2)
            all_channels_slope_concat = pd.concat(all_channels_slope)
            all_channels_slope_concat.to_csv(os.path.join(self.output_path, f'{animal}_all_channel_slope.csv'))
            print(f'{animal} saved')

    def get_clean_indices(self, animal):
        clean_file_name = f'{animal}_clean_power.csv'
        clean_indices = np.unique(pd.read_csv(os.path.join(self.clean_folder, clean_file_name))['Epoch'])
        return clean_indices
    
    def process_channel(self, animal, channel, clean_indices, clean_power=False):
        if clean_power:
            channel_df = pd.read_csv(os.path.join(self.channel_folder, f'channel_2/', f'{animal}_clean_power.csv'))
        else:
            channel_df = pd.read_csv(os.path.join(self.channel_folder, f'channel_{channel}/', f'{animal}_power.csv'))
            channel_df = channel_df[channel_df['Epoch'].isin(clean_indices)]

        frequency_bands = {
            'delta': (1, 5),
            'theta': (5, 10),
            'sigma': (10, 16),
            'beta': (16, 30),
            'gamma': (30, 48)
        }
        all_frequencies = []
        for band, (low, high) in frequency_bands.items():
            band_df = channel_df.loc[(channel_df['Frequency'] >= low) & (channel_df['Frequency'] < high)]
            average_band = band_df.groupby(['Epoch', 'Animal_ID', 'Channel'])['Power'].mean().reset_index()
            average_band.rename(columns={'Power': f'{band}_power'}, inplace=True)
            all_frequencies.append(average_band)
        
        # Merge all frequency bands on the common columns
        all_frequencies_df = all_frequencies[0]
        for freq_df in all_frequencies[1:]:
            all_frequencies_df = pd.merge(all_frequencies_df, freq_df, on=['Epoch', 'Animal_ID', 'Channel'], how='outer')
        
        return all_frequencies_df

    def save_results(self, animal, all_channels):
        all_channels_concat = pd.concat(all_channels)
        output_path = os.path.join(self.output_path, f'{animal}_all_channels.csv')
        all_channels_concat.to_csv(output_path, index=False)
        