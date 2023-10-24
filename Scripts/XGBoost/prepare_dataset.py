import os
import numpy as np
import pandas as pd

class BuildDataFrame:
    def __init__(self, br_directory, data_directory, SYNGAP_het, SYNGAP_wt, second_br_list, feature_list):
        
        '''
        This class is to prepare a dataframe for binary classification
        where you can load sleep scores for epochs and different metrics
        of quantification (e.g entropy, cross correlation, etc)
        
        br_directory = /path/to/sleep/score/files
        data_directory = /path/to/data/files
        SYNGAP_het = ids of positive class (e.g mutant) e.g ['71']
        SYNGAP_wt = ids of negative class (e.g wild-type) e.g ['70']
        second_br_list = list of ids if there are multiple recordings for one animal e.g ['70', '71']
        feature_list = list of features to include e.g ['entropy', 'cross_correlation', 'phase_locking"value']
        '''
        
        self.br_directory = br_directory
        self.data_directory = data_directory
        self.SYNGAP_het = SYNGAP_het
        self.SYNGAP_wt = SYNGAP_wt
        self.second_br_list = second_br_list

    def load_and_preprocess_data(self, animal):
        br_state = None  # Initialize br_state to None, keep this as none if you have no sleep scores

        if animal not in self.second_br_list:
            br_1 = pd.read_csv(os.path.join(self.br_directory, f"{animal}_BL1.csv"))
            br_state = br_1['brainstate'].to_numpy()

        if animal in self.second_br_list:
            br_1 = pd.read_csv(os.path.join(self.br_directory, f"{animal}_BL1.csv"))
            br_state_1 = br_1['brainstate'].to_numpy()
            br_2 = pd.read_csv(os.path.join(self.br_directory, f"{animal}_BL2.csv"))
            br_state_2 = br_2['brainstate'].to_numpy()
            br_state = np.concatenate([br_state_1, br_state_2])

        # Load and preprocess data for different regions
        combined_data = self.load_and_preprocess_feature_dataframe(animal)

        #genotype data - used to train the features to predict positive and negative classes    
        if animal in self.SYNGAP_het:
            genotype = 1
            combined_data['Genotype'] = [genotype]*len(combined_data)
        elif animal in self.SYNGAP_wt:
            genotype = 0
            combined_data['Genotype'] = [genotype]*len(combined_data) 
        else:
            raise ValueError("Animal ID not in positive or negative class")
        
        combined_data['Animal_ID'] = [animal]*len(combined_data) #used for bootstrapping and to test model's accuracy by animals/patients rather than by epoch

        if br_state is not None:
            combined_data['SleepStage'] = br_state

        return combined_data
    
    #input list of features to load 
    def load_and_preprocess_feature_dataframe(self, animal):
        '''Build dataframe given list of features for one animal'''
        feature_dataframe = {}
        for feature in self.feature_list:
            feature_dir = os.path.join(self.data_directory + f"{feature}")
            feature_dataframe[f"{feature}"] = self.load_and_preprocess_feature(animal, f"{feature}/", feature_dir)
    
        return pd.DataFrame(feature_dataframe)

    def load_and_preprocess_feature(self, animal, feature_name, feature_dir):
        feature_data = np.load(os.path.join(feature_dir, f"{animal}_{feature_name}.npy"))

        #remove 
        #_indices = np.load(os.path.join(self.data_directory, f"{animal}_error_br_1.npy"))
        #if len(error_indices) > 0:
        #    feature_data = np.delete(feature_data, error_indices)

        return feature_data

    def prepare_datafile(self, ids_ls):
        feature_df = []

        for animal in ids_ls:
            print(animal)
            animal_data = self.load_and_preprocess_data(animal)

            if animal_data is not None:
                feature_df.append(animal_data)

        concat_df = pd.concat(feature_df)
        return concat_df

# Define your directory paths
#data_directory = '/home/melissa/RESULTS/XGBoost/SYNGAP1'
#second_br_directory = '/path/to/second/br/directory'  # Specify the directory for the second br file
#SYNGAP_het = [...]  # List of SYNGAP_het animals
#SYNGAP_wt = [...]   # List of SYNGAP_wt animals
#ids_1_ls = [...]     # List of animal IDs

# Create an instance of the DataPreparation class with the second br directory
#data_preparer = DataPreparation(data_directory, SYNGAP_het, SYNGAP_wt, second_br_directory)

# Call the main function to prepare the DataFrame
#result_df = data_preparer.prepare_df_one_datafile(ids_1_ls)
