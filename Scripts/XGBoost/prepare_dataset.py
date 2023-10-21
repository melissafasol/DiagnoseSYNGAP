import os
import numpy as np
import pandas as pd

class DataPreparation:
    def __init__(self, data_directory, SYNGAP_het, SYNGAP_wt, second_br_list):
        self.data_directory = data_directory
        self.SYNGAP_het = SYNGAP_het
        self.SYNGAP_wt = SYNGAP_wt
        self.second_br_list = second_br_list

    def load_and_preprocess_data(self, animal):
        
        if animal not in self.second_br_list:
            br_1 = pd.read_csv(os.path.join(self.data_directory, f"{animal}_BL1.csv"))
            br_state = br_1['brainstate'].to_numpy()
        
            # Load and preprocess data for different regions
            motor_data = self.load_and_preprocess_region_data(animal, "Motor")
            soma_data = self.load_and_preprocess_region_data(animal, "Somatosensory")
            vis_data = self.load_and_preprocess_region_data(animal, "Visual")

            # Merge data from different regions
            combined_data = pd.concat([motor_data, soma_data, vis_data], axis=1)

            # Add genotype information
            genotype = 1 if animal in self.SYNGAP_het else 0 if animal in self.SYNGAP_wt else None
            combined_data['Genotype'] = genotype
            
            return combined_data

        if animal in self.second_br_list:
            br_1 = pd.read_csv(os.path.join(self.data_directory, f"{animal}_BL1.csv"))
            br_state_1 = br_1['brainstate'].to_numpy()
            br_2 = pd.read_csv(os.path.join(self.data_directory, f"{animal}_BL2.csv"))
            br_state_2 = br_2['brainstate'].to_numpy()
            br_state = np.concatenate([br_state_1, br_state_2])
            
            # Load and preprocess data for different regions
            motor_data = self.load_and_preprocess_region_data(animal, "Motor")
            soma_data = self.load_and_preprocess_region_data(animal, "Somatosensory")
            vis_data = self.load_and_preprocess_region_data(animal, "Visual")

            # Merge data from different regions
            combined_data = pd.concat([motor_data, soma_data, vis_data], axis=1)

            # Add genotype information
            genotype = 1 if animal in self.SYNGAP_het else 0 if animal in self.SYNGAP_wt else None
            combined_data['Genotype'] = genotype

            return combined_data

    def load_and_preprocess_region_data(self, animal, region):
        region_data = {}
        region_dir = os.path.join(self.data_directory, region)

        # Load specific data for the region
        region_data["HFD"] = self.load_and_preprocess_feature(animal, f"{region}/HFD", region_dir)
        region_data["DispEn"] = self.load_and_preprocess_feature(animal, f"{region}/DispEn", region_dir)
        region_data["Gamma"] = self.load_and_preprocess_feature(animal, f"{region}/Gamma_Power", region_dir)
        region_data["Theta"] = self.load_and_preprocess_feature(animal, f"{region}/Theta_Power", region_dir)
        region_data["CrossCorr"] = self.load_and_preprocess_feature(animal, f"{region}/CrossCorr", region_dir)
        region_data["PhaseLock"] = self.load_and_preprocess_feature(animal, f"{region}/Phase_Lock_{region}", region_dir)

        return pd.DataFrame(region_data)

    def load_and_preprocess_feature(self, animal, feature_name, feature_dir, second_br_list):
        feature_data = np.load(os.path.join(feature_dir, f"{animal}_{feature_name}.npy"))

        #remove 
        #error_indices = np.load(os.path.join(self.data_directory, f"{animal}_error_br_1.npy"))
        #if len(error_indices) > 0:
        #    feature_data = np.delete(feature_data, error_indices)

        return feature_data

    def prepare_df_one_datafile(self, ids_ls):
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
