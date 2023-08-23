import os 
import pandas as pd 
import numpy as np 

results_path = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Phase_Lock_Val/'
br_directory = '/home/melissa/PREPROCESSING/SYNGAP1/numpyformat_baseline/'
 
ids = [ 'S7088', 'S7092', 'S7094' , 'S7098', 'S7068', 'S7074', 'S7076', 'S7071', 'S7075',
      'S7070','S7072','S7083', 'S7063','S7064','S7069', 'S7086', 'S7091', 'S7101', 'S7096']
    
SYNGAP_1_ID_ls = [ 'S7088', 'S7092', 'S7094' , 'S7098', 'S7068', 'S7074', 'S7076', 'S7071', 'S7075', 'S7101']
SYNGAP_2_ID_ls = ['S7070','S7072','S7083', 'S7063','S7064','S7069', 'S7086','S7091', 'S7096']

SYNGAP_het = ['S7063', 'S7064', 'S7069', 'S7072', 'S7075', 'S7076', 'S7088', 'S7092', 'S7094', 'S7096']
SYNGAP_wt = ['S7068', 'S7070', 'S7071', 'S7074', 'S7083', 'S7086', 'S7091', 'S7098', 'S7101']

train_2_ids = ['S7070', 'S7072', 'S7083', 'S7064', 'S7096']#, 'S7069', 'S7091']
train_1_ids = ['S7088', 'S7092', 'S7094', 'S7075', 'S7071'] #, 'S7076', 'S7101']
val_2_ids = ['S7069', 'S7091']
val_1_ids = ['S7076', 'S7101']
test_2_ids = ['S7086'] # 'S7063'
test_1_ids = ['S7076'] #'S7074'


def prepare_df_one_datafile(ids_1_ls, br_directory, SYNGAP_het, SYNGAP_wt):
    motor_hurst_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Motor/Hurst/'
    motor_hfd_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1//Motor/HFD/'
    motor_dispen_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Motor/DispEn/'
    motor_gamma_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Motor/Gamma_Power/'
    motor_theta_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Motor/Theta_Power/'

    soma_hurst_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/Hurst/'
    soma_hfd_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/HFD/'
    soma_dispen_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/DispEn/'
    soma_gamma_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/Gamma_Power/'
    soma_theta_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/Theta_Power/'

    vis_hurst_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/Hurst/'
    vis_hfd_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/HFD/'
    vis_dispen_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/DispEn/'
    vis_gamma_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/Gamma_Power/'
    vis_theta_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/Theta_Power/'

    #connectivity indices
    cross_cor_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/CrossCorr/'
    mot_cross_cor_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Motor/CrossCorr_Motor/' 
    som_cross_cor_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/CrossCorr_Somatosensory/'
    vis_cross_cor_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/CrossCorr_Visual/'

    mot_phase_lock_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Motor/Phase_Lock_Motor/' 
    som_phase_lock_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/Phase_Lock_Somato/'
    vis_phase_lock_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/Phase_Lock_Visual/'
    
    feature_df_1_ids = []
    for animal in ids_1_ls:
        print(animal)
        #load br file 
        br_1 = pd.read_pickle(br_directory + str(animal) + '_BL1.pkl')
        br_state = br_1['brainstate'].to_numpy()

        #motor 
        motor_hfd = np.load(motor_hfd_dir + animal + '_hfd_concat.npy')
        motor_hfd_avg = [value[0] for value in motor_hfd]
        motor_hurst = np.load(motor_hurst_dir + animal + '_hurst_concat.npy')
        motor_hurst_avg = [value[0] for value in motor_hurst]

        motor_dispen = np.load(motor_dispen_dir + animal + '_dispen.npy')
        motor_gamma = np.load(motor_gamma_dir + animal + '_power.npy') 
        motor_theta = np.load(motor_theta_dir + animal + '_power.npy') 

        #somatosensory 
        soma_hfd = np.load(soma_hfd_dir + animal + '_hfd_concat.npy')
        soma_hfd_avg = [value[0] for value in soma_hfd]
        soma_hurst = np.load(soma_hurst_dir + animal + '_hurst_concat.npy')
        soma_hurst_avg = [value[0] for value in soma_hurst]

        soma_dispen = np.load(soma_dispen_dir + animal + '_dispen.npy')
        soma_gamma = np.load(soma_gamma_dir + animal + '_power.npy') 
        soma_theta = np.load(soma_theta_dir + animal + '_power.npy') 

        #somatosensory 
        vis_hfd = np.load(vis_hfd_dir + animal + '_hfd_concat.npy')
        vis_hfd_avg = [value[0] for value in vis_hfd]
        vis_hurst = np.load(vis_hurst_dir + animal + '_hurst_concat.npy')
        vis_hurst_avg = [value[0] for value in vis_hurst]

        vis_dispen = np.load(vis_dispen_dir + animal + '_dispen.npy')
        vis_gamma = np.load(vis_gamma_dir + animal + '_power.npy') 
        vis_theta = np.load(vis_theta_dir + animal + '_power.npy') 

        #cross cor
        mot_cross_corr_left = np.load(mot_cross_cor_dir + str(animal) + '_mot_left_cross_cor.npy')
        mot_cross_corr_right = np.load(mot_cross_cor_dir + str(animal) + '_mot_right_cross_cor.npy')
        som_cross_corr_left = np.load(som_cross_cor_dir + str(animal) + '_som_left_cross_cor.npy')
        som_cross_corr_right = np.load(som_cross_cor_dir + str(animal) + '_som_right_cross_cor.npy')
        vis_cross_corr_left = np.load(vis_cross_cor_dir + str(animal) + '_vis_left_cross_cor.npy')
        vis_cross_corr_right = np.load(vis_cross_cor_dir + str(animal) + '_vis_right_cross_cor.npy')

        #phase lock 
        mot_phase_lock_left = np.load(mot_phase_lock_dir + str(animal) + '_mot_left_phase_lock.npy')
        mot_phase_lock_right = np.load(mot_phase_lock_dir + str(animal) + '_mot_right_phase_lock.npy')
        som_phase_lock_left = np.load(som_phase_lock_dir + str(animal) + '_som_left_phase_lock.npy')
        som_phase_lock_right = np.load(som_phase_lock_dir + str(animal) + '_som_right_phase_lock.npy')
        vis_phase_lock_left = np.load(vis_phase_lock_dir + str(animal) + '_vis_left_phase_lock.npy')
        vis_phase_lock_right = np.load(vis_phase_lock_dir + str(animal) + '_vis_right_phase_lock.npy')

        #cross_corr_errors
        error_1 = np.load(cross_cor_dir + animal + '_error_br_1.npy')


        if len(error_1) > 0:
            br_state = np.delete(br_state, error_1)
            motor_hfd_avg = np.delete(motor_hfd_avg, error_1)
            motor_hurst_avg = np.delete(motor_hurst_avg, error_1)
            motor_dispen = np.delete(motor_dispen, error_1)
            motor_gamma = np.delete(motor_gamma, error_1)
            motor_theta = np.delete(motor_theta, error_1)
            soma_hfd_avg = np.delete(soma_hfd_avg, error_1)
            soma_hurst_avg = np.delete(soma_hurst_avg, error_1)
            soma_dispen = np.delete(soma_dispen, error_1)
            soma_gamma = np.delete(soma_gamma, error_1)
            soma_theta = np.delete(soma_theta, error_1)
            vis_hfd_avg = np.delete(vis_hfd_avg, error_1)
            vis_hurst_avg = np.delete(vis_hurst_avg, error_1)
            vis_dispen = np.delete(vis_dispen, error_1)
            vis_gamma = np.delete(vis_gamma, error_1)
            vis_theta = np.delete(vis_theta, error_1)
            mot_phase_lock_left = np.delete(mot_phase_lock_left, error_1)
            mot_phase_lock_right = np.delete(mot_phase_lock_right, error_1)
            som_phase_lock_left = np.delete(som_phase_lock_left, error_1)
            som_phase_lock_right = np.delete(som_phase_lock_right, error_1)
            vis_phase_lock_left = np.delete(vis_phase_lock_left, error_1)
            vis_phase_lock_right = np.delete(vis_phase_lock_right, error_1)
        else:
            pass
        
        
        #print(len(br_state))
        #print(len(motor_hfd_avg))
        #print(len(motor_hurst_avg))
        #print(len(motor_dispen))
        #print(len(motor_gamma))
        #print(len(soma_hfd_avg))
        #print(len(soma_hurst_avg))
        #print(len(soma_dispen))
        #print(len(soma_gamma))
        #print(len(vis_phase_lock_right))

         #clean arrays
        #clean_offset = np.delete(fooof_offset_nan, nan_indices)
        #clean_exponent = np.delete(fooof_exponent_nan, nan_indices)
        #clean_dispen = np.delete(dispen, nan_indices)
        #clean_gamma = np.delete(gamma, nan_indices)
        #clean_br_state = np.delete(br_state, nan_indices)


        if animal in SYNGAP_het:
            genotype = 1
            print(animal + ' ' + str(genotype))
        elif animal in SYNGAP_wt:
            genotype = 0
            print(animal + ' ' + str(genotype))
        else:
            print(animal + ' not in either list')


        region_dict = {'Genotype': [genotype]*len(motor_dispen), 
                       'Animal_ID': [animal]*len(motor_gamma),
                       'SleepStage': br_state,
                       'Motor_DispEn': motor_dispen, 'Motor_Hurst': motor_hurst_avg, 
                       'Motor_HFD': motor_hfd_avg,
                       'Motor_Gamma': motor_gamma, 'Motor_Theta': motor_theta,
                       #'Soma_DispEn': soma_dispen,
                       'Soma_Hurst': soma_hurst_avg,
                       'Soma_HFD': soma_hfd_avg,
                       'Soma_Gamma': soma_gamma,'Soma_Theta': soma_theta,
                       'Visual_DispEn': vis_dispen,
                       #'Visual_Hurst': vis_hurst_avg,
                       #'Visual_HFD': vis_hfd_avg, 
                       'Vis_Gamma': vis_gamma, 
                       'Vis_Theta': vis_theta,
                       'Mot_CC_Right': mot_cross_corr_right,
                       #'Mot_CC_Left': mot_cross_corr_left,
                       'Som_CC_Right': som_cross_corr_right, 'Som_CC_Left': som_cross_corr_left,
                       'Vis_CC_Right': vis_cross_corr_right, 'Mot_CC_Left': vis_cross_corr_left,
                       'Mot_PL_Right': mot_phase_lock_right, 'Mot_PL_Left': mot_phase_lock_left,
                       'Som_PL_Right': som_phase_lock_right, 'Som_PL_Left': som_phase_lock_left,
                       'Vis_PL_Right': vis_phase_lock_right} 
                       #'Vis_PL_Left': vis_phase_lock_left}


        region_df = pd.DataFrame(data = region_dict)
        clean_df = region_df[region_df["SleepStage"].isin([0,1,2])]
        feature_df_1_ids.append(clean_df)

    concat_df = pd.concat(feature_df_1_ids)
    return concat_df



def prepare_df_two_datafile(ids_2_ls, br_directory, SYNGAP_het, SYNGAP_wt):
    motor_hurst_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Motor/Hurst/'
    motor_hfd_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1//Motor/HFD/'
    motor_dispen_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Motor/DispEn/'
    motor_gamma_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Motor/Gamma_Power/'
    motor_theta_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Motor/Theta_Power/'

    soma_hurst_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/Hurst/'
    soma_hfd_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/HFD/'
    soma_dispen_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/DispEn/'
    soma_gamma_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/Gamma_Power/'
    soma_theta_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/Theta_Power/'

    vis_hurst_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/Hurst/'
    vis_hfd_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/HFD/'
    vis_dispen_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/DispEn/'
    vis_gamma_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/Gamma_Power/'
    vis_theta_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/Theta_Power/'

    #connectivity indices
    cross_cor_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/CrossCorr/'
    mot_cross_cor_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Motor/CrossCorr_Motor/' 
    som_cross_cor_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/CrossCorr_Somatosensory/'
    vis_cross_cor_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/CrossCorr_Visual/'

    mot_phase_lock_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Motor/Phase_Lock_Motor/' 
    som_phase_lock_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/Phase_Lock_Somato/'
    vis_phase_lock_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/Phase_Lock_Visual/'
    
    feature_df_2_ids = []
    for animal in ids_2_ls:
        print(animal)
        #load br file 
        br_1 = pd.read_pickle(br_directory + str(animal) + '_BL1.pkl')
        br_2 = pd.read_pickle(br_directory + str(animal) + '_BL2.pkl')
        br_state_1 = br_1['brainstate'].to_numpy()
        br_state_2 = br_2['brainstate'].to_numpy()
        br_state = np.concatenate([br_state_1, br_state_2])

        #motor 
        motor_hfd = np.load(motor_hfd_dir + animal + '_hfd_concat.npy')
        motor_hfd_avg = [value[0] for value in motor_hfd]
        motor_hurst = np.load(motor_hurst_dir + animal + '_hurst_concat.npy')
        motor_hurst_avg = [value[0] for value in motor_hurst]

        motor_dispen = np.load(motor_dispen_dir + animal + '_dispen.npy')
        motor_gamma = np.load(motor_gamma_dir + animal + '_power.npy') 
        motor_theta = np.load(motor_theta_dir + animal + '_power.npy') 

        #somatosensory 
        soma_hfd = np.load(soma_hfd_dir + animal + '_hfd_concat.npy')
        soma_hfd_avg = [value[0] for value in soma_hfd]
        soma_hurst = np.load(soma_hurst_dir + animal + '_hurst_concat.npy')
        soma_hurst_avg = [value[0] for value in soma_hurst]
    
        soma_dispen = np.load(soma_dispen_dir + animal + '_dispen.npy')
        soma_gamma = np.load(soma_gamma_dir + animal + '_power.npy') 
        soma_theta = np.load(soma_theta_dir + animal + '_power.npy') 
    
        #somatosensory 
        vis_hfd = np.load(vis_hfd_dir + animal + '_hfd_concat.npy')
        vis_hfd_avg = [value[0] for value in vis_hfd]
        vis_hurst = np.load(vis_hurst_dir + animal + '_hurst_concat.npy')
        vis_hurst_avg = [value[0] for value in vis_hurst]
    
        vis_dispen = np.load(vis_dispen_dir + animal + '_dispen.npy')
        vis_gamma = np.load(vis_gamma_dir + animal + '_power.npy') 
        vis_theta = np.load(vis_theta_dir + animal + '_power.npy')
    
        #cross cor
        mot_cross_corr_left = np.load(mot_cross_cor_dir + str(animal) + '_mot_left_cross_cor.npy')
        mot_cross_corr_right = np.load(mot_cross_cor_dir + str(animal) + '_mot_right_cross_cor.npy')
        som_cross_corr_left = np.load(som_cross_cor_dir + str(animal) + '_som_left_cross_cor.npy')
        som_cross_corr_right = np.load(som_cross_cor_dir + str(animal) + '_som_right_cross_cor.npy')
        vis_cross_corr_left = np.load(vis_cross_cor_dir + str(animal) + '_vis_left_cross_cor.npy')
        vis_cross_corr_right = np.load(vis_cross_cor_dir + str(animal) + '_vis_right_cross_cor.npy')
    
        #phase lock 
        mot_phase_lock_left = np.load(mot_phase_lock_dir + str(animal) + '_mot_left_phase_lock.npy')
        mot_phase_lock_right = np.load(mot_phase_lock_dir + str(animal) + '_mot_right_phase_lock.npy')
        som_phase_lock_left = np.load(som_phase_lock_dir + str(animal) + '_som_left_phase_lock.npy')
        som_phase_lock_right = np.load(som_phase_lock_dir + str(animal) + '_som_right_phase_lock.npy')
        vis_phase_lock_left = np.load(vis_phase_lock_dir + str(animal) + '_vis_left_phase_lock.npy')
        vis_phase_lock_right = np.load(vis_phase_lock_dir + str(animal) + '_vis_right_phase_lock.npy')
        
        #cross_corr_errors
        error_1 = np.load(cross_cor_dir + animal + '_error_br_1.npy')
        error_2 = np.load(cross_cor_dir + animal + '_error_br_2.npy')
        
        
        if len(error_1) > 0 and len(error_2) >0:
            print(animal + ' error')
            error_2_correct = error_2 + 17280
            errors = np.concatenate([error_1, error_2_correct])
            br_state = np.delete(br_state, errors)
            motor_hfd_avg = np.delete(motor_hfd_avg, errors)
            motor_hurst_avg = np.delete(motor_hurst_avg, errors)
            motor_dispen = np.delete(motor_dispen, errors)
            motor_gamma = np.delete(motor_gamma, errors)
            motor_theta = np.delete(motor_theta, errors)
            soma_hfd_avg = np.delete(soma_hfd_avg, errors)
            soma_hurst_avg = np.delete(soma_hurst_avg, errors)
            soma_dispen = np.delete(soma_dispen, errors)
            soma_gamma = np.delete(soma_gamma, errors)
            soma_theta = np.delete(soma_theta, errors)
            vis_hfd_avg = np.delete(vis_hfd_avg, errors)
            vis_hurst_avg = np.delete(vis_hurst_avg, errors)
            vis_dispen = np.delete(vis_dispen, errors)
            vis_gamma = np.delete(vis_gamma, errors)
            vis_theta = np.delete(vis_theta, errors)
            mot_phase_lock_left = np.delete(mot_phase_lock_left, errors)
            mot_phase_lock_right = np.delete(mot_phase_lock_right, errors)
            som_phase_lock_left = np.delete(som_phase_lock_left, errors)
            som_phase_lock_right = np.delete(som_phase_lock_right, errors)
            vis_phase_lock_left = np.delete(vis_phase_lock_left, errors)
            vis_phase_lock_right = np.delete(vis_phase_lock_right, errors)
            
        elif len(error_1) > 0:
            br_state = np.delete(br_state, error_1)
            motor_hfd_avg = np.delete(motor_hfd_avg, error_1)
            motor_hurst_avg = np.delete(motor_hurst_avg, error_1)
            motor_dispen = np.delete(motor_dispen, error_1)
            motor_gamma = np.delete(motor_gamma, error_1)
            motor_theta = np.delete(motor_theta, error_1)
            soma_hfd_avg = np.delete(soma_hfd_avg, error_1)
            soma_hurst_avg = np.delete(soma_hurst_avg, error_1)
            soma_dispen = np.delete(soma_dispen, error_1)
            soma_gamma = np.delete(soma_gamma, error_1)
            soma_theta = np.delete(soma_theta, error_1)
            vis_hfd_avg = np.delete(vis_hfd_avg, error_1)
            vis_hurst_avg = np.delete(vis_hurst_avg, error_1)
            vis_dispen = np.delete(vis_dispen, error_1)
            vis_gamma = np.delete(vis_gamma, error_1)
            vis_theta = np.delete(vis_theta, error_1)
            mot_phase_lock_left = np.delete(mot_phase_lock_left, error_1)
            mot_phase_lock_right = np.delete(mot_phase_lock_right, error_1)
            som_phase_lock_left = np.delete(som_phase_lock_left, error_1)
            som_phase_lock_right = np.delete(som_phase_lock_right, error_1)
            vis_phase_lock_left = np.delete(vis_phase_lock_left, error_1)
            vis_phase_lock_right = np.delete(vis_phase_lock_right, error_1)
        elif len(error_2) > 0:
            print(animal + ' error 2')
            error_2_br_2 = error_2 + 17280
            br_state = np.delete(br_state, error_2_br_2)
            motor_hfd_avg = np.delete(motor_hfd_avg, error_2_br_2)
            motor_hurst_avg = np.delete(motor_hurst_avg, error_2_br_2)
            motor_dispen = np.delete(motor_dispen, error_2_br_2)
            motor_gamma = np.delete(motor_gamma, error_2_br_2)
            motor_theta = np.delete(motor_theta, error_2_br_2)
            soma_hfd_avg = np.delete(soma_hfd_avg, error_2_br_2)
            soma_hurst_avg = np.delete(soma_hurst_avg, error_2_br_2)
            soma_dispen = np.delete(soma_dispen, error_2_br_2)
            soma_gamma = np.delete(soma_gamma, error_2_br_2)
            soma_theta = np.delete(soma_theta, error_2_br_2)
            vis_hfd_avg = np.delete(vis_hfd_avg, error_2_br_2)
            vis_hurst_avg = np.delete(vis_hurst_avg, error_2_br_2)
            vis_dispen = np.delete(vis_dispen, error_2_br_2)
            vis_gamma = np.delete(vis_gamma, error_2_br_2)
            vis_theta = np.delete(vis_theta, error_2_br_2)
            mot_phase_lock_left = np.delete(mot_phase_lock_left, error_2_br_2)
            mot_phase_lock_right = np.delete(mot_phase_lock_right, error_2_br_2)
            som_phase_lock_left = np.delete(som_phase_lock_left, error_2_br_2)
            som_phase_lock_right = np.delete(som_phase_lock_right, error_2_br_2)
            vis_phase_lock_left = np.delete(vis_phase_lock_left, error_2_br_2)
            vis_phase_lock_right = np.delete(vis_phase_lock_right, error_2_br_2)
        else:
            pass
        
        
        
         #clean arrays
        #clean_offset = np.delete(fooof_offset_nan, nan_indices)
        #clean_exponent = np.delete(fooof_exponent_nan, nan_indices)
        #clean_dispen = np.delete(dispen, nan_indices)
        #clean_gamma = np.delete(gamma, nan_indices)
        #clean_br_state = np.delete(br_state, nan_indices)
        
        
        if animal in SYNGAP_het:
                genotype = 1
                print(animal + ' ' + str(genotype))
        elif animal in SYNGAP_wt:
                genotype = 0
                print(animal + ' ' + str(genotype))
        else:
                print(animal + ' not in either list')
    
        
        region_dict = {'Genotype': [genotype]*len(motor_gamma), 
                       'Animal_ID': [animal]*len(motor_gamma),
                       'SleepStage': br_state,
                       'Motor_DispEn': motor_dispen, 
                       'Motor_Hurst': motor_hurst_avg, 
                       'Motor_HFD': motor_hfd_avg,
                       'Motor_Gamma': motor_gamma,
                       'Motor_Theta': motor_theta,
                       #'Soma_DispEn': soma_dispen,
                       'Soma_Hurst': soma_hurst_avg,
                       'Soma_HFD': soma_hfd_avg,
                       'Soma_Gamma': soma_gamma,'Soma_Theta': soma_theta,
                       'Visual_DispEn': vis_dispen,
                       #'Visual_Hurst': vis_hurst_avg,'Visual_HFD': vis_hfd_avg,
                       'Vis_Gamma': vis_gamma, 
                       'Vis_Theta': vis_theta,
                       'Mot_CC_Right': mot_cross_corr_right, 
                       #'Mot_CC_Left': mot_cross_corr_left,
                       'Som_CC_Right': som_cross_corr_right, 'Som_CC_Left': som_cross_corr_left,
                       'Vis_CC_Right': vis_cross_corr_right, 'Mot_CC_Left': vis_cross_corr_left,
                       'Mot_PL_Right': mot_phase_lock_right, 'Mot_PL_Left': mot_phase_lock_left,
                       'Som_PL_Right': som_phase_lock_right, 'Som_PL_Left': som_phase_lock_left,
                       'Vis_PL_Right': vis_phase_lock_right}
                       #'Vis_PL_Left': vis_phase_lock_left}
        
        
        region_df = pd.DataFrame(data = region_dict)
        clean_df = region_df[region_df["SleepStage"].isin([0,1,2])]
        feature_df_2_ids.append(clean_df)
    
    df_concat = pd.concat(feature_df_2_ids)
    
    return df_concat



def prepare_df_one_no_feature_selection(ids_1_ls, br_directory, SYNGAP_het, SYNGAP_wt):
    motor_hurst_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Motor/Hurst/'
    motor_hfd_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1//Motor/HFD/'
    motor_dispen_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Motor/DispEn/'
    motor_delta_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Motor/Delta_Power/'
    motor_theta_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Motor/Theta_Power/'
    motor_sigma_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Motor/Sigma_Power/'
    motor_beta_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Motor/Beta_Power/'
    motor_gamma_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Motor/Gamma_Power/'

    soma_hurst_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/Hurst/'
    soma_hfd_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/HFD/'
    soma_dispen_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/DispEn/'
    soma_delta_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/Delta_Power/'
    soma_theta_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/Theta_Power/'
    soma_sigma_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/Sigma_Power/'
    soma_beta_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/Beta_Power/'
    soma_gamma_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/Gamma_Power/'

    vis_hurst_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/Hurst/'
    vis_hfd_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/HFD/'
    vis_dispen_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/DispEn/'
    vis_delta_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/Delta_Power/'
    vis_theta_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/Theta_Power/'
    vis_sigma_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/Sigma_Power/'
    vis_beta_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/Beta_Power/'
    vis_gamma_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/Gamma_Power/'
    
    #fooof
    motor_fooof_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Motor/Motor_FOOOF/'
    soma_fooof_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/Somatosensory_FOOOF/'
    vis_fooof_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/Visual_FOOOF/'

    #connectivity indices
    cross_cor_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/CrossCorr/'
    mot_cross_cor_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Motor/CrossCorr_Motor/' 
    som_cross_cor_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/CrossCorr_Somatosensory/'
    vis_cross_cor_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/CrossCorr_Visual/'

    mot_phase_lock_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Motor/Phase_Lock_Motor/' 
    som_phase_lock_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/Phase_Lock_Somato/'
    vis_phase_lock_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/Phase_Lock_Visual/'
    
    feature_df_1_ids = []
    for animal in ids_1_ls:
        print(animal)
        #load br file 
        br_1 = pd.read_pickle(br_directory + str(animal) + '_BL1.pkl')
        br_state = br_1['brainstate'].to_numpy()

        #motor 
        motor_hfd = np.load(motor_hfd_dir + animal + '_hfd_concat.npy')
        motor_hfd_avg = [value[0] for value in motor_hfd]
        motor_hurst = np.load(motor_hurst_dir + animal + '_hurst_concat.npy')
        motor_hurst_avg = [value[0] for value in motor_hurst]

        motor_dispen = np.load(motor_dispen_dir + animal + '_dispen.npy')
        motor_delta = np.load(motor_delta_dir + animal + '_power.npy')
        motor_theta = np.load(motor_theta_dir + animal + '_power.npy') 
        motor_sigma = np.load(motor_sigma_dir + animal + '_power.npy')
        motor_beta = np.load(motor_beta_dir + animal + '_power.npy')
        motor_gamma = np.load(motor_gamma_dir + animal + '_power.npy') 
        
        #motor fooof
        motor_offset_right = np.load(motor_fooof_dir + animal + '_motor_offset_right.npy')
        motor_offset_left = np.load(motor_fooof_dir + animal + '_motor_offset_left.npy')
        motor_exponent_right = np.load(motor_fooof_dir + animal + '_motor_exponent_right.npy')
        motor_exponent_left = np.load(motor_fooof_dir + animal + '_motor_exponent_left.npy')
        motor_offset = np.mean([motor_offset_left, motor_offset_right])
        motor_exponent = np.mean([motor_exponent_left, motor_exponent_right])
        

        #somatosensory 
        soma_hfd = np.load(soma_hfd_dir + animal + '_hfd_concat.npy')
        soma_hfd_avg = [value[0] for value in soma_hfd]
        soma_hurst = np.load(soma_hurst_dir + animal + '_hurst_concat.npy')
        soma_hurst_avg = [value[0] for value in soma_hurst]

        soma_dispen = np.load(soma_dispen_dir + animal + '_dispen.npy')
        soma_delta = np.load(soma_delta_dir + animal + '_power.npy') 
        soma_theta = np.load(soma_theta_dir + animal + '_power.npy') 
        soma_sigma = np.load(soma_sigma_dir + animal + '_power.npy') 
        soma_beta = np.load(soma_beta_dir + animal + '_power.npy')
        soma_gamma = np.load(soma_gamma_dir + animal + '_power.npy') 
        
        #somatosensory fooof
        soma_offset_right = np.load(soma_fooof_dir + animal + '_somato_offset_right.npy')
        soma_offset_left = np.load(soma_fooof_dir + animal + '_somato_offset_left.npy')
        soma_exponent_right = np.load(soma_fooof_dir + animal + '_somato_exponent_right.npy')
        soma_exponent_left = np.load(soma_fooof_dir + animal + '_somato_exponent_left.npy')
        soma_offset = np.mean([soma_offset_left, soma_offset_right])
        soma_exponent = np.mean([soma_exponent_left, soma_exponent_right])

        #vis 
        vis_hfd = np.load(vis_hfd_dir + animal + '_hfd_concat.npy')
        vis_hfd_avg = [value[0] for value in vis_hfd]
        vis_hurst = np.load(vis_hurst_dir + animal + '_hurst_concat.npy')
        vis_hurst_avg = [value[0] for value in vis_hurst]

        vis_dispen = np.load(vis_dispen_dir + animal + '_dispen.npy')
        vis_delta = np.load(vis_delta_dir + animal + '_power.npy')
        vis_theta = np.load(vis_theta_dir + animal + '_power.npy') 
        vis_sigma = np.mean(vis_sigma_dir + animal + '_power.npy')
        vis_beta = np.load(vis_gamma_dir + animal + '_power.npy')
        vis_gamma = np.load(vis_gamma_dir + animal + '_power.npy') 
        
        #vis fooof
        vis_offset_right = np.load(vis_fooof_dir + animal + '_vis_offset_right.npy')
        vis_offset_left = np.load(vis_fooof_dir + animal + '_vis_offset_left.npy')
        vis_exponent_right = np.load(vis_fooof_dir + animal + '_vis_exponent_right.npy')
        vis_exponent_left = np.load(vis_fooof_dir + animal + '_vis_exponent_left.npy')
        vis_offset = np.mean([vis_offset_left, vis_offset_right])
        vis_exponent = np.mean([vis_exponent_left, vis_exponent_right])
        

        #cross cor
        mot_cross_corr_left = np.load(mot_cross_cor_dir + str(animal) + '_mot_left_cross_cor.npy')
        mot_cross_corr_right = np.load(mot_cross_cor_dir + str(animal) + '_mot_right_cross_cor.npy')
        som_cross_corr_left = np.load(som_cross_cor_dir + str(animal) + '_som_left_cross_cor.npy')
        som_cross_corr_right = np.load(som_cross_cor_dir + str(animal) + '_som_right_cross_cor.npy')
        vis_cross_corr_left = np.load(vis_cross_cor_dir + str(animal) + '_vis_left_cross_cor.npy')
        vis_cross_corr_right = np.load(vis_cross_cor_dir + str(animal) + '_vis_right_cross_cor.npy')

        #phase lock 
        mot_phase_lock_left = np.load(mot_phase_lock_dir + str(animal) + '_mot_left_phase_lock.npy')
        mot_phase_lock_right = np.load(mot_phase_lock_dir + str(animal) + '_mot_right_phase_lock.npy')
        som_phase_lock_left = np.load(som_phase_lock_dir + str(animal) + '_som_left_phase_lock.npy')
        som_phase_lock_right = np.load(som_phase_lock_dir + str(animal) + '_som_right_phase_lock.npy')
        vis_phase_lock_left = np.load(vis_phase_lock_dir + str(animal) + '_vis_left_phase_lock.npy')
        vis_phase_lock_right = np.load(vis_phase_lock_dir + str(animal) + '_vis_right_phase_lock.npy')

        #cross_corr_errors
        error_1 = np.load(cross_cor_dir + animal + '_error_br_1.npy')


        if len(error_1) > 0:
            br_state = np.delete(br_state, error_1)
            motor_hfd_avg = np.delete(motor_hfd_avg, error_1)
            motor_hurst_avg = np.delete(motor_hurst_avg, error_1)
            motor_dispen = np.delete(motor_dispen, error_1)
            motor_delta = np.delete(motor_delta, error_1)
            motor_theta = np.delete(motor_theta, error_1)
            motor_sigma = np.delete(motor_sigma, error_1)
            motor_beta = np.delete(motor_beta, error_1)
            motor_gamma = np.delete(motor_gamma, error_1)
            soma_hfd_avg = np.delete(soma_hfd_avg, error_1)
            soma_hurst_avg = np.delete(soma_hurst_avg, error_1)
            soma_dispen = np.delete(soma_dispen, error_1)
            soma_delta = np.delete(soma_delta, error_1)
            soma_theta = np.delete(soma_theta, error_1)
            soma_sigma = np.delete(soma_sigma, error_1)
            soma_beta = np.delete(soma_beta, error_1) 
            soma_gamma = np.delete(soma_gamma, error_1)
            vis_hfd_avg = np.delete(vis_hfd_avg, error_1)
            vis_hurst_avg = np.delete(vis_hurst_avg, error_1)
            vis_dispen = np.delete(vis_dispen, error_1)
            vis_delta = np.delete(vis_delta, error_1)
            vis_theta = np.delete(vis_theta, error_1)
            vis_sigma = np.delete(vis_sigma, error_1)
            vis_beta = np.delete(vis_beta, error_1)
            vis_gamma = np.delete(vis_gamma, error_1)
            mot_phase_lock_left = np.delete(mot_phase_lock_left, error_1)
            mot_phase_lock_right = np.delete(mot_phase_lock_right, error_1)
            som_phase_lock_left = np.delete(som_phase_lock_left, error_1)
            som_phase_lock_right = np.delete(som_phase_lock_right, error_1)
            vis_phase_lock_left = np.delete(vis_phase_lock_left, error_1)
            vis_phase_lock_right = np.delete(vis_phase_lock_right, error_1)
            motor_offset = np.delete(motor_offset, error_1)
            motor_exponent = np.delete(motor_exponent, error_1)
            soma_offset = np.delete(soma_offset, error_1)
            soma_exponent = np.delete(soma_exponent, error_1)
            vis_offset = np.delete(vis_offset, error_1)
            vis_exponent = np.delete(vis_exponent, error_1)
        else:
            pass
        
        
        #print(len(br_state))
        #print(len(motor_hfd_avg))
        #print(len(motor_hurst_avg))
        #print(len(motor_dispen))
        #print(len(motor_gamma))
        #print(len(soma_hfd_avg))
        #print(len(soma_hurst_avg))
        #print(len(soma_dispen))
        #print(len(soma_gamma))
        #print(len(vis_phase_lock_right))

         #clean arrays
        #clean_offset = np.delete(fooof_offset_nan, nan_indices)
        #clean_exponent = np.delete(fooof_exponent_nan, nan_indices)
        #clean_dispen = np.delete(dispen, nan_indices)
        #clean_gamma = np.delete(gamma, nan_indices)
        #clean_br_state = np.delete(br_state, nan_indices)


        if animal in SYNGAP_het:
            genotype = 1
            print(animal + ' ' + str(genotype))
        elif animal in SYNGAP_wt:
            genotype = 0
            print(animal + ' ' + str(genotype))
        else:
            print(animal + ' not in either list')


        region_dict = {'Genotype': [genotype]*len(motor_dispen), 
                       'Animal_ID': [animal]*len(motor_dispen),
                       'SleepStage': br_state,
                       'Motor_DispEn': motor_dispen, 
                       'Motor_Hurst': motor_hurst_avg, 
                       'Motor_HFD': motor_hfd_avg,
                       'Motor_Delta': motor_delta,
                       'Motor_Theta': motor_theta,
                       'Motor_Sigma': motor_sigma,
                       'Motor_Beta': motor_beta,
                       'Motor_Gamma': motor_gamma,
                       'Soma_DispEn': soma_dispen,
                       'Soma_Hurst': soma_hurst_avg,
                       'Soma_HFD': soma_hfd_avg,
                       'Soma_Delta': soma_delta,
                       'Soma_Theta': soma_theta,
                       'Soma_Sigma': soma_sigma,
                       'Soma_Beta': soma_beta,
                       'Soma_Gamma': soma_gamma,
                       'Visual_DispEn': vis_dispen,
                       'Visual_Hurst': vis_hurst_avg,
                       'Visual_HFD': vis_hfd_avg, 
                       'Vis_Delta': vis_delta,
                       'Vis_Theta': vis_theta,
                       'Vis_Sigma': vis_sigma,
                       'Vis_Beta': vis_beta,
                       'Vis_Gamma': vis_gamma, 
                       'Motor_Offset': motor_offset,
                       'Motor_Exponent': motor_exponent,
                       'Soma_Offset': soma_offset,
                       'Soma_Exponent': soma_exponent,
                       'Vis_Offset': vis_offset,
                       'Vis_Exponent': vis_exponent,
                       'Mot_CC_Right': mot_cross_corr_right, 'Mot_CC_Left': mot_cross_corr_left,
                       'Som_CC_Right': som_cross_corr_right, 'Som_CC_Left': som_cross_corr_left,
                       'Vis_CC_Right': vis_cross_corr_right, 'Mot_CC_Left': vis_cross_corr_left,
                       'Mot_PL_Right': mot_phase_lock_right, 'Mot_PL_Left': mot_phase_lock_left,
                       'Som_PL_Right': som_phase_lock_right, 'Som_PL_Left': som_phase_lock_left,
                       'Vis_PL_Right': vis_phase_lock_right, 'Vis_PL_Left': vis_phase_lock_left}


        region_df = pd.DataFrame(data = region_dict)
        clean_df = region_df[region_df["SleepStage"].isin([0,1,2])]
        feature_df_1_ids.append(clean_df)

    concat_df = pd.concat(feature_df_1_ids)
    return concat_df



def prepare_df_two_no_feature_selection(ids_2_ls, br_directory, SYNGAP_het, SYNGAP_wt):
    motor_hurst_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Motor/Hurst/'
    motor_hfd_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1//Motor/HFD/'
    motor_dispen_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Motor/DispEn/'
    motor_delta_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Motor/Delta_Power/'
    motor_theta_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Motor/Theta_Power/'
    motor_sigma_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Motor/Sigma_Power/'
    motor_beta_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Motor/Beta_Power/'
    motor_gamma_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Motor/Gamma_Power/'

    soma_hurst_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/Hurst/'
    soma_hfd_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/HFD/'
    soma_dispen_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/DispEn/'
    soma_delta_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/Delta_Power/'
    soma_theta_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/Theta_Power/'
    soma_sigma_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/Sigma_Power/'
    soma_beta_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/Beta_Power/'
    soma_gamma_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/Gamma_Power/'

    vis_hurst_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/Hurst/'
    vis_hfd_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/HFD/'
    vis_dispen_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/DispEn/'
    vis_delta_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/Delta_Power/'
    vis_theta_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/Theta_Power/'
    vis_sigma_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/Sigma_Power/'
    vis_beta_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/Beta_Power/'
    vis_gamma_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/Gamma_Power/'
    
    #fooof
    motor_fooof_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Motor/Motor_FOOOF/'
    soma_fooof_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/Somatosensory_FOOOF/'
    vis_fooof_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/Visual_FOOOF/'

    #connectivity indices
    cross_cor_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/CrossCorr/'
    mot_cross_cor_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Motor/CrossCorr_Motor/' 
    som_cross_cor_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/CrossCorr_Somatosensory/'
    vis_cross_cor_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/CrossCorr_Visual/'

    mot_phase_lock_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Motor/Phase_Lock_Motor/' 
    som_phase_lock_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Somatosensory/Phase_Lock_Somato/'
    vis_phase_lock_dir = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/Phase_Lock_Visual/'
    
    feature_df_2_ids = []
    for animal in ids_2_ls:
        print(animal)
        #load br file 
        br_1 = pd.read_pickle(br_directory + str(animal) + '_BL1.pkl')
        br_2 = pd.read_pickle(br_directory + str(animal) + '_BL2.pkl')
        br_state_1 = br_1['brainstate'].to_numpy()
        br_state_2 = br_2['brainstate'].to_numpy()
        br_state = np.concatenate([br_state_1, br_state_2])

         #motor 
        motor_hfd = np.load(motor_hfd_dir + animal + '_hfd_concat.npy')
        motor_hfd_avg = [value[0] for value in motor_hfd]
        motor_hurst = np.load(motor_hurst_dir + animal + '_hurst_concat.npy')
        motor_hurst_avg = [value[0] for value in motor_hurst]

        motor_dispen = np.load(motor_dispen_dir + animal + '_dispen.npy')
        motor_delta = np.load(motor_delta_dir + animal + '_power.npy') 
        motor_theta = np.load(motor_theta_dir + animal + '_power.npy') 
        motor_sigma = np.load(motor_sigma_dir + animal + '_power.npy') 
        motor_beta = np.load(motor_beta_dir + animal + '_power.npy') 
        motor_gamma = np.load(motor_gamma_dir + animal + '_power.npy') 
        
        #motor fooof
        motor_offset_right = np.load(motor_fooof_dir + animal + '_motor_offset_right.npy')
        motor_offset_left = np.load(motor_fooof_dir + animal + '_motor_offset_left.npy')
        motor_exponent_right = np.load(motor_fooof_dir + animal + '_motor_exponent_right.npy')
        motor_exponent_left = np.load(motor_fooof_dir + animal + '_motor_exponent_left.npy')
        motor_offset = np.mean([motor_offset_left, motor_offset_right])
        motor_exponent = np.mean([motor_exponent_left, motor_exponent_right])
        
        #somatosensory 
        soma_hfd = np.load(soma_hfd_dir + animal + '_hfd_concat.npy')
        soma_hfd_avg = [value[0] for value in soma_hfd]
        soma_hurst = np.load(soma_hurst_dir + animal + '_hurst_concat.npy')
        soma_hurst_avg = [value[0] for value in soma_hurst]

        soma_dispen = np.load(soma_dispen_dir + animal + '_dispen.npy')
        soma_delta = np.load(soma_delta_dir + animal + '_power.npy') 
        soma_theta = np.load(soma_theta_dir + animal + '_power.npy') 
        soma_sigma = np.load(soma_sigma_dir + animal + '_power.npy') 
        soma_beta = np.load(soma_beta_dir + animal + '_power.npy') 
        soma_gamma = np.load(soma_gamma_dir + animal + '_power.npy') 
        
        #somatosensory fooof
        soma_offset_right = np.load(soma_fooof_dir + animal + '_somato_offset_right.npy')
        soma_offset_left = np.load(soma_fooof_dir + animal + '_somato_offset_left.npy')
        soma_exponent_right = np.load(soma_fooof_dir + animal + '_somato_exponent_right.npy')
        soma_exponent_left = np.load(soma_fooof_dir + animal + '_somato_exponent_left.npy')
        soma_offset = np.mean([soma_offset_left, soma_offset_right])
        soma_exponent = np.mean([soma_exponent_left, soma_exponent_right])

        #vis 
        vis_hfd = np.load(vis_hfd_dir + animal + '_hfd_concat.npy')
        vis_hfd_avg = [value[0] for value in vis_hfd]
        vis_hurst = np.load(vis_hurst_dir + animal + '_hurst_concat.npy')
        vis_hurst_avg = [value[0] for value in vis_hurst]

        vis_dispen = np.load(vis_dispen_dir + animal + '_dispen.npy')
        vis_delta = np.load(vis_delta_dir + animal + '_power.npy') 
        vis_theta = np.load(vis_theta_dir + animal + '_power.npy') 
        vis_sigma = np.load(vis_sigma_dir + animal + '_power.npy') 
        vis_beta = np.load(vis_beta_dir + animal + '_power.npy') 
        vis_gamma = np.load(vis_gamma_dir + animal + '_power.npy') 
        
        #vis fooof
        vis_offset_right = np.load(vis_fooof_dir + animal + '_vis_offset_right.npy')
        vis_offset_left = np.load(vis_fooof_dir + animal + '_vis_offset_left.npy')
        vis_exponent_right = np.load(vis_fooof_dir + animal + '_vis_exponent_right.npy')
        vis_exponent_left = np.load(vis_fooof_dir + animal + '_vis_exponent_left.npy')
        vis_offset = np.mean([vis_offset_left, vis_offset_right])
        vis_exponent = np.mean([vis_exponent_left, vis_exponent_right])
        

        #cross cor
        mot_cross_corr_left = np.load(mot_cross_cor_dir + str(animal) + '_mot_left_cross_cor.npy')
        mot_cross_corr_right = np.load(mot_cross_cor_dir + str(animal) + '_mot_right_cross_cor.npy')
        som_cross_corr_left = np.load(som_cross_cor_dir + str(animal) + '_som_left_cross_cor.npy')
        som_cross_corr_right = np.load(som_cross_cor_dir + str(animal) + '_som_right_cross_cor.npy')
        vis_cross_corr_left = np.load(vis_cross_cor_dir + str(animal) + '_vis_left_cross_cor.npy')
        vis_cross_corr_right = np.load(vis_cross_cor_dir + str(animal) + '_vis_right_cross_cor.npy')

        #phase lock 
        mot_phase_lock_left = np.load(mot_phase_lock_dir + str(animal) + '_mot_left_phase_lock.npy')
        mot_phase_lock_right = np.load(mot_phase_lock_dir + str(animal) + '_mot_right_phase_lock.npy')
        som_phase_lock_left = np.load(som_phase_lock_dir + str(animal) + '_som_left_phase_lock.npy')
        som_phase_lock_right = np.load(som_phase_lock_dir + str(animal) + '_som_right_phase_lock.npy')
        vis_phase_lock_left = np.load(vis_phase_lock_dir + str(animal) + '_vis_left_phase_lock.npy')
        vis_phase_lock_right = np.load(vis_phase_lock_dir + str(animal) + '_vis_right_phase_lock.npy')

        #cross_corr_errors
        error_1 = np.load(cross_cor_dir + animal + '_error_br_1.npy')
        error_2 = np.load(cross_cor_dir + animal + '_error_br_2.npy')
        
       
        
        if len(error_1) > 0 and len(error_2) >0:
            print(animal + ' error')
            error_2_correct = error_2 + 17280
            errors = np.concatenate([error_1, error_2_correct])
            br_state = np.delete(br_state, errors)
            motor_hfd_avg = np.delete(motor_hfd_avg, errors)
            motor_hurst_avg = np.delete(motor_hurst_avg, errors)
            motor_dispen = np.delete(motor_dispen, errors)
            motor_delta = np.delete(motor_delta, errors)
            motor_theta = np.delete(motor_theta, errors)
            motor_sigma = np.delete(motor_sigma, errors)
            motor_beta = np.delete(motor_beta, errors)
            motor_gamma = np.delete(motor_gamma, errors)
            soma_hfd_avg = np.delete(soma_hfd_avg, errors)
            soma_hurst_avg = np.delete(soma_hurst_avg, errors)
            soma_dispen = np.delete(soma_dispen, errors)
            soma_delta = np.delete(soma_delta, errors)
            soma_theta = np.delete(soma_theta, errors)
            soma_sigma = np.delete(soma_sigma, errors)
            soma_beta = np.delete(soma_beta, errors) 
            soma_gamma = np.delete(soma_gamma, errors)
            vis_hfd_avg = np.delete(vis_hfd_avg, errors)
            vis_hurst_avg = np.delete(vis_hurst_avg, errors)
            vis_dispen = np.delete(vis_dispen, errors)
            vis_delta = np.delete(vis_delta, errors)
            vis_theta = np.delete(vis_theta, errors)
            vis_sigma = np.delete(vis_sigma, errors)
            vis_beta = np.delete(vis_beta, errors)
            vis_gamma = np.delete(vis_gamma, errors)
            mot_phase_lock_left = np.delete(mot_phase_lock_left, errors)
            mot_phase_lock_right = np.delete(mot_phase_lock_right, errors)
            som_phase_lock_left = np.delete(som_phase_lock_left, errors)
            som_phase_lock_right = np.delete(som_phase_lock_right, errors)
            vis_phase_lock_left = np.delete(vis_phase_lock_left, errors)
            vis_phase_lock_right = np.delete(vis_phase_lock_right, errors)
            motor_offset = np.delete(motor_offset, errors)
            motor_exponent = np.delete(motor_exponent, errors)
            soma_offset = np.delete(soma_offset, errors)
            soma_exponent = np.delete(soma_exponent, errors)
            vis_offset = np.delete(vis_offset, errors)
            vis_exponent = np.delete(vis_exponent, errors)
            
        elif len(error_1) > 0:
            br_state = np.delete(br_state, error_1)
            motor_hfd_avg = np.delete(motor_hfd_avg, error_1)
            motor_hurst_avg = np.delete(motor_hurst_avg, error_1)
            motor_dispen = np.delete(motor_dispen, error_1)
            motor_delta = np.delete(motor_delta, error_1)
            motor_theta = np.delete(motor_theta, error_1)
            motor_sigma = np.delete(motor_sigma, error_1)
            motor_beta = np.delete(motor_beta, error_1)
            motor_gamma = np.delete(motor_gamma, error_1)
            soma_hfd_avg = np.delete(soma_hfd_avg, error_1)
            soma_hurst_avg = np.delete(soma_hurst_avg, error_1)
            soma_dispen = np.delete(soma_dispen, error_1)
            soma_delta = np.delete(soma_delta, error_1)
            soma_theta = np.delete(soma_theta, error_1)
            soma_sigma = np.delete(soma_sigma, error_1)
            soma_beta = np.delete(soma_beta, error_1) 
            soma_gamma = np.delete(soma_gamma, error_1)
            vis_hfd_avg = np.delete(vis_hfd_avg, error_1)
            vis_hurst_avg = np.delete(vis_hurst_avg, error_1)
            vis_dispen = np.delete(vis_dispen, error_1)
            vis_delta = np.delete(vis_delta, error_1)
            vis_theta = np.delete(vis_theta, error_1)
            vis_sigma = np.delete(vis_sigma, error_1)
            vis_beta = np.delete(vis_beta, error_1)
            vis_gamma = np.delete(vis_gamma, error_1)
            mot_phase_lock_left = np.delete(mot_phase_lock_left, error_1)
            mot_phase_lock_right = np.delete(mot_phase_lock_right, error_1)
            som_phase_lock_left = np.delete(som_phase_lock_left, error_1)
            som_phase_lock_right = np.delete(som_phase_lock_right, error_1)
            vis_phase_lock_left = np.delete(vis_phase_lock_left, error_1)
            vis_phase_lock_right = np.delete(vis_phase_lock_right, error_1)
            motor_offset = np.delete(motor_offset, error_1)
            motor_exponent = np.delete(motor_exponent, error_1)
            soma_offset = np.delete(soma_offset, error_1)
            soma_exponent = np.delete(soma_exponent, error_1)
            vis_offset = np.delete(vis_offset, error_1)
            vis_exponent = np.delete(vis_exponent, error_1)
            
        elif len(error_2) > 0:
            print(animal + ' error 2')
            error_2_br_2 = error_2 + 17280
            br_state = np.delete(br_state, error_2_br_2)
            motor_hfd_avg = np.delete(motor_hfd_avg, error_2_br_2)
            motor_hurst_avg = np.delete(motor_hurst_avg, error_2_br_2)
            motor_dispen = np.delete(motor_dispen, error_2_br_2)
            motor_delta = np.delete(motor_delta, error_2_br_2)
            motor_theta = np.delete(motor_theta, error_2_br_2)
            motor_sigma = np.delete(motor_sigma, error_2_br_2)
            motor_beta = np.delete(motor_beta, error_2_br_2)
            motor_gamma = np.delete(motor_gamma, error_2_br_2)
            soma_hfd_avg = np.delete(soma_hfd_avg, error_2_br_2)
            soma_hurst_avg = np.delete(soma_hurst_avg, error_2_br_2)
            soma_dispen = np.delete(soma_dispen, error_2_br_2)
            soma_delta = np.delete(soma_delta, error_2_br_2)
            soma_theta = np.delete(soma_theta, error_2_br_2)
            soma_sigma = np.delete(soma_sigma, error_2_br_2)
            soma_beta = np.delete(soma_beta, error_2_br_2) 
            soma_gamma = np.delete(soma_gamma, error_2_br_2)
            vis_hfd_avg = np.delete(vis_hfd_avg, error_2_br_2)
            vis_hurst_avg = np.delete(vis_hurst_avg, error_2_br_2)
            vis_dispen = np.delete(vis_dispen, error_2_br_2)
            vis_delta = np.delete(vis_delta, error_2_br_2)
            vis_theta = np.delete(vis_theta, error_2_br_2)
            vis_sigma = np.delete(vis_sigma, error_2_br_2)
            vis_beta = np.delete(vis_beta, error_2_br_2)
            vis_gamma = np.delete(vis_gamma, error_2_br_2)
            mot_phase_lock_left = np.delete(mot_phase_lock_left, error_2_br_2)
            mot_phase_lock_right = np.delete(mot_phase_lock_right, error_2_br_2)
            som_phase_lock_left = np.delete(som_phase_lock_left, error_2_br_2)
            som_phase_lock_right = np.delete(som_phase_lock_right, error_2_br_2)
            vis_phase_lock_left = np.delete(vis_phase_lock_left, error_2_br_2)
            vis_phase_lock_right = np.delete(vis_phase_lock_right, error_2_br_2)
            motor_offset = np.delete(motor_offset, error_2_br_2)
            motor_exponent = np.delete(motor_exponent, error_2_br_2)
            soma_offset = np.delete(soma_offset, error_2_br_2)
            soma_exponent = np.delete(soma_exponent, error_2_br_2)
            vis_offset = np.delete(vis_offset, error_2_br_2)
            vis_exponent = np.delete(vis_exponent, error_2_br_2)
        else:
            pass
        
        
        
         #clean arrays
        #clean_offset = np.delete(fooof_offset_nan, nan_indices)
        #clean_exponent = np.delete(fooof_exponent_nan, nan_indices)
        #clean_dispen = np.delete(dispen, nan_indices)
        #clean_gamma = np.delete(gamma, nan_indices)
        #clean_br_state = np.delete(br_state, nan_indices)
        
        
        if animal in SYNGAP_het:
                genotype = 1
                print(animal + ' ' + str(genotype))
        elif animal in SYNGAP_wt:
                genotype = 0
                print(animal + ' ' + str(genotype))
        else:
                print(animal + ' not in either list')
    
        
        region_dict = {'Genotype': [genotype]*len(motor_gamma), 
                       'Animal_ID': [animal]*len(motor_gamma),
                       'SleepStage': br_state,
                       'Motor_DispEn': motor_dispen,
                       'Motor_Hurst': motor_hurst_avg, 
                       'Motor_HFD': motor_hfd_avg,
                       'Motor_Delta': motor_delta,
                       'Motor_Theta': motor_theta,
                       'Motor_Sigma': motor_sigma,
                       'Motor_Beta': motor_beta,
                       'Motor_Gamma': motor_gamma,
                       'Soma_DispEn': soma_dispen,
                       'Soma_Hurst': soma_hurst_avg,
                       'Soma_HFD': soma_hfd_avg,
                       'Soma_Delta': soma_delta,
                       'Soma_Theta': soma_theta,
                       'Soma_Sigma': soma_sigma,
                       'Soma_Beta': soma_beta,
                       'Soma_Gamma': soma_gamma,
                       'Visual_DispEn': vis_dispen,
                       'Visual_Hurst': vis_hurst_avg,
                       'Visual_HFD': vis_hfd_avg,
                       'Vis_Delta': vis_delta,
                       'Vis_Theta': vis_theta,
                       'Vis_Sigma': vis_sigma,
                       'Vis_Beta': vis_beta,
                       'Vis_Gamma': vis_gamma, 
                       'Motor_Offset': motor_offset,
                       'Motor_Exponent': motor_exponent,
                       'Soma_Offset': soma_offset,
                       'Soma_Exponent': soma_exponent,
                       'Vis_Offset': vis_offset,
                       'Vis_Exponent': vis_exponent,
                       'Mot_CC_Right': mot_cross_corr_right, 'Mot_CC_Left': mot_cross_corr_left,
                       'Som_CC_Right': som_cross_corr_right, 'Som_CC_Left': som_cross_corr_left,
                       'Vis_CC_Right': vis_cross_corr_right, 'Mot_CC_Left': vis_cross_corr_left,
                       'Mot_PL_Right': mot_phase_lock_right, 'Mot_PL_Left': mot_phase_lock_left,
                       'Som_PL_Right': som_phase_lock_right, 'Som_PL_Left': som_phase_lock_left,
                       'Vis_PL_Right': vis_phase_lock_right, 'Vis_PL_Left': vis_phase_lock_left}
        
        
        region_df = pd.DataFrame(data = region_dict)
        clean_df = region_df[region_df["SleepStage"].isin([0,1,2])]
        feature_df_2_ids.append(clean_df)
    
    df_concat = pd.concat(feature_df_2_ids)
    
    return df_concat