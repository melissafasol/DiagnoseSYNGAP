import os 
import numpy as np 
import pandas as pd 


class NoiseFilter():
    
    def __init__(self, unfiltered_data, number_of_epochs, noise_limit):
        '''input unfiltered data and the number of epochs data needs to be split into
        - this should be equivalent to the length of a brainstate file'''
        self.unfiltered_data = unfiltered_data
        self.number_of_epochs = number_of_epochs
        self.noise_limit = noise_limit
        
    def find_packetloss_indices(self, reshaped_data):
        def packet_loss(epoch):
            mask = epoch.max() < self.noise_limit
            return mask 
        
        def get_dataset(reshaped_data):
            packet_loss_score = []
            for epoch in reshaped_data:
                packet_loss_score.append(0) if packet_loss(epoch) == True else packet_loss_score.append(6)
                return packet_loss_score
    
        split_epochs = np.split(self.unfiltered_data, self.number_of_epochs, axis = 1)  #split raw data into epochs 
        packet_loss_score = get_dataset(split_epochs) #find indices where value in epoch exceeds 3000mV
        noise_indices = []
        for idx, i in enumerate(packet_loss_score):
            if i == 6:
                noise_indices.append(idx)
                
        return noise_indices
        
        