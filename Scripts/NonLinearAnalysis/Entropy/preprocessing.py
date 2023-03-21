import os 
import sys
import numpy as np 
import pandas as pd 
from scipy import signal

sys.path.insert(0, '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Scripts/Preprocessing')
from filter import NoiseFilter


class EntropyFilter(NoiseFilter):
    '''Class inherits from filtering class in Preprocessing folder but changes lower and upper
    bound of IIR filter for entropy analysis to avoid power-line artefacts'''
    def __init__(self, unfiltered_data, brain_state_file, channelvariable, ch_type):
        super.__init__(unfiltered_data, brain_state_file, channelvariable, ch_type)
        self.order = 3
        self.low = 0.2/125.2
        self.high = 48/125.2