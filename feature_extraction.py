#Where the features will be extracted from the data (time consuming so it's better to do this first, and not every time)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wave
import scipy
import math
from scipy.signal import lfilter
import librosa.core as lib
import scipy.io as spio
import os
import sklearn

import functions as f   #Where my own functions are stored

dir = "/Users/joaobinenbojm/google-cloud-sdk/data/"     #Where original audio files were originally stored
X,y = f.file_walking(dir)
X = X[6:-1].reshape(20000,7)

#Saving different data sections in different pandas dataframes
data = pd.DataFrame(X, columns=('F1(Hz)','F2(Hz)', 'F3(Hz)','F_mean(Hz)','F2-F1(Hz)',
                                                'Word Duration(ms)','Relative Max Amplitude'))
data.to_csv(r"\Users\joaobinenbojm\desktop\audio_ml_challenge\Novoic-Audio-Challenge\data.csv")

out = pd.DataFrame(y)
out.to_csv(r"\Users\joaobinenbojm\desktop\audio_ml_challenge\Novoic-Audio-Challenge\out.csv")

