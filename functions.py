import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wave
import scipy
import math
from scipy.signal import lfilter
import librosa.core as lib
import scipy.io as spio
import os, os.path

#Functions List:

#Code used online to estimate formants
#Formant estimation as adapted from a MATLAB script
#https://uk.mathworks.com/help/signal/ug/formant-estimation-with-lpc-coefficients.html
def get_formants(file_path):

    # # Read from file.
    # spf = wave.open(file_path, 'r') # http://www.linguistics.ucla.edu/people/hayes/103/Charts/VChart/ae.wav
    #
    # # Get file as numpy array.
    # x = spf.readframes(-1)
    # x = np.fromstring(x, 'Int16')

    from scipy.io import wavfile
    fs, data = wavfile.read(file_path)

    x = np.asarray(data)

    try:
       x = x[:, 0]
    except:
        pass

    # Get Hamming window.
    N = len(x)
    w = np.hamming(N)

    # Apply window and high pass filter.
    x1 = x * w
    x1 = lfilter([1], [1., 0.63], x1)

    # Get LPC.
    ncoeff = int(fs/1000) + 2
    A = lib.lpc(x1, ncoeff)

    # Get roots.
    rts = np.roots(A)
    rts = [r for r in rts if np.imag(r) >= 0]


    # Get angles.
    angz = np.arctan2(np.imag(rts), np.real(rts))

    # Get frequencies.
    frqs = np.dot(angz, fs / (2 * np.pi))

    l = int(len(frqs))
    indices = np.zeros(l)
    bw = indices
    frqs2 = np.copy(frqs)

    for i in range(0, l, 1):    #Determining indices of the maximum and their bandwidth
        if(frqs2[i] == 0):
            frqs2[i] = -30*i
        ind = np.where(frqs2 == np.amax(frqs2))
        try:
            indices[l - 1 - i] = int(ind[0])
            l2 = indices[l - 1 - i]
            frqs2[int(l2)] = -i - 1  # keeping all indices different to prevent different dimensions of tuples
            # Adding bandwidth
            bw[l - 1 - i] = -0.5 * (fs / (2 * np.pi)) * np.log(np.abs(rts[int(ind[0])]))

        except:
            indices[l - 1 - i] = int(ind[0][0])
            l2 = indices[l - 1 - i]
            frqs2[int(l2)] = -i - 1 #keeping all indices different to prevent different dimensions of tuples
            #Adding bandwidth
            bw[l - 1 - i] = -0.5 * (fs / (2 * np.pi)) * np.log(np.abs(rts[int(ind[0][0])]))

    frqs = sorted(frqs)

    frqs = np.asarray(frqs)
    formants = np.copy(frqs)


    for kk in range(0,l,1):
        if ((frqs[kk] < 90) or (bw[kk] > 400)):
            formants[kk] = -1   #TO BE IGNORED

    for i in range(l-1,-1,-1):  #Removing non-formant elements
        if(int(formants[i]) == -1):
            formants = np.delete(formants,i)


    return formants[0:3]

#Function to detect word duration in miliseconds: assumes consonants are short relative to vowels
#Also that takes the difference between the normalised peak(1) and the normalised mean
#This function should hypothetically help identify plosives (also normalised by duration)
def f_extract2(filename):

    from scipy.io import wavfile
    fs, data = wavfile.read(filename)

    data_signal = np.asarray(data)

    try:
       data_signal = data_signal[:, 0]
    except:
        pass
    l = len(list(data_signal))
    peak = np.max(np.abs(data_signal))
    #Calculating desired feature
    diff = 1 - (np.mean(data_signal))/(peak)

    for i in range(0,l-1,1): #removing elements before the word
        if(abs(data_signal[i]) > 0.2*peak): #thresholding step
            start = i
            break

    for i in range(l-1,0,-1):  #removing elements after the word
        if (abs(data_signal[i]) > 0.2 * peak):  # thresholding step
            fin = i
            break

    duration = (fin - start)/fs

    return (1000*duration), (diff/duration)      #returns duration in ms and other measure desired


#Function to read all audio files, extract and return relevant information
def file_walking(init_dir):     #Argument is the path to the directory containing the various folders
    folder_names = ['down','go','left','no','off','on','right','up','stop','yes']

    f_num = 7   #seven features to be used with training algorithm

    #Iterating over all folders
    matrix_hold = np.zeros(f_num)
    y = -1  #to correspond to no element
    word_count = 0

    for word in folder_names:
        count = 0   #For simple testing purposes
        y = np.append(y,np.ones(2000)*word_count) #saves the target output
        word_count +=1  #move onto next word
        for dirpath, dirnames,filenames, in os.walk(init_dir + word):
            for f_name in filenames:
                x = np.zeros(f_num) #creating empty row to hold feature values for current row
                x[0:3] = get_formants(init_dir + word + '/' + f_name)     #three formants
                x[3] = np.mean(x[0:2])               #mean of first three formants (almost like a measure of pitch)
                x[4] = x[1] - x[2]      #Difference between first and second formant in frequency
                x[5],x[6] = f_extract2(init_dir + word + '/' + f_name)

                try:
                    matrix_hold = np.append(matrix_hold, np.transpose(x), axis = 0)
                except:
                    matrix_hold = np.append(matrix_hold, x, axis=0)
                count+=1

                #print(count)
                #print(matrix_hold.shape)
                if(count == 2000):
                     break


    return matrix_hold, y  #returns calculated X vector


#Will take some form of data and desired output accompanying it, apply the Linear SVM classfication and return accuracies
def get_accuracy(X,y):

    #Splitting data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0)

    #Determining data accuracy
    from sklearn.svm import LinearSVC
    linear_svm = LinearSVC(C=0.0001).fit(X_train, y_train)

    print("Training set score: {:.3f}".format(linear_svm.score(X_train, y_train)))
    print("Test set score: {:.3f}".format(linear_svm.score(X_test, y_test)))
    return format(linear_svm.score(X_train, y_train)),format(linear_svm.score(X_test, y_test))

