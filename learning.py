#This is where a machine learning algorithm is going to be applied
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
import mglearn

import functions as f   #Where my own functions are stored


#Opening files extracted from feature_extraction.py

data = pd.read_csv("/Users/joaobinenbojm/desktop/audio_ml_challenge/Novoic-Audio-Challenge/data.csv")
out= pd.read_csv("/Users/joaobinenbojm/desktop/audio_ml_challenge/Novoic-Audio-Challenge/out.csv")

print(data.columns)

#Converting data into numpy form and deleting non-necessary

X = data.to_numpy()
X = np.delete(X,[0],1)

y = out.to_numpy()
y = np.ravel(np.delete(y,[0],1))

#Splitting input and output into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
     X, y, random_state=0)

#I will be applying a linear support vector machine algorithm for Multiclass binary classification
#All this will be done based on the six features being experimented

from sklearn.svm import LinearSVC
linear_svm = LinearSVC(C =0.00001).fit(X_train, y_train)
print("Coefficient shape: ", linear_svm.coef_.shape)
print("Intercept shape: ", linear_svm.intercept_.shape)

#Computing accuracy of classifications
print("Training set score: {:.3f}".format(linear_svm.score(X_train, y_train)))
print("Test set score: {:.3f}".format(linear_svm.score(X_test, y_test)))


#Simple binary classification between words

#2000 samples of each class, compare each class to each other for simple binary classification

acc_test = 0 #Used to calculate the mean test accuracy
acc_train = 0 # Used to calculate the mean training accuracy
track = 0
words = ['down','go','left','no','off','on','right','up','stop','yes']

for i in range(0,10,1):     # number of combinations of possible out of 10 classes
    for j in range(i,10,1):
        if(i == j):
              continue
        data1 = X[ 2000*i:(2000*(i+1)) -1,:]
        data2  = X[ 2000*j:(2000*(j+1)) -1,:]
        data_merge = np.append(data1,data2,0)

        out1 = y[2000 * i:(2000 * (i + 1)) - 1]
        out2 = y[2000 * j:(2000 * (j + 1)) - 1]
        out_merge = np.append(out1,out2,0)

        print("Classification for: ",words[i]," and",words[j])
        train,test = f.get_accuracy(data_merge,out_merge)
        acc_test = acc_test + float(test)
        acc_train = acc_train + float(train)
        track +=1

print("Mean training score: ",acc_train/track)
print("Mean test score: ",acc_test/track)

#Plotting top 3 by differentiating throughout all different features to see why they stood out:

#Plotting between 'no' and 'up'
no = X[ 6000:7999,:]
up = X[ 14000:15999,:]
no_up = np.append(no,up,0)

no_out = y[6000:7999]
up_out = y[14000:15999]
no_up_out = np.append(no_out,up_out,0)

no_up_df = pd.DataFrame(no_up, columns=('F1(Hz)','F2(Hz)', 'F3(Hz)','F_mean(Hz)','F2-F1(Hz)',
                                                'Word Duration(ms)','Relative Max Amplitude'))
grr = pd.plotting.scatter_matrix(no_up_df, c=no_up_out, figsize=(48, 48), marker='o',
hist_kwds={'bins': 20}, s=60, alpha=.8)
plt.show()

#Plotting between 'yes' and 'up'
yes = X[ 18000:19999,:]
up = X[ 14000:15999,:]
yes_up = np.append(yes,up,0)

yes_out = y[18000:19999]
up_out = y[14000:15999]
yes_up_out = np.append(yes_out,up_out,0)

yes_up_df = pd.DataFrame(yes_up, columns=('F1(Hz)','F2(Hz)', 'F3(Hz)','F_mean(Hz)','F2-F1(Hz)',
                                                'Word Duration(ms)','Relative Max Amplitude'))
grr2 = pd.plotting.scatter_matrix(yes_up_df, c=yes_up_out, figsize=(48, 48), marker='o',
hist_kwds={'bins': 20}, s=60, alpha=.8)
plt.show()

#Plotting between 'np' and 'off'
no = X[ 6000:7999,:]
off = X[ 14000:15999,:]
no_off = np.append(no,off,0)

no_out = y[6000:7999]
off_out = y[14000:15999]
no_off_out = np.append(no_out,off_out,0)

no_off_df = pd.DataFrame(no_off, columns=('F1(Hz)','F2(Hz)', 'F3(Hz)','F_mean(Hz)','F2-F1(Hz)',
                                                'Word Duration(ms)','Relative Max Amplitude'))
grr3 = pd.plotting.scatter_matrix(no_off_df, c=no_off_out, figsize=(48, 48), marker='o',
hist_kwds={'bins': 20}, s=60, alpha=.8)
plt.show()