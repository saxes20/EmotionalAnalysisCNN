#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 19:17:48 2018

@author: sameer
"""

import glob
import librosa
import librosa.display
import numpy as np
import random
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle
import matplotlib.pyplot as plt

#RUN IN TERMINAL:
#    import glob
#    import os
#    files = glob.glob('/Users/sameer/Desktop/PJAS/voices/Actor_02/*')
#    for file in files:
#        result = file[:len(file)-4] + '.wav'
#        command = 'ffmpeg -i ' + file + ' ' + result
#        os.system(command)

precedeString = '/Users/sameer/Desktop/PJAS/voices'

datafp = precedeString + '/Actors'
myData = glob.glob(datafp + '/*')
random.shuffle(myData)
emotions = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

finalData = []
data = []
tags = []

for d in myData: #remove all non-audio files
    parts = d.split("/")[-1].split(".wav")[0].split("-")
    #print(parts)
    if parts[0] != '02':
        finalData.append(d)
        data.append(parts)
        tags.append(emotions[int(parts[2])-1])

def convert_To_Index (s):
    return int(emotions.index(s))

train_setX = finalData[:int(len(finalData)*.75)]
train_setY = tags[:int(len(finalData)*.75)]
train_setY = list(map(convert_To_Index, train_setY))
val_setX = finalData[-int(len(finalData)*.25):]
val_setY = tags[-int(len(finalData)*.25):]
val_setY = list(map(convert_To_Index, val_setY))


def extract_features(file_name):
    try:
      X, sample_rate = librosa.load(file_name) 
      # we features from data
      stft = np.abs(librosa.stft(X))
      mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
      chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
      mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
      contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
      tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sample_rate).T,axis=0)
    except Exception as e:
      print("Error encountered while parsing file: ", file_name)
      return None, None
    return mfccs, chroma, mel, contrast, tonnetz

#Training Set
features = np.empty((0, 193)) 
labels = np.empty(0)
print('Parsing training set')
X = []
y = []
for f in range(len(train_setX)):
    print(train_setX[f])
    print(f)
    mfccs, chroma, mel, contrast, tonnetz = extract_features(train_setX[f])
    ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
    features = np.vstack([features,ext_features])
    labels = np.append(labels,train_setY[f])

X = np.array(features)
y = np.array(labels, dtype = np.int)

#Validation Set
featuresVal = np.empty((0,193))
labelsVal = np.empty(0)
print('Parsing validation set')
valX = []
valy = []
for f in range(len(val_setX)):
    print(val_setX[f])
    print(f)
    mfccs, chroma, mel, contrast, tonnetz = extract_features(val_setX[f])
    ext_featuresVal = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
    featuresVal = np.vstack([featuresVal, ext_featuresVal])
    labelsVal = np.append(labelsVal, val_setY[f])
    
valX = np.array(featuresVal)
valy = np.array(labelsVal, dtype = np.int)

model = svm.SVC(kernel='poly', C=1, gamma=1)
model.fit(X,y)

cross_validation = model.predict(valX)
print(accuracy_score(valy, cross_validation))

saveFileName = 'emotion_audio_polysvmModel.sav'
pickle.dump(model, open(saveFileName, 'wb'))

##model_json = model.to_json()
##model
##saveFileName = 'emotion_audio_svmModel.sav'
##import pickle
##model
##pickle.dump(model, open(saveFileName, 'wb'))
##cross_validation = model.predict(valX)
##cross_validation
##valy
##accuracy_score(valy, cross_validation)
##from sklearn.metrics import accuracy_score
##accuracy_score(valy, cross_validation) - output 90%
##prediction = emotions[model.predict(valX[0])[0]]