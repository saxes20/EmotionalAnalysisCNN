#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 15:13:01 2018

@author: sameer
"""

import glob
from shutil import copyfile

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
subjects = glob.glob('source_emotion/Emotion/*') #Check

fileCount = 0
totalCount = 0

def sortByKey (s):
    return int(s[-12:].split('.png')[0])

for x in subjects:
    
    partNum = x[-4:]
    sessions = glob.glob('%s/*' % x)
    
    for session in sessions:
        
        files = glob.glob('%s/*' % session)
        current_session = session[-3:]
        
        totalCount += 1
        if len(files) != 0:
            fileCount += 1
        
        for file in files:
            
            f = open(file, 'r')
            emotionNum = int(float(f.readline()))
            emotion = emotions[emotionNum]
            
            sourcefileImages = glob.glob("source_images/cohn-kanade-images/%s/%s/*" %(partNum, current_session))
            sourcefileImages = sorted(sourcefileImages, key=sortByKey)
            
            sourcefile_emotion = sourcefileImages[-1]
            sourcefile_neutral = sourcefileImages[0]
            
            destination_neutral = "sorted_set/neutral/%s" %sourcefile_neutral[42:]
            destination_emotion = "sorted_set/%s/%s" %(emotion, sourcefile_emotion[42:])
            print(destination_neutral)
            copyfile(sourcefile_neutral, destination_neutral)
            copyfile(sourcefile_emotion, destination_emotion)
    
    

    