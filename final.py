import pylab
import cv2
import emotionImages
from collections import defaultdict
import subprocess
import os

filename = '/Users/sameer/Desktop/PJAS/voices/Others/Actor_01/01-01-03-01-01-02-01.mp4'
frames = []

cap = cv2.VideoCapture(filename)
while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:
    #cv2.imshow('Frame',frame)
    frames.append(frame)
  else: 
    break
cap.release()
cv2.destroyAllWindows()

def convertWav ():
    result = filename[:len(filename)-4] + '.wav'
    subprocess.call(['ffmpeg', '-i', filename, result])

convertWav()

accuracy = emotionImages.recognizer()
print("Accuracy of FisherFace: " + str(accuracy))

facialExpressions = []
for f in frames:
    p = emotionImages.rec_predict(f)
    facialExpressions.append(p)

face = defaultdict(int)
for em in facialExpressions:
    face[em] += 1






