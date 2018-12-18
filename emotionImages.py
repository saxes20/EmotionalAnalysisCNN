import cv2
import glob
import random
import numpy as np

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
classifier = cv2.face.FisherFaceRecognizer_create()

faceDet_1 = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceDet_2 = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceDet_3 = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceDet_4 = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

data = {}

def retrieve_sets(emotion):
    files = glob.glob("dataset/%s/*" %emotion)
    for k in range(3):
        random.shuffle(files)
    training_set = files[:int(len(files)*.75)]
    validation_set = files[-int(len(files)*.25):]
    return training_set, validation_set

def create_sets():
    X = []
    y = []
    valX = []
    valY = []
    for emotion in emotions:
        training, validation = retrieve_sets(emotion)
        label = emotions.index(emotion)
        for img in training:
            image = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY)
            X.append(image)
            y.append(label)
        for img in validation:
            image = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY)
            valX.append(image)
            valY.append(label)
    return X, y, valX, valY

def recognizer():
    X, y, testX, testY = create_sets()
    print("Training on " + str(len(X)) + " images")
    classifier.train(X, np.asarray(y))
    print("Validating on " + str(len(testX)) + " images")
    count = 0
    correct = 0
    for image in testX:
        prediction, confidence = classifier.predict(image)
        if (prediction == testY[count]):
            correct += 1
        count += 1
    print(correct)
    print(count)
    return 100 * (float(correct) / float(count))

def process_image(headFrame):
    gray = cv2.cvtColor(headFrame, cv2.COLOR_BGR2GRAY)
    face_1 = faceDet_1.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_2 = faceDet_2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_3 = faceDet_3.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_4 = faceDet_4.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    facefeatures = ""
    if len(face_1) == 1:
        facefeatures = face_1
    elif len(face_2) == 1:
        facefeatures = face_2
    elif len(face_3) == 1:
        facefeatures = face_3
    elif len(face_4) == 1:
        facefeatures = face_4
    for (x, y, w, h) in facefeatures:
        gray = gray[y:y+h, x:x+w]
        try:
            result = cv2.resize(gray, (350,350))
            return result
        except:
            print("ERROR")
            pass
    return ""

def rec_predict(img):
    img = process_image(img)
    prediction, confidence = classifier.predict(img)
    print(emotions[prediction])
    print("Confidence: " + str(confidence))
    return emotions[prediction]

#metascore = []
#for i in range(0,5):
#    perc = recognizer()
#    print ("got " + str(perc) + " percent correct!")
#    metascore.append(perc)
#print(np.mean(metascore))
