import cv2
import glob

faceDet_1 = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceDet_2 = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceDet_3 = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceDet_4 = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]

def detect_faces(emotion):
    files = glob.glob("sorted_set/%s/*" %emotion)
    fileNum = 0
    for f in files:
        headFrame = cv2.imread(f)
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
                cv2.imwrite("dataset/%s/%s.jpg" %(emotion, fileNum), result)
            except:
                print("ERROR")
                pass
        fileNum += 1

for emotion in emotions:
    detect_faces(emotion)
