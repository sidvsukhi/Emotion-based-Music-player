# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 18:35:53 2018

@author: Siddhant
"""

import cv2
import numpy as np
import argparse
import time
import glob
import os
import Update_Model
import pandas, random
import sys, subprocess
video_capture = cv2.VideoCapture(0) #the default webcam (0, the first webcam device detected). Change if you have more than one webcam connected and want to use another one than the default one
facecascade = cv2.CascadeClassifier("C:\\Users\siddh\Anaconda3\Lib\haarcascades\haarcascade_frontalface_default.xml") #load the trained classifier model
facecascade_two = cv2.CascadeClassifier("C:\\Users\siddh\Anaconda3\Lib\haarcascades\haarcascade_frontalface_alt2.xml")
#C:\\Users\Mukund\Anaconda3\Lib\haarcascades\haarcascade_frontalface_alt.xml
facecascade_three = cv2.CascadeClassifier("C:\\Users\siddh\Anaconda3\Lib\haarcascades\haarcascade_frontalface_alt.xml")
facecascade_four = cv2.CascadeClassifier("C:\\Users\siddh\Anaconda3\Lib\haarcascades\haarcascade_frontalface_alt1.xml")
fishface = cv2.face.FisherFaceRecognizer_create()
#try:
fishface.read("trained_emoclassifier.xml")
#except:
   # print("no xml found. Using --update will create one.")
parser = argparse.ArgumentParser(description="Options for the emotion-based music player") #Create parser object
parser.add_argument("--update", help="Call to grab new images and update the model accordingly", action="store_true") #Add --update argument
args = parser.parse_args() #Store any given arguments in an object
facedict = {}
emotions = ["angry", "happy", "sad", "neutral"]
facedict = {} #Create face dictionary
df = pandas.read_excel("EmotionLinks.xlsx") #open Excel file
actions = {}
actions["angry"] = [x for x in df.angry.dropna()] #We need de dropna() when columns are uneven in length, which creates NaN values at missing places. The OS won't know what to do with these if we try to open them.
actions["happy"] = [x for x in df.happy.dropna()]
actions["sad"] = [x for x in df.sad.dropna()]
actions["neutral"] = [x for x in df.neutral.dropna()]
def open_stuff(filename): #Open the file, credit to user4815162342, on the stackoverflow link in the text above
    if sys.platform == "win32":
        os.startfile("C:\\Project\\abc.m4a")
    else:
        opener ="open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, filename])
def crop_face(gray, face):
    for (x, y, w, h) in face:
        faceslice = gray[y:y+h, x:x+w]
        faceslice = cv2.resize(faceslice, (350,350))
    facedict["face%s" %(len(facedict)+1)] = faceslice #append sliced face as a numbered face to the dictionary
    return faceslice
def update_model(emotions):
    print("Model update mode active")
    check_folders(emotions)
    for i in range(0, len(emotions)):
        save_face(emotions[i])
    print("collected images, looking good! Now updating model...")
    Update_Model.update(emotions)
    print("Done!")
def check_folders(emotions): #check if folder infrastructure is there, create if absent
    for x in emotions:
        if os.path.exists("dataset\\%s" %x):
            pass
        else:
            os.makedirs("dataset\\%s" %x)
def save_face(emotion):
    print("\n\nplease look " + emotion + " when the timer expires and keep the expression stable until instructed otherwise.")
    for i in range(0,5):#Timer to give you time to read what emotion to express
        print(5-i)
        time.sleep(1)
    while len(facedict.keys()) < 16: #Grab 15 images for each emotion
        detect_face()
    for x in facedict.keys(): #save contents of dictionary to files
        cv2.imwrite("dataset\\%s\\%s.jpg" %(emotion, len(glob.glob("dataset\\%s\\*" %emotion))), facedict[x])
    facedict.clear() #clear dictionary so that the next emotion can be stored
def recognize_emotion():
    predictions = []
    confidence = []
    for x in facedict.keys():
        pred, conf = fishface.predict(facedict[x])
        cv2.imwrite("images\\%s.jpg" %x, facedict[x])
        predictions.append(pred)
        confidence.append(conf)
    recognized_emotion = emotions[max(set(predictions), key=predictions.count)]
    print("I think you're %s" %recognized_emotion)
    actionlist = [x for x in actions[recognized_emotion]] #<----- get list of actions/files for detected emotion
    random.shuffle(actionlist) #<----- Randomly shuffle the list
    open_stuff(actionlist[0]) #<----- Open the first entry in the list
def grab_webcamframe():
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)
    return clahe_image
def detect_face():
    clahe_image = grab_webcamframe()
    face = facecascade.detectMultiScale(clahe_image, scaleFactor=1.1, minNeighbors=15, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
    face_two = facecascade_two.detectMultiScale(clahe_image, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_three = facecascade_three.detectMultiScale(clahe_image, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_four = facecascade_four.detectMultiScale(clahe_image, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    if len(face) == 1:
        faceslice = crop_face(clahe_image, face)
        #faceslice = cv2.resize(faceslice, (350, 350))
        return faceslice
    elif len(face_two) == 1:
        faceslice = crop_face(clahe_image, face_two)
        #faceslice = cv2.resize(faceslice, (350, 350))
        return faceslice
    elif len(face_three) == 1:
        faceslice = crop_face(clahe_image, face_three)
        #faceslice = cv2.resize(faceslice, (350, 350))
        return faceslice
    elif len(face_four) == 1:
        faceslice = crop_face(clahe_image, face_four)
        #faceslice = cv2.resize(faceslice, (350, 350))
        return faceslice
    else:
        print("no/multiple faces detected, passing over frame")
while True:
    detect_face()
    #if args.update: #If update flag is present, call update function
    #update_model(emotions)
    #break
    #elif len(facedict) == 10: #otherwise it's regular a runtime, continue normally with emotion detection functionality
    recognize_emotion()
    break