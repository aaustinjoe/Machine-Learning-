#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 18:49:54 2019

@author: austin
"""


import numpy as np
import cv2
import os
import matplotlib.pyplot as plt 

face_cascade = cv2.CascadeClassifier("\home/austin/.config/spyder-py3/haarcascade_frontalcatface.xml") 
color = {"b":(255,0,0),"r":(0,0,255),"g":(0,255,0)}
count=0
person_id=1
#person_id = input('\n enter user id end press <return> ==>  ')
#print("\n [INFO] Initializing face capture. Look the camera and wait ...")

def detect(img, classifier,scaleFactor,minNeighbors,color,text):  #classifier is my cascade
    #img = cv2.flip(img, -1) # flip video image vertically
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray,scaleFactor,minNeighbors,minSize=(4,4)) #minSize == smaller than are ignored
    Nper1frame = len(faces) #number of person in frame
    print (Nper1frame)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)  # 2 is borderthickness
        #cv2.putText(img,text,(x,y-4)) # 0.8 is text size and 2 is fontweight
        cv2.imwrite('home/austin/.config/spyder-py3/face' + str(person_id)+'.' +str(count)+'.jpeg', gray[y:y+h,x:x+w])
        
    return img

#def detect(img,classifier,count):
#    coords,img = rectangle(img,classifier,1.2,5,color["b"],"FACE")
#    if len(coords)== 4: 
#        roi_img = img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]] #region of interest and coordinates must be in order of(y,x)
#        roi_img = cv2.cvtColor(roi_img,cv2.COLOR_BGR2GRAY)
#        save(roi_img,person,count)
#    return img 

#def save(img,person,img_id):
#    cv2.imwrite('D:\\AI-ML projects\\face_recognition\\data\\person.' + str(person_id)+'.' +str(img_id)+'.jpeg', img)
    

capture=cv2.VideoCapture(0)


while(True):
    _, img = capture.read()
    img = detect(img,face_cascade,1.2,5,color["b"],"FACE")

    #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    """for (x,y,w,h) in faces:
        print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]# region of interest and coordinates of the face
        img_item = 'yy.jpg' # file that save image
        cv2.imwrite(img_item, roi_gray)"""
        
    count+=1
    cv2.imshow('FACERECOGNITON',img)
    if cv2.waitKey(20) & 0xFF ==ord('q'):
        break
    elif count == 30:   # this will capture only 30 images of a single person
        break
capture.release()
cv2.destroyAllWindows()

#plt.imshow(img)    
#img = plt.imread("C:\\Users\\Atharva Kulkarni\\Desktop\\Test_Data\\face_0_2458.jpeg")
#img = detect(img,face_cascade,1.2,5,color["b"],"FACE")
