# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 17:27:39 2022

@author: ishan
"""

import cv2
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Loading the saved model
model = load_model(r"C:\Users\ishan\data science\Projects\Face Mask Detection\Model\model_best_acc.h5")

img_width, img_height = 200,200
# Load the cascade face classifier
face_cascade = cv2.CascadeClassifier(r"C:\Users\ishan\data science\Projects\Face Mask Detection\live mask detection app\haarcascade_frontalface_default.xml")

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(r"C:\Users\ishan\Downloads\videoplayback (1).mp4")

img_count_full = 0

#parameters for text
font = cv2.FONT_HERSHEY_SIMPLEX
org = (1,1) # origin
class_label = ''
fontScale = 1
color= (255,0,0)
thickness = 2

while True:
    img_count_full+=1
    response, color_img = cap.read()
    #color_img = cv2.imread('image_link')
    if response==False:
        break
    # resize image with 50% ration    
    scale = 50
    width = int(color_img.shape[1] * scale/100)
    height = int(color_img.shape[0] * scale/100)
    dim = (width,height)
    # resize image
    color_img = cv2.resize(color_img, dim, interpolation = cv2.INTER_AREA)
    
    # Convert to grayscale
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    
    #Detect the Faces
    faces = face_cascade.detectMultiScale(gray_img,1.1,6)
    img_count = 0
    for (x,y,w,h) in faces:
        org = (x-10,y-10)
        img_count+=1
        color_face = color_img[y:y+h,x:x+w]
        cv2.imwrite(r"C:\Users\ishan\data science\Projects\Face Mask Detection\Faces\%d%dface.jpg"%(img_count_full,img_count),color_face)
        img = load_img(r"C:\Users\ishan\data science\Projects\Face Mask Detection\Faces\%d%dface.jpg"%(img_count_full,img_count),target_size=(img_width,img_height))
        
        img = img_to_array(img)/255
        img = np.expand_dims(img,axis=0)
        pred_prob = model.predict(img)
        pred = np.argmax(pred_prob)
        
        if pred==0:
            print('User with mask - predict =',pred_prob[0][0])
            class_label = "Mask"
            color = (255,0,0)
            cv2.imwrite(r'C:\Users\ishan\data science\Projects\Face Mask Detection\Faces\with_mask\%d%dface.jpg'%(img_count_full,img_count),color_face)
            cv2.rectangle(color_img,(x,y),(x+w,y+h), (0,0,255),3 )
            cv2.putText(color_img, class_label,org,font,fontScale,color,thickness,cv2.LINE_AA)
            
        else:
            print('Person not wearing mask - prob = ',pred_prob[0][1])
            class_label = 'No Mask'
            color = (0,255,0)
            cv2.imwrite(r'C:\Users\ishan\data science\Projects\Face Mask Detection\Faces\without_mask\%d%dface.jpg'%(img_count_full,img_count),color_face)
            import winsound
            frequency = 2500  # Set Frequency To 2500 Hertz
            duration = 100 # Set Duration To 1000 ms == 1 second
            winsound.Beep(frequency, duration)
            
            
        cv2.rectangle(color_img,(x,y),(x+w,y+h), (0,0,255),3 )
        cv2.putText(color_img, class_label,org,font,fontScale,color,thickness,cv2.LINE_AA)
        
    # Display Image
    cv2.imshow('Live Face Mask detection', color_img)
    
    if cv2.waitKey(1) == ord('q'):
        break
            

cap.release()
cv2.destroyAllWindows() 