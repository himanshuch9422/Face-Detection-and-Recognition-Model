import cv2
import os

from cv2 import CascadeClassifier  #For accessing camera permissions

# No Parameter so write 0
cam = cv2.VideoCapture(0)

#set height and width of frame of camera
cam.set(3, 640)  #width
cam.set(4, 480)  #height


# variable for storing faces
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

# CascadeClassifier - opencv wants to detect faces so by using haarcascade file


#Ask user id and store image

face_id = input('\n Enter user id')

#default messages

print("\n [INFO] Initializing face capture..")


#create dataset as per need 

count = 0
while(True):
    ret, img=cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #This method used to convert BGR(Blue Green Red) to gray 
    faces = face_detector.detectMultiScale(gray, 1.3, 5)


#Write images in dataset

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0))  #Default code for every face detection algo.\
        count+=1

        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h, x:x+w])
        #Show images
        cv2.imshow('image', img)


    #Specify how much time user need to wait in order to capture particular face

    k = cv2.waitKey(100) & 0xff
    if k==27:
        break
    elif count>=30:  #for efficiency take 30 images
        break


print("\n [INFO] Exiting program")

#Release the cam started initially

cam.release()
cv2.destroyAllWindows()







