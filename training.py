import cv2
import numpy as np
from PIL import Image
import os


#Redirects to database that we created
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()

#Specify detector
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#Method to finding labels to each image

def getImagesAndLabels(path):

    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    ids=[]

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy=np.array(PIL_img, 'uint8')

        id=int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)


        for(x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)


    return faceSamples, ids

print("\n [INFO] Training faces.....")
faces,ids=getImagesAndLabels(path)
recognizer.train(faces, np.array(ids)) #training model



#Save the model in trainer.yml file

recognizer.write('trainer/trainer.yml')

print("\n [INFO] {0} faces trained.".format(len(np.unique(ids))))
