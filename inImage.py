#Import openCV to perform image processing
import cv2

trainedDataset=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#haarcascade_frontalface_default it was designed by openCV.
#it is used to detect front view of the face

img=cv2.imread('images/img1.jpg')
#imread() is an built in function.it is used to read the image.

#convert the colored image into gray color.
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#detecting the face.this function is suitable for detecting multiple or single image
faces=trainedDataset.detectMultiScale(gray)
print(faces)
#printing co-ordinates of the image.

for x,y,w,h in faces:

    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

#imshow() is the builtin function is used to show the image.
cv2.imshow('img1',img)
#waitkey() is the builtin function is used to display the result.
cv2.waitKey()
