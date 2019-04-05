
# coding: utf-8

# Mini Project # 1 - Live Sketch Using Webcam


import cv2
import numpy as np
i =1

# Our sketch generating function
def sketch(image,i):
    
    mask =img= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    mask = cv2.flip(mask,1)        
    cv2.imwrite('new_f/dataset/test_set/Ayush/'+str(i)+'.jpg', img)
    i+=1
    img_flip = cv2.flip(img,1)    
    cv2.imwrite('new_f/dataset/test_set/Ayush/'+str(i)+'.jpg', img_flip)
    i+=1
        
    img_gray_blur = cv2.GaussianBlur(img,(5,5),0)
    cv2.imwrite('new_f/dataset/test_set/Ayush/'+str(i)+'.jpg', img_gray_blur)
    i+=1
        
    img_flip_gray_blur = cv2.GaussianBlur(img_flip, (5,5), 0)
    cv2.imwrite('new_f/dataset/test_set/Ayush/'+str(i)+'.jpg', img_flip_gray_blur)
    i+=1
        
        
    kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1,9,-1], 
                              [-1,-1,-1]])
        
        
    sharpened_img = cv2.filter2D(img, -1, kernel_sharpening)
    cv2.imwrite('new_f/dataset/test_set/Ayush/'+str(i)+'.jpg',sharpened_img)
    i+=1
        
    sharpened_img_flip = cv2.filter2D(img_flip, -1, kernel_sharpening)
    cv2.imwrite('new_f/dataset/test_set/Ayush/'+str(i)+'.jpg',sharpened_img_flip)
    i+=1
        
    rotated_img = cv2.transpose(img)
    cv2.imwrite('new_f/dataset/test_set/Ayush/'+str(i)+'.jpg',rotated_img)
    i+=1
        
        
    rotated_img_flip = cv2.transpose(img_flip)
    cv2.imwrite('new_f/dataset/test_set/Ayush/'+str(i)+'.jpg',rotated_img_flip)
    i = i+1
    
    return mask,i


# Initialize webcam, cap is the object provided by VideoCapture
# It contains a boolean indicating if it was sucessful (ret)
# It also contains the images collected from the webcam (frame)
cap = cv2.VideoCapture(0)

while True and i < 98:
    ret, frame = cap.read()
    mask, i = sketch(frame,i)
    cv2.imshow('Our Live Sketcher', mask)
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
        
# Release camera and close windows
cap.release()
cv2.destroyAllWindows()      

