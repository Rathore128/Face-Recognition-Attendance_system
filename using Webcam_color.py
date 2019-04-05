
# coding: utf-8

# Mini Project # 1 - Live Sketch Using Webcam


import cv2
import numpy as np
i = 0
# Our sketch generating function
def sketch(image,i):
    
    mask =img= image
    if i%100 == 0:
        
        t = 0
        cv2.imwrite('output'+str(i)+str(t)+'.jpg', img)
        t = t+1
        
       
        img_flip = cv2.flip(img,1)
        cv2.imwrite('output'+str(i)+str(t)+'.jpg', img_flip)
        t = t+1
        
        img_gray_blur = cv2.GaussianBlur(img,(5,5),0)
        cv2.imwrite('output'+str(i)+str(t)+'.jpg', img_gray_blur)
        t = t+1
        
        img_flip_gray_blur = cv2.GaussianBlur(img_flip, (5,5), 0)
        cv2.imwrite('output'+str(i)+str(t)+'.jpg', img_flip_gray_blur)
        t = t+1
        
        canny_edges = cv2.Canny(img, 10, 70)
        ret, maskl = cv2.threshold(canny_edges, 70, 255, cv2.THRESH_BINARY_INV)
        cv2.imwrite('output'+str(i)+str(t)+'.jpg',maskl)
        t=t+1
        
        canny_edges = cv2.Canny(img_flip, 10, 70)
        ret, maskl = cv2.threshold(canny_edges, 70, 255, cv2.THRESH_BINARY_INV)
        cv2.imwrite('output'+str(i)+str(t)+'.jpg',maskl)
        t = t+1
        
        kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1,9,-1], 
                              [-1,-1,-1]])
        
        
        sharpened_img = cv2.filter2D(img, -1, kernel_sharpening)
        cv2.imwrite('output'+str(i)+str(t)+'.jpg',sharpened_img)
        t = t+1
        
        sharpened_img_flip = cv2.filter2D(img_flip, -1, kernel_sharpening)
        cv2.imwrite('output'+str(i)+str(t)+'.jpg',sharpened_img_flip)
        t = t+1
        
        rotated_img = cv2.transpose(img)
        cv2.imwrite('output'+str(i)+str(t)+'.jpg',rotated_img)
        t = t+1
        
        
        rotated_img_flip = cv2.transpose(img_flip)
        cv2.imwrite('output'+str(i)+str(t)+'.jpg',rotated_img_flip)
        t = t+1
        
        color_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        cv2.imwrite('output'+str(i)+str(t)+'.jpg',color_img)
        t=t+1
        
        color_img_flip = cv2.cvtColor(img_flip,cv2.COLOR_BGR2HSV)
        cv2.imwrite('output'+str(i)+str(t)+'.jpg',color_img_flip)
        t=t+1
        
        B, G, R = cv2.split(mask)
        color_img1 = cv2.merge([B+20,G,R])
        color_img2 = cv2.merge([B,G+20,R])
        color_img3 = cv2.merge([B,G,R+20])
        cv2.imwrite('output'+str(i)+str(t)+'.jpg',color_img1)
        t=t+1
        cv2.imwrite('output'+str(i)+str(t)+'.jpg',color_img2)
        t=t+1
        cv2.imwrite('output'+str(i)+str(t)+'.jpg',color_img3)
        t=t+1
        
    i = i+1
    return mask,i


# Initialize webcam, cap is the object provided by VideoCapture
# It contains a boolean indicating if it was sucessful (ret)
# It also contains the images collected from the webcam (frame)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    mask, i = sketch(frame,i)
    cv2.imshow('Our Live Sketcher', mask)
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
        
# Release camera and close windows
cap.release()
cv2.destroyAllWindows()      

