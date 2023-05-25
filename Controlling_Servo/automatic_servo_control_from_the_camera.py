#import numpy as np
#import cv2
#import RPI.GPIO as IO
#from time import sleep
#IO.setwarnings(False)
#
#IO.setmode(IO.BCM)
#IO.setup(18, IO.OUT)
#pwm = IO.PWM(18,50)

#def servoAngle(angle):  #to change the square wave size
#    duty = float(angle) / 18.0 + 2.5
#    pwm.ChangeDutyCycle(duty)

lower_range = np.array([50,150,35]) #HSV lower bound
upper_range = np.array([100,255,255]) #HSV upper limit

cap = cv2.VideoCapture(0)
kernelOpen = np.ones((5,5)) #morphology field settings
kernelClose = np.ones((20,20))

rect = False

try:
    pwm.start(7,5)
    while True:
        ret, img = cap.read()
        if not ret:
            continue
        
        img = cv2.resize(img, (340,220))
        imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        
        mask = cv2.inRange(imgHSV, lower_range, upper_range)
        
        maskOpen = cv2.morphologyEx(mask, cv2.MORFE_OPEN, kernelOpen)
        maskClose = cv2.morphologyEx(maskOpen, cv2.MORFE_CLOSE, kernelClose)
        
        maskFinal = maskClose
        
        image, conts, h = cv2.findContours(maskFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        cv2.drawContours(img, conts, -1, (255,0,0), 3)
        rect = False
        
        for i in range(len(conts)):
            x,y,w,h = cv2.boundingRect(conts[i])
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
            if(w>20 and h>20):
                rect = True
            cv2.imshow("rect", rect)
        cv2.imshow("HSV", imgHSV)
        cv2.imshow("mask", mask)
        cv2.imshow("maskOpen", maskOpen)
        cv2.imshow("maskClose", maskClose)
        cv2.imshow("img", img)
        cv2.waitKey(1)
        
        if rect:
            servoAngle(160)
        else:
            servoAngle(90)
            
except KeyboardInterrupt:
    IO.cleanup()
    cv2.destroyAllWindows()