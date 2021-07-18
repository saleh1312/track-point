import cv2
import numpy as np
from detect2 import detection

cap =cv2.VideoCapture('v4.wmv')


out = cv2.VideoWriter('res.avi',cv2.VideoWriter_fourcc("X", "V", "I", "D"),20, (512,512))

first= detection.process_img(cap.read()[1])
pro=detection(first)


while True:
    ret,sec = cap.read()
    
    
    if ret==False:
        break
    
    sec=detection.process_img(sec)
    
    copy=sec.copy()
    copy=cv2.cvtColor(copy,cv2.COLOR_GRAY2BGR)

    img3=pro.dnd(sec)
    

    cv2.circle(copy,(pro.point[0,0],pro.point[0,1]),7,(0,255,255),-1)
    cv2.imshow('sss',copy)
    cv2.imshow('sss2',img3)
    out.write(copy)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
out.release()
cv2.destroyAllWindows()


