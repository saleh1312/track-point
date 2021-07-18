import cv2
import numpy as np


class detection:
    #detect first frame and his feautures
    def __init__(self,im1,):
        self.im1=im1
        self.detector = cv2.xfeatures2d.SIFT_create()
        self.matcher = cv2.FlannBasedMatcher(
                dict(algorithm=0,tress=5),dict())
            
        self.kp1, self.des1 = self.detector.detectAndCompute(
                self.im1,None)
            
        self.point=np.float32([[175,242]])

        

    #detect , match points, extract best points( points has low variation
    # becuase we programe in video with frames) , compute the matrix,
    #and change the point coordinates and last step
    #to switch between first and second frame
    def dnd(self,im2):
        kp2, des2 = self.detector.detectAndCompute(im2,None)

        matches = self.matcher.knnMatch(self.des1,des2,k=2)

        good = []
        for m,n in matches:
            if m.distance < 0.6*n.distance:
                good.append([m])

        img3 = cv2.drawMatchesKnn(self.im1,self.kp1,im2,kp2
                                  ,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        points = np.zeros((len(good), 4), dtype=np.float32)  

        for i, m in enumerate(good):
           
            points[i, :2] = self.kp1[m[0].queryIdx].pt    #gives index of the descriptor in the list of query descriptors
            points[i, 2:] = kp2[m[0].trainIdx].pt    #gives index of the descriptor in the list of train descriptors
            
         

        l1=points[:,:2].astype(np.float32)
        l2=points[:,2:].astype(np.float32)
        h,mask=cv2.findHomography(l1, l2, cv2.RANSAC,5.0)
        
        self.mapp(h)


        self.des1=des2
        self.kp1=kp2
        self.im1=im2
        return img3

    @staticmethod
    def process_img(img):
        c=img.copy()
        c=cv2.resize(c,(512,512))
        c=cv2.cvtColor(c,cv2.COLOR_BGR2GRAY)

        return c
    
    def mapp(self,h):
        #sometimes it fails to detect the matrix
        if type(h)==type(None):
            return
        ds=cv2.perspectiveTransform(self.point[None,:,:],h)
       
        x=ds[0][0][0]
        y=ds[0][0][1]
     
 
        self.point[0,1]=y
        self.point[0,0]=x
        

