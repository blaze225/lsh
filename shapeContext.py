'''
Created on 04-Apr-2017

@author: ethan
'''

from SC import SC
import numpy as np
import math
from scipy.interpolate import Rbf
import sys
from math import sin, cos, sqrt, pi
import cv
import cv2

CANNY = 1

def get_points_from_img(src,treshold=50,simpleto=100,t=CANNY):
    # Check for valid src #
    if isinstance(src,str):  
        src = cv.LoadImage(src, cv.CV_LOAD_IMAGE_GRAYSCALE)     #Load as grayscale image 
    # Canny edge detection #
    if t == CANNY:
        dst = cv.CreateImage(cv.GetSize(src), 8, 1) # dst contains border
        storage = cv.CreateMemStorage(0)
        cv.Canny(src, dst, treshold, treshold*3, 3)
    
    # Testing #
#     ABC = np.zeros((cv.GetSize(dst)[1],cv.GetSize(dst)[0]))
#     for y in xrange(cv.GetSize(src)[1]):
#         for x in xrange(cv.GetSize(src)[0]):
#             ABC[y,x] = dst[y,x] 
#     cv2.imwrite('color_img.jpg',ABC)
#     cv2.imshow("image", ABC);
#     cv2.waitKey();
#     cv.SaveImage("akshay123.png", dst);  
   
    points = []                 # WILL CONTAIN BOUNDARY POINTS
    w,h = cv.GetSize(src)       # get width and height of image
    for y in xrange(h):
        for x in xrange(w):
            try:
                c = dst[y,x]
            except:
                print "====== EXCEPTION OCCURED ====="
                print x,y
            if c == 255:    #====ADDING BOUNDARY POINTS===="
                points.append((x,y))
    
    r = 2    # STEP SIZE
    # Removing extra points from points #
    while len(points) > simpleto:
        newpoints = points
        xr = range(0,w,r)
        yr = range(0,h,r)
        for p in points:
            if p[0] not in xr and p[1] not in yr:
                newpoints.remove(p)
                if len(points) <= simpleto:
                    break    
        r += 1   
    return newpoints

if __name__ == '__main__':
    
    a = SC()        # Shape context object 
    sampls = 100    # No. of points to select 
    points1 = get_points_from_img('coil-20-proc/obj1__0.png',simpleto=sampls)
    points2 = get_points_from_img('coil-20-proc/obj2__0.png',simpleto=sampls)
    P = a.compute(points1)
    print P
    Q = a.compute(points2)
    print Q
    print "============================================  DONE ================================================"
    