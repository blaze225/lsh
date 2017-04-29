from sklearn.neighbors import LSHForest
from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import NearestNeighbors
from SC import SC
import numpy as np
import math
from scipy.interpolate import Rbf
import sys
from math import sin, cos, sqrt, pi
import cv
import cv2
import random;
import math

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
	points=[];
	fname="coil-20-proc/obj";
	### TAINING ###
	for i in range(1,21):
		print str(i) + "  "
		for j in range(0,72,2):
			points.append( get_points_from_img(fname+str(i)+"__" + str(j) + ".png",simpleto=sampls))
	
	Points_train=[]
	for i in range(len(points)):
		temp=a.compute(points[i])
		temp=temp.flatten()
		Points_train.append(temp)	

	lshf = LSHForest(random_state=42)
	lshf.fit(Points_train)

	
	count=0
	testPoints=[]
	Points_test=[]
	fname="coil-20-proc/obj14__";
	for i in range(1,72,2):
		testPoints.append(get_points_from_img(fname+str(i)+".png",simpleto=sampls)) 
	

	#print P
	for i in range(len(testPoints)):
		points11hash=a.compute(testPoints[i]);
		points11hash=points11hash.flatten()
		Points_test.append(points11hash)

	indices = lshf.kneighbors(Points_test, n_neighbors=20,return_distance=False)
	print indices


	#myindices=[P[i] for i in indices]
	# nbrs = NearestNeighbors(algorithm='brute', metric='euclidean',n_neighbors=10).fit(myindices)
	# exact_neighbors = nbrs.kneighbors(points11, return_distance=False)

	countarr=[0,0,0,0];

	for k in range(1,5):
		for i in range(len(indices)):
			for j in range(k):
				if(indices[i][j]>=468 and indices[i][j]<=503):
					countarr[k-1]+=1
					# print indices[i][j]
					break

	for i in range(len(countarr)):
		print i,"accuracy =",((countarr[i])*100)/36.0
	# else:
		# print "used exact neighbour here got ",exact_neighbors[0][0];
	# print count
	# print "ACCURACY BITCH => ",((count)*100)/36.0