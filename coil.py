from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import LSHForest
from SC import SC
import numpy as np
from plot import plot               # For plotting accuracies
import cv2
import time

Points_train=[]
CANNY = 1
a = SC()        # Shape context object 
sampls = 100    # No. of points to select

# def timing(f):
#     def wrap(*args):
#         time1 = time.time()
#         ret = f(*args)
#         time2 = time.time()
#         print '%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0)
#         return ret
#     return wrap

def get_points_from_img(src,treshold=50,simpleto=100,t=CANNY):
	''' Returns #simpleto points representing the image boundary ''' 
	# Check for valid src #
	if isinstance(src,str):  
		src=cv2.imread(src,cv2.IMREAD_GRAYSCALE);     #Load as grayscale image
	# Canny edge detection #
	if t == CANNY:
		dst=cv2.Canny(src,treshold, treshold*3, 3)
   
	points = []                 # WILL CONTAIN BOUNDARY POINTS
	h,w=np.shape(src)
	for y in xrange(h):
		for x in xrange(w):
			try:
				c = dst[y,x]
			except:
				print("====== EXCEPTION OCCURED =====")
				print(x,y)
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

def get_training_vectors():
	points=[]
	fname="coil-20-proc/obj"
	### TAINING ###
	for i in range(1,21):
		for j in range(0,72,2):
			points.append( get_points_from_img(fname+str(i)+"__" + str(j) + ".png",simpleto=sampls))

	for i in range(len(points)):
		temp=a.compute(points[i])
		temp=temp.flatten()
		Points_train.append(temp)
	print "SHAPE CONTEXT CALCULATED"	

def knn():
	neigh = NearestNeighbors(10,metric='l1')
	neigh.fit(Points_train)

	## Testing for objects with complex shapes ##
	objects_to_test=['1','3','4','5','9','14']
	avg_accuracies=[0,0,0,0]
	for obj in objects_to_test:
		fname="coil-20-proc/obj"+obj+"__";
		testPoints=[]
		Points_test=[]
		
		for i in range(1,72,2):
			testPoints.append(get_points_from_img(fname+str(i)+".png",simpleto=sampls)) 
		
		#print P
		for i in range(len(testPoints)):
			points11hash=a.compute(testPoints[i]);
			points11hash=points11hash.flatten()
			Points_test.append(points11hash)

		indices = neigh.kneighbors(Points_test, n_neighbors=10,return_distance=False)
		#print indices
		countarr=[]
		for k in range(1,5):
			count=0
			for i in range(len(indices)):
				for j in range(k):
					if(indices[i][j]>=36*(int(obj)-1) and indices[i][j]<36*int(obj)):
						count+=1
						break
			countarr.append(count)        

		accs=[]
		for i in countarr:
			accs.append((i*100)/36.0)

		for i,item in enumerate(avg_accuracies):
			avg_accuracies[i]=item+accs[i]
		# print countarr
	for i,item in enumerate(avg_accuracies):
		avg_accuracies[i]=item/len(objects_to_test)
			
	print "KNN ACCURACIES => ",avg_accuracies
	return avg_accuracies


def lsh_forest():
	lshf = LSHForest(random_state=42)
	lshf.fit(Points_train)

	## Testing for objects with complex shapes ##
	objects_to_test=['1','3','4','5','9','14']
	avg_accuracies=[0,0,0,0]
	for obj in objects_to_test:
		fname="coil-20-proc/obj"+obj+"__";
		testPoints=[]
		Points_test=[]
		
		for i in range(1,72,2):
			testPoints.append(get_points_from_img(fname+str(i)+".png",simpleto=sampls)) 
		
		#print P
		for i in range(len(testPoints)):
			points11hash=a.compute(testPoints[i]);
			points11hash=points11hash.flatten()
			Points_test.append(points11hash)

		indices = lshf.kneighbors(Points_test, n_neighbors=10,return_distance=False)
		#print indices
		countarr=[]
		for k in range(1,5):
			count=0
			for i in range(len(indices)):
				for j in range(k):
					if(indices[i][j]>=36*(int(obj)-1) and indices[i][j]<36*int(obj)):
						count+=1
						break
			countarr.append(count)        

		accs=[]
		for i in countarr:
			accs.append((i*100)/36.0)

		for i,item in enumerate(avg_accuracies):
			avg_accuracies[i]=item+accs[i]
		# print countarr
	for i,item in enumerate(avg_accuracies):
		avg_accuracies[i]=item/len(objects_to_test)
			
	print "LSH ACCURACIES => ",avg_accuracies
	return avg_accuracies

if __name__ == '__main__':
	get_training_vectors()
	knn_accs=knn()
	lsh_accs=lsh_forest()
	plot(lsh_accs,knn_accs)