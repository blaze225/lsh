from sklearn.neighbors import LSHForest
from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import NearestNeighbors
import numpy as np
import math
from scipy.interpolate import Rbf
import cv2
import random;
import math
import time
import matplotlib.pyplot as plt

def plot(lsh_acc,knn_acc):
	plt.xlabel('K (Nearest neighbors)')
	plt.title('Accuracy vs K')
	plt.ylabel('Accuracy')
	ks=[];
	for i in range(len(lsh_acc)):
		ks.append(i+1)
	print len(lsh_acc)
	plt.plot(ks,lsh_acc,'bo--',label="K-NN with LSH")
	plt.plot(ks,knn_acc,'go--',label="Exhaustive K-NN")
	plt.legend()
	plt.show()

# def timing(f):
#     def wrap(*args):
#         time1 = time.time()
#         ret = f(*args)
#         time2 = time.time()
#         print '%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0)
#         return ret
#     return wrap

def getData():
	train = np.genfromtxt('mnist_data/train_data.txt')
	test = np.genfromtxt('mnist_data/test_data.txt')
	val = np.genfromtxt('mnist_data/val_data.txt')
	return (train,test,val)

def applyLSH(lshf,test_data):
	indicesLSH = lshf.kneighbors(test_data, n_neighbors=20,return_distance=False)
	return indicesLSH

def lshFunct(test,val,n_feat,lshf,train_labels):
	
	# Parse data for separating training labels and dataset
	test_data = test[:, :-1]
	test_labels = test[:, n_feat - 1]
	test_error=[]
	training_error=[]
	
	indicesLSH=applyLSH(lshf,test_data)

	countarrLSH=[0,0,0,0,0];
	for k in range(1,6):
		for i in range(len(indicesLSH)):
			for j in range(k):
				currptr=indicesLSH[i][j];
				if(train_labels[currptr]==test_labels[i]):
					countarrLSH[k-1]+=1
					break
	for i in range(len(countarrLSH)):
		countarrLSH[i]=((countarrLSH[i])*100)/float(len(indicesLSH))

	return countarrLSH

def trainLSH(train,test,val):
	n_feat = train[0].size
	train_data = train[:, :-1]
	train_labels = train[:, n_feat - 1]
	val_data = val[:, :-1]
	val_labels = val[:, n_feat - 1]
	lshf = LSHForest(random_state=42)
	lshf.fit(train_data);
	countarrLSH=lshFunct(test,val,n_feat,lshf,train_labels)
	return countarrLSH

def knnFunct(test,val,n_feat,neigh,train_labels):
	
	# Parse data for separating training labels and dataset
	test_labels = test[:, n_feat - 1]
	
	test_data = test[:, :-1]
	test_error=[]
	training_error=[]
	indicesKNN = neigh.kneighbors(test_data, n_neighbors=20,return_distance=False)
	
	countarrKNN=[0,0,0,0,0];

	for k in range(1,6):
		for i in range(len(indicesKNN)):
			for j in range(k):
				currptr=indicesKNN[i][j];
				if(train_labels[currptr]==test_labels[i]):
					countarrKNN[k-1]+=1
					break
	for i in range(len(countarrKNN)):
		countarrKNN[i]=((countarrKNN[i])*100)/float(len(indicesKNN))
	return countarrKNN

def trainKNN(train,test,val):
	n_feat = train[0].size
	train_data = train[:, :-1]
	train_labels = train[:, n_feat - 1]
	val_data = val[:, :-1]
	val_labels = val[:, n_feat - 1]
	neigh = NearestNeighbors(20,0.5,metric='cosine',algorithm="brute")
	neigh.fit(train_data)
	countarrKNN=knnFunct(test,val,n_feat,neigh,train_labels)
	return countarrKNN

if __name__ == '__main__':

	train,test,val=getData();

	accuracyLSH=trainLSH(train,test,val);
	accuracyKNN=trainKNN(train,test,val)
	print accuracyKNN
	print accuracyLSH
	plot(accuracyLSH,accuracyKNN)