import matplotlib.pyplot as plt

def plot(lsh_acc,knn_acc):
	plt.xlabel('K (Nearest neighbors)')
	plt.title('Accuracy vs K')
	plt.ylabel('Accuracy')
	ks=[1,2,3,4]
	plt.plot(ks,lsh_acc,'bo--',label="K-NN with LSH")
	plt.plot(ks,knn_acc,'go--',label="Exhaustive K-NN")
	plt.legend()
	plt.show()