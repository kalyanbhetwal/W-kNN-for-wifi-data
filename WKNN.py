#this is the current code being executed

import random
import math
import operator
import csv
 
def loadDataset(filename):
        trainingSet=[]
	with open(filename, 'rb') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    for x in range(1,len(dataset)):
	        for y in range(9):
	            dataset[x][y] = float(dataset[x][y])
	        trainingSet.append(dataset[x])
	return trainingSet
	      
	            
def sorensonDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance +=abs(instance1[x] - instance2[x])/abs(instance1[x] + instance2[x]) #sorenson distance
	return distance
		           
def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-2   #4
	for x in range(len(trainingSet)): 
		dist = sorensonDistance(testInstance, trainingSet[x], length)
		distances.append([trainingSet[x], dist])
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	normalizer= 0
	print distances[0][1]
	for x in range(k):
		if  distances[x][1]== 0:
			neighbors.append(distances[x])
			return neighbors
		else:
	         normalizer+=(1/distances[x][1])  #total of sorenson distance
	for x in range(k):
	        distances[x][1]= 1/(distances[x][1]*normalizer)   #normalized weight 
		neighbors.append(distances[x])
	return neighbors
	
	
def initialCor(testInstance):
        trainingSet=[]
        pos = []
        trainingSet = loadDataset('D:\knn\\rssi.csv')
        #print trainingSet
        #loadDataset('D:\knn\\validationData.csv', testInstance)
        k = 3;
        pos=getNeighbors(trainingSet, testInstance, k)
        x_cor = 0.0;
        y_cor = 0.0;
        
        if len(pos) < 3  or pos[0][1] > 0.75:
        	x_cor  = pos[0][0][7]
        	y_cor  = pos[0][0][8]
        else:
	        for i in range(k):
	            x_cor += pos[i][0][7]*pos[i][1]
	            y_cor += pos[i][0][8]*pos[i][1]
	            
        
        return x_cor,y_cor
initialCor(getRSSI())    #signal strength values as received by device