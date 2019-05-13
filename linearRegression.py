'''

Name :- Siddharth Nahar
Entry No :- 2016csb1043
Date :- 11/9/18
Purpose :- Extract Features data and create vectors

'''

import numpy as np
from numpy.linalg import inv
import math
import random
import matplotlib.pyplot as plt
#------------------------------------------------------------------------

#Standardization of vector

def vectorStandardize(X,TestX):

	#print(X.shape)
	Rows = (X.shape)[0]

	RowsTest = TestX.shape[0]

	Cols = (X.shape)[1]
	#Calculating mean for each column
	meanX = np.mean(X,axis = 0)

	#Calculate Std for each Column
	StdX = np.std(X,axis = 0)

	#Standardizing Each column of matrix

	#Standardizing Same for Test Set
	for i in range(0,Cols):
		X[:,i] = (X[:,i] - meanX[i])/StdX[i]
		TestX[:,i] = (TestX[:,i] - meanX[i])/StdX[i]

	Ones = np.ones((Rows,1),dtype = float)
	TestOnes = np.ones((RowsTest,1),dtype = float)

	X = np.append(Ones,X,axis = 1)
	TestX = np.append(TestOnes,TestX,axis = 1)
	
	return X,TestX
#------------------------------------------------------------------------

#Linear Ridgression Solution

def mylinridgereg(X,Y,lam):


	# W = (Xt*X + lam*I)^-1*Xt*Y
	Xt = X.transpose()

	Cols = X.shape[1]

	I = np.identity(Cols)
	#I[0,0] = 0  Not Regularizing w0 weight
	I[0,0] = 0
	RegMat = lam*I

	W = Xt@X + RegMat;

	W = inv(W)

	W = W@Xt
	W = W@Y
	#print(W)

	return W

#------------------------------------------------------------------------

#Predict the value of output
def mylinridgeregeval(X,W):

	#Xt = X.transpose()
	Y = X@W

	#print(Y)
	return Y

#------------------------------------------------------------------------

#Function for Calculation of Mean Squared Error
def meanSquaredError(T,Tdash):

	Error = T - Tdash
	Error = Error**2

	Sum = np.sum(Error)

	N = Error.shape[0]

	return Sum/N

#-------------------------------------------------------------------------
#Function to Partition the TrainSet according to frac

def getPartition(X,Y,Indices,N):

	#N = math.ceil(frac*X.shape[0])
	ind = np.random.choice(Indices,N,replace = False)
	#print(Y.shape)
	
	Xdash = X[ind,:]
	Ydash = Y[ind]


	return Xdash,Ydash 

#------------------------------------------------------------------------

#Function to Run from each matrix
def run(featureMatrix):

	size = math.ceil(0.2*featureMatrix.shape[0])

	np.random.shuffle(featureMatrix)


	#Extarct dimension of matrix
	tuples = featureMatrix.shape

	#Create X vector and Y vector
	featureMatrixX = featureMatrix[:,0:tuples[1]-1]

	featureMatrixY = featureMatrix[:,-1]

	#Got The Test Set
	TestX = featureMatrixX[:size]
	TestY = featureMatrixY[:size]


	featureMatrixX = featureMatrixX[size:]
	featureMatrixY = featureMatrixY[size:]


	TotalSize = featureMatrixX.shape[0]
	randomIndices = np.arange(TotalSize)
	arr = np.linspace(100,TotalSize,100,dtype = int)

	#print("Values for Examples of Train Set : " + str(arr))
	lamValues = np.linspace(0.0001,50.0,20)

	meanErrorTrain = list()
	meanErrorTest  = list()

	minErrorTest = 100.0
	BestFraction = None
	BestLam = None
	#minPredictedY = np
	#For Training Predicted vs Actual
	BestYTrainPred = None
	BestYTrainAct = None

	#For Test Set Predicted vs Actual
	BestYTestPred = None
	BestYTestAct = TestY


	Repetitions = 50
	#Best Model is one having minimum Test Error

	for i in range(0,len(lamValues)):
		meanErrorTrain.append(list())
		meanErrorTest.append(list())
	
	for rowValue in np.nditer(arr):
		
		'''
		X,Y = getPartition(featureMatrixX,featureMatrixY,randomIndices,rowValue)
		#Standardize matrix for all columns
		XTrain,XTest = vectorStandardize(X.copy(),TestX.copy())
		'''
		count = 0
		for val in np.nditer(lamValues):

			errorTrain = 0
			errorTest = 0

			for repeat in range(0,Repetitions):


				X,Y = getPartition(featureMatrixX,featureMatrixY,randomIndices,rowValue)
				#Standardize matrix for all columns
				XTrain,XTest = vectorStandardize(X.copy(),TestX.copy())

				#Train model for each fraction of Train Set and Lambda values
				W = mylinridgereg(XTrain,Y,val)

				#Calculate Train and Test Output from values
				YTrain = mylinridgeregeval(XTrain,W)
				YTest = mylinridgeregeval(XTest,W)

				#Calculate Error on Train and Test
				errorTraindash = meanSquaredError(Y,YTrain)
				errorTestdash  = meanSquaredError(TestY,YTest)
			
				if(errorTestdash <= minErrorTest):
					minErrorTest = errorTestdash
					BestYTrainPred = YTrain
					BestYTrainAct = Y
					BestYTestPred = YTest
					BestFraction = rowValue
					BestLam = val

				errorTrain += errorTraindash
				errorTest += errorTestdash

			errorTrain = errorTrain/Repetitions
			errorTest  = errorTest/Repetitions
			
			#Append it to list of Train Error and Test Error
			meanErrorTrain[count].append(errorTrain)
			meanErrorTest[count].append(errorTest)

			print("For Lamda Value : " +str(val)+ " And For the length of following Train sEt = " + str(rowValue))
			print("Train Error : " + str(errorTrain) + "\nTest Error : " +str(errorTest))

			print("-------------------------------------------------------------------------")

			count += 1

	errorTrainNP = np.array(meanErrorTrain)
	errorTestNP  = np.array(meanErrorTest)

	print(BestFraction)
	print(BestLam)
	count = 0
	#Plots for each Training fraction lambda vs mean squared error
	#Found that as Train Examples Increases Variance decreases so PArameter doesn't cause sudden changes
	
	for rowValue in np.nditer(arr):
		#frac = 1.0*rowValue/TotalSize
		#plt.figure(count)
		plt.plot(lamValues,errorTrainNP[:,count],'--b',marker = 'o',markersize=5,markerfacecolor = 'red',label = 'Training Error')
		plt.plot(lamValues,errorTestNP[:,count],'--r',marker = 'o',markersize=5,markerfacecolor = 'blue',label = 'Test Set Error')
		plt.title("Number of Training Exaples = " + str(rowValue) + ",Total Examples = " + str(TotalSize))
		plt.xlabel("Lambda Values")
		plt.ylabel("Mean Squared Error")
		plt.gca().set_xlim([0,50])
		plt.gca().set_ylim([3,8])
		plt.legend(loc = 'upper right')
		plt.savefig("Picture"+str(count))
		plt.clf()
		
		count += 1
	
	
	
	#Plots for Minimum mean squaredTest Error vs Training frac and lamda vs training frac
	minValues = np.amin(errorTestNP,axis = 0)
	minLambda = np.argmin(errorTestNP,axis = 0)

	#print(minLambda)
	minL = lamValues[minLambda]
	fractions = 1.0*arr/TotalSize	

	plt.plot(1)
	#plt.subplot(211)
	plt.plot(fractions,minValues,linestyle = '--',color = 'green',marker = 'o',markersize='3',markerfacecolor='red')
	
	plt.xlabel("Fraction Values")
	plt.ylabel("Minimum Squared Error")
	plt.gca().set_xlim([0,1])
	plt.gca().set_ylim([3,7])
	plt.title("Minimum Mean squared vs Data Fraction")
	plt.savefig("Part 8a : Min MSE vs Fraction")
	plt.show()
	plt.clf()

	#plt.subplot(213)
	plt.plot(2)
	plt.plot(fractions,minL,linestyle='--',marker = 'o',markersize='3',markerfacecolor='red',color='green')
	
	plt.xlabel("Fraction Values")
	plt.ylabel("Lambda Values")
	plt.gca().set_xlim([0,1])
	plt.gca().set_ylim([-10,60])
	plt.title("Minimum Lambda values vs Data Fraction")
	plt.savefig("Part 8b : Min Lambda vs Fraction")
	
	plt.show()
	plt.clf()
	#print(errorTrainNP.shape)
	#print(errorTestNP.shape)

	stX = np.linspace(0,30,10)
	stY = np.linspace(0,30,10)

	plt.plot(1)
	plt.scatter(BestYTrainPred,BestYTrainAct)
	plt.title("Actual Train Y vs Predicted Train Y \n Fraction = "+str(BestFraction)+ ",Total Examples = " + str(TotalSize)+"\nLambda = "+str(BestLam) +"\nError : "+str(minErrorTest))
	plt.xlabel("Predicted Training Values")
	plt.ylabel("Actual Training Values")
	plt.gca().set_xlim([0,30])
	plt.gca().set_ylim([0,30])


	plt.plot(stX,stY,'--r')
	plt.savefig("Part 9a : Training Set")
	
	plt.show()

	plt.clf()
	plt.plot(1)
	plt.scatter(BestYTestPred,BestYTestAct)
	plt.title("Actual Test Y vs Predicted Test Y \n Fraction = "+str(BestFraction)+ ",Total Examples = " + str(TotalSize)+"\nLambda = "+str(BestLam)+"\nError : "+str(minErrorTest))
	plt.xlabel("Predicted Test Values")
	plt.ylabel("Actual Test Values")
	plt.gca().set_xlim([0,30])
	plt.gca().set_ylim([0,30])

	plt.plot(stX,stY,'--r')
	plt.savefig("Part 9b : Test Set")
	
	plt.show()
#--------------------------------------------------------------------
if __name__ == "__main__":

	#Feature Extraction from linregdata file
	linRegData = open('linregdata','r')

	featureVector = list()
	#Traverse through file and create feature vector


	for line in linRegData:
		line = line.split(',')

		vector = list()

		for data in line:
			if data == 'F':
				vector.append(1.0)
				vector.append(0.0)
				vector.append(0.0)
			elif data == 'I':
				vector.append(0.0)
				vector.append(1.0)
				vector.append(0.0)
			elif data == 'M':
				vector.append(0.0)
				vector.append(0.0)
				vector.append(1.0)
			else:
				vector.append(float(data))

		
		featureVector.append(vector)
		
	#Convert Feature list of list to numpy array
	featureMatrix = np.array(featureVector)
	run(featureMatrix)
