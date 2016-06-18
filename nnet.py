import random
import math
import csv

class nnet:
	def __init__(self,arch):
		self.L = len(arch)-1
		self.d = arch
		self.w = []
		for i in range(1,self.L+1):
			W = Matrix(self.d[i-1]+1,self.d[i],0.0)
			self.w.append(W)

	def dim(self,k):
		return self.d[k]

	def get_weights(self,k):
		return self.w[k]

	def mod_weights(self,k,matrix):
		self.w[k].assign(matrix)

	def initialize_weights(self,sigma):
		for k in range(0,self.L):
			tmp = Matrix(self.d[k]+1,self.d[k+1],0.0)
			for i in range(0,self.d[k]+1):
				for j in range(0,self.d[k+1]):
					tmp.mod(i,j,random.normalvariate(0.0,sigma))
			self.w[k].assign(tmp)


	def make_input(self):
		result = []
		for i in range(0,self.L+1):
			result.append(Vector(self.d[i]+1,0.0))
		return result

	def make_signal(self):
		result = []
		for i in range(1,self.L+1):
			result.append(Vector(self.d[i],0.0))
		return result

	def make_sensitivity(self):
		result = []
		for i in range(1,self.L+1):
			result.append(Vector(self.d[i],0.0))
		return result

	def make_gradient(self):
		result = []
		for i in range(0,self.L):
			result.append(self.w[i].multiply_by_constant(0.0))
		return result

	def clear_gradient(self,gradients):
		for k in range(0,len(gradients)):
			for i in range(0,gradients[k].rowDim):
				for j in range(0,gradients[k].colDim):
					gradients[k].mod(i,j,0.0)

	def forward_propagate(self,dataPoint,inputs,signals):
		inputs[0].assign(dataPoint)
		for l in range(0,self.L):
			signals[l].assign(inputs[l].multiply_by_matrix(self.w[l].transpose()))
			inputs[l+1].assign(augment_one(sigmoid_map(signals[l])))
		return inputs[self.L].get(1)

	def back_propagate(self,inputs,sensitivities):
		sensitivities[self.L-1].assign(inverse_signal(inputs[self.L]))
		T = Vector(0,0.0)
		for l in range(self.L-2,-1,-1):
			T.assign(inverse_signal(inputs[l+1]))
			sensitivities[l].assign(sensitivities[l+1].multiply_remove(self.w[l+1]).multiply_componentwise(T))

	def predict_internal(self, data, inputs, signals):
		result = Vector(len(data))
		for i in range(0,len(data)):
			dataPoint = Vector(0,0,0)
			dataPoint.assign(augment_one(data[i]))
			result[i] = self.forward_propagate(dataPoint,inputs,signals)
		return result

	def update_gradient(self,inputs,sensitivities,gradients,gradients_update,y,N):
		for l in range(0,self.L):
			gradients_update[l].assign(sensitivities[l].outer_product(inputs[l]).multiply_by_constant(2*(inputs[self.L].get(1)-y)/N))
			gradients[l].assign(gradients[l].add(gradients_update[l]))

	def update_weights(self,gradients,learningRate):
		for l in range(0,self.L):
			self.w[l].assign(self.w[l].add(gradients[l].multiply_by_constant(-learningRate)))

	def train(self,data,label,dataVal,labelVal,learningRate,maxIteration):
		N = len(data)
		errorIn = 0.0
		errorInMin = 1000000000000000000.0
		errorVal = 0.0
		errorValMin = 10000000000000000000.0
		wOpt = self.w
		iterationOptimal = 0

		dataNorms = compute_norms(data)
		maxNorm = max(dataNorms)
		self.initialize_weights(1/maxNorm)

		inputs = self.make_input()
		signals = self.make_signal()
		sensitivities = self.make_sensitivity()
		gradients = self.make_gradient()
		gradients_update = self.make_gradient()
		dataPoint = Vector(0,0.0)
		prediction = []
		for i in range(0,len(labelVal)):
			prediction.append(0.0)

		iteration = 1
		while iteration <= maxIteration:
			errorIn = 0.0
			self.clear_gradient(gradients)
			self.clear_gradient(gradients_update)
			for i in range(0,len(data)):
				dataPoint.assign(augment_one(data[i]))
				pointPrediction = self.forward_propagate(dataPoint,inputs,signals)
				self.back_propagate(inputs,sensitivities)
				errorIn += ((pointPrediction-label[i])**2)/N
				self.update_gradient(inputs,sensitivities,gradients,gradients_update,label[i],N)

			self.update_weights(learningRate,gradients)

			predictionVal = predict_internal(dataVal,inputs,signals)
			errorVal = regression_error(predictionVal,labelVal)
			if errorVal < errorValMin:
				wOpt = self.w
				iterationOptimal = iteration
				errorValMin = errorVal
			iteration += 1

		self.w = wOpt
		print("Optimal validation error is:")
		print(errorValMin)
		print("This occured at iteration:")
		print(iterationOptimal)

	def predict(self,dataTest):
		inputs = self.make_input()
		signals = self.make_signal()
		dataPoint = Vector(0,0.0)
		prediction = []
		for i in range(0,len(dataTest)):
			dataPoint.assign(augment_one(dataTest[i]))
			pointPrediction = self.forward_propagate(dataPoint,inputs,signals)
			prediction.append(sign(pointPrediction))
		return prediction
	

	def print_network(self):
		print("printing network \n")
		for i in range(0,self.L):
			self.w[i].print_matrix()


class Vector:
	def __init__(self,length,value):
		self.length = length
		self.values = []
		for i in range(0,self.length):
			self.values.append(value)

	def get(self,i):
		return self.values[i]

	def mod(self,i,value):
		self.values[i] = value

	def assign(self,vector):
		self.length = vector.length
		self.values = vector.values

	def concatenate(self,vector):
		result = Vector(self.length+vector.length,0.0)
		index = 0
		for i in range(0,self.length):
			result.mod(index,self.values[i])
			index += 1
		for i in range(0,vector.length):
			result.mod(index,vector.get(i))
			index += 1
		return result

	def add(self,vector):
		if self.length == vector.length:
			result = Vector(self.length,0.0)
			for i in range(0,self.length):
				result.mod(i,self.values[i]+vector.get(i))
			return result
		else:
			print("incorrect dimensions")
			return

	def multiply_by_constant(self,c):
		result = Vector(self.length,0.0)
		for i in range(0,self.length):
			result.mod(i,c*self.values[i])
		return result

	def multiply_by_matrix(self,matrix):
		if self.length != matrix.colDim:
			print("incorrect dimensions, aborting multiplication")
			return
		else:
			result = Vector(self.length,0.0)
			for i in range(0,matrix.rowDim):
				tmp = 0.0
				for j in range(0,matrix.colDim):
					tmp += matrix.get(i,j)*self.values[j]
				result.mod(i,tmp)
			return result

	def multiply_remove(self,matrix):
		if self.length != matrix.colDim:
			print("incorrect dimensions")
			return
		else:
			tmp = self.multiply_by_matrix(matrix)
			result = Vector(self.length-1,0.0)
			for i in range(1,self.length):
				result.mod(i-1,tmp.get(i))
			return result

	def multiply_componentwise(self,vector):
		if self.length != vector.length:
			print("incorrect dimensions")
			return
		else:
			result = Vector(self.length,0.0)
			for i in range(0,self.length):
				result.mod(i,vector.get(i)*self.values[i])
			return result

	def outer_product(self,vector):
		result = Matrix(vector.length,self.length,0.0)
		for i in range(0,vector.length):
			for j in range(0,self.length):
				result.mod(i,j,vector.get(i)*self.values[j])
		return result


	def print_vector(self):
		print(self.values)
		print("\n")


class Matrix:
	def __init__(self,rowDim,colDim,value):
		self.rowDim = rowDim
		self.colDim = colDim
		self.values = []
		for i in range(0,rowDim):
			self.values.append([])
			for j in range(0,colDim):
				self.values[i].append(value)

	def get(self,i,j):
		return self.values[i][j]

	def mod(self,i,j,value):
		self.values[i][j] = value

	def assign(self,matrix):
		self.rowDim = matrix.rowDim
		self.colDim = matrix.colDim
		self.values = matrix.values

	def transpose(self):
		result = Matrix(self.colDim,self.rowDim,0.0)
		for i in range(0,self.rowDim):
			for j in range(0,self.colDim):
				result.mod(j,i,self.values[i][j])
		return result

	def add(self,matrix):
		result = Matrix(self.rowDim,self.colDim,0.0)
		for i in range(0,self.rowDim):
			for j in range(0,self.colDim):
				result.mod(i,j,matrix.get(i,j)+self.values[i][j])
		return result

	def multiply_by_constant(self,c):
		result = Matrix(self.rowDim,self.colDim,0.0)
		for i in range(0,self.rowDim):
			for j in range(0,self.colDim):
				result.mod(i,j,c*self.values[i][j])
		return result

	def multiply_by_matrix(self,matrix):
		result = Matrix(matrix.rowDim,self.colDim,0.0)
		for i in range(0,matrix.rowDim):
			for j in range(0,self.colDim):
				tmp = 0.0
				for k in range(0,self.rowDim):
					tmp += matrix.get(i,k)*self.values[k][j]
				result.mod(i,j,tmp)
		return result

	def print_matrix(self):
		for i in range(0,self.rowDim):
			print(self.values[i])
		print "\n"


#Helper functions

def read_data(fileName):
	result = []
	with open(fileName) as file:
		filereader = csv.reader(file,delimiter=',')
		for row in filereader:
			result.append(row)
	for i in range(0,len(result)):
		for j in range(0,len(result[i])):
			result[i][j] = float(result[i][j])
	return result

def read_labels(fileName):
 	result = []
 	file = open(fileName)
 	for line in file:
 		result.append(line)
 	for i in range(0,len(result)):
 		result[i] = int(result[i])
 	return result

def compute_norms(data):
	result = []
	for i in range(0,len(data)):
		tmp = 0.0
		for j in range(0,len(data[i])):
			tmp += data[i][j]**2
		result.append(tmp**0.5)
	return result

def augment_one(dataPoint):
	one = Vector(1,1.0)
	arg = Vector(len(dataPoint),0.0)
	for i in range(0,len(dataPoint)):
		arg.mod(i,dataPoint[i])
	return one.concatenate(arg)

def sigmoid_map(vector):
	result = []
	for i in range(0,vector.length):
		result.append(math.tanh(vector.get(i)))
	return result

def inverse_signal(vector):
	result = Vector(vector.length()-1,0.0)
	for i in range(1,vector.length()):
		result.mod(i-1,1-vector.get(i)**2)
	return result

def sign(number):
	if number > 0: 
		return 1
	elif number < 0: 
		return -1
	else: 
		return 0

def regression_error(prediction,trueValue):
	if len(prediction) != len(trueValue):
		print("incorrect dimensions")
		return
	else:
		error = 0.0
		for i in range(0,len(trueValue)):
			error += (prediction[i]-trueValue[i])**2
		return error/len(trueValue)