import machinery as mach
import random as random

class nnet:
	def __init__(self,arch):
		self.L = len(arch)-1
		self.d = arch
		self.w = []
		for i in range(1,self.L+1):
			W = mach.Matrix(self.d[i-1]+1,self.d[i],0.0)
			self.w.append(W)

	def dim(self,k):
		return self.d[k]

	def get_weights(self,k):
		return self.w[k]

	def mod_weights(self,k,matrix):
		self.w[k].assign(matrix)

	def initialize_weights(self,sigma):
		for k in range(0,self.L):
			tmp = mach.Matrix(self.d[k]+1,self.d[k+1],0.0)
			for i in range(0,self.d[k]+1):
				for j in range(0,self.d[k+1]):
					tmp.mod(i,j,random.normalvariate(0.0,sigma))
			self.w[k].assign(tmp)


	def make_input(self):
		result = []
		for i in range(0,self.L+1):
			result.append(mach.Vector(self.d[i]+1,0.0))
		return result

	def make_signal(self):
		result = []
		for i in range(1,self.L+1):
			result.append(mach.Vector(self.d[i],0.0))
		return result

	def make_sensitivity(self):
		result = []
		for i in range(1,self.L+1):
			result.append(mach.Vector(self.d[i],0.0))
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
			inputs[l+1].assign(augment_one(simoid_map(signals[l])))
		return inputs[self.L].get(1)

	def back_propagate(self,inputs,sensitivities):
		sensitivities[self.L-1].assign(inverse_signal(inputs[self.L]))
		T = mach.Vector(0,0.0)
		for l in range(self.L-2,-1,-1)
			T.assign(inverse_signal(inputs[l+1]))
			sensitivities[l].assign(sensitivities[l+1].multiply_remove(self.w[l+1]).multiply_componentwise(T))

	def predict_internal(self, data, inputs, signals):
		result = mach.Vector(len(data))
		for i in range(0,len(data)):
			dataPoint = mach.Vector(0,0,0)
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

	def train(data,label,dataVal,labelVal,learningRate,maxIteration):
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
		dataPoint = mach.Vector(0,0.0)
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
		dataPoint = mach.Vector(0,0.0)
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


#Helper functions

def compute_norms(data):
	result = []
	for i in range(0,len(data)):
		tmp = 0.0
		for j in range(0,len(data[i])):
			tmp += data[i][j]**2
		result.append(tmp**0.5)
	return result

	pass
def augment_one(dataPoint):
	one = mach.Vector(1,1.0)
	arg = mach.Vector(len(dataPoint),0.0)
	for i in range(0,len(dataPoint)):
		arg.mod(i,dataPoint[i])
	return one.concatenate(arg)

def sigmoid_map(vector):
	result = []
	for i in range(0,len(vector)):
		result.append(math.tanh(vector.get(i)))
	return result

def inverse_signal(vector):
	result = mach.Vector(vector.length()-1,0.0)
	for i in range(1,vector.length()):
		result.mod(i-1,1-vector.get(i)**2)
	return result

def sign(number):
	if number > 0: 
		return 1
	else if number < 0: 
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

