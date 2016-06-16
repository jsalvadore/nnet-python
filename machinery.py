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
			print("incorrect dimensions")
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
