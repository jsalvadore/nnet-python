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

	def add(self,vector):
		if self.length == vector.length:
			result = Vector(self.length,0.0)
			for i in range(0,self.length):
				result.mod(i,self.values[i]+vector.get(i))
			return result
		else:
			print("Attempted to add vectors of differing dimension")
			return

	def multiply_by_constant(self,c):
		result = Vector(self.length,0.0)
		for i in range(0,self.length):
			result.mod(i,c*self.values[i])
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
