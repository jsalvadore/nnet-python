import machinery as mach

class nnet:
	def __init__(self,arch):
		self.L = len(arch)-1
		self.d = arch
		self.w = []
		for i in range(1,self.L+1):
			W = mach.Matrix(self.d[i-1]+1,self.d[i],0.0)
			self.w.append(W)
		print(len(self.w))

	def print_network(self):
		print("printing network \n")
		for i in range(0,self.L):
			self.w[i].print_matrix()

