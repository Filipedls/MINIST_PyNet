class Layer:

	def __init__( self, type, has_weights):
		self.type = type
		self.has_weights = has_weights
		
	def __del__(self):
		class_name = self.__class__.__name__
		print class_name, " was destroyed"