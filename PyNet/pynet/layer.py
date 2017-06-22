class Layer:

	def __init__( self, type):
		self.type = type
		
	def __del__(self):
		class_name = self.__class__.__name__
		print class_name, " was destroyed"