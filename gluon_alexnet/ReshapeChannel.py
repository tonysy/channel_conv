from mxnet.gluon.nn import Block, HybridBlock
from mxnet import nd
class ReshapeChannel(HybridBlock):
	"""Reshape the input for channel convolution
	
	Input shape:

	Output shape:

	"""
	def __init__(self, **kwargs):
		super(ReshapeChannel, self).__init__(**kwargs)

	def hybrid_forward(self, F, x):
		x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
		x = nd.expand_dims(x, axis=1)
		return x

	def __repr__(self):
		return self.__class_.__name__
