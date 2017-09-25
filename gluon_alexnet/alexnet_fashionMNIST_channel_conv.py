from mxnet import gluon
from mxnet import ndarray as nd 
from mxnet import autograd
from mxnet import init 
import utils
from mxnet import image
from mxnet.gluon import nn

net = gluon.nn.Sequential()
with net.name_scope():
	# Stage 1
	net.add(nn.Conv2D(
		channels=96, kernel_size=11, strides=4, activation='relu'))
	net.add(nn.MaxPool2D(pool_size=3, strides=2))
	# Stage 2
	net.add(nn.Conv2D(
		channels=256, kernel_size=5, padding=2, activation='relu'))
	net.add(nn.MaxPool2D(pool_size=3, strides=2))
	# Stage 3
	net.add(nn.Conv2D(
		channels=384, kernel_size=3, padding=1, activation='relu'))
	net.add(nn.Conv2D(
		channels=384, kernel_size=3, padding=1, activation='relu'))
	net.add(nn.Conv2D(
		channels=256, kernel_size=3, padding=1, activation='relu'))
	net.add(nn.MaxPool2D(pool_size=3, strides=2))

	# Stage 4
	net.add(nn.Flatten())
	net.add(nn.Dense(4096, activation='relu'))
	net.add(nn.Dropout(.5))

	# Stage 5
	net.add(nn.Dense(4096, activation='relu'))
	net.add(nn.Dropout(.5))

	# Stage 6
	net.add(nn.Dense(10))
net_new = gluon.nn.Sequential()
with net_new.name_scope():
	# Stage 1
	net.add(nn.Conv3D(
		channels=96, kernel_size=(1,11,11), strides=(1,4,4), activation='relu'))
	net.add(nn.MaxPool3D(pool_size=(1,3,3),strides=(1,2,2)))
	# Stage 2
	net.add(nn.Conv3D(
		channels=32, kernel_size=(16,5,5), strides=(8,1,1),padding=(0,2,2), activation='relu'))
	net.add(nn.MaxPool3D(pool_size=(1,3,3),strides=(1,2,2)))
	# Stage 3
	ned.add(nn.Conv3D(
		channels=))
def transform(data, label):
	# Resize from 28 x 28 to 224 x 224
	data = image.imresize(data, 224, 224)
	return utils.transform_mnist(data, label)
batch_size = 64
train_data, test_data = utils.load_data_fashion_mnist(batch_size, transform)
ctx = utils.try_gpu()

net.initialize(ctx=ctx, init=init.Xavier())
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(
	net.collect_params(), 'sgd', {'learning_rate': 0.01})

for epoch in range(50):
	train_loss = 0.
	train_acc = 0.
	for data, label in train_data:
		label = label.as_in_context(ctx)
		with autograd.record():
			output = net(data.as_in_context(ctx))
			loss = softmax_cross_entropy(output, label)
		loss.backward()
		trainer.step(batch_size)

		train_loss += nd.mean(loss).asscalar()
		train_acc += utils.accuracy(output, label)

	test_acc = utils.evaluate_accuracy(test_data, net, ctx)
	print("Epoch %d. Loss: %f, Train acc: %f, Test acc: %f" % (epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))
	
