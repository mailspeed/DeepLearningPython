# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.regularizers import l2
from keras import backend as K

class CNN6:
	@staticmethod
	def build(width, height, depth, classes, reg=0.0002):
		# initialize the model along with the input shape to be 
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1
		
		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1
		
		# Block #1: first CONV => RELU => POOL layer set
		model.add(Conv2D(32, (3, 3), strides=(1, 1),
			input_shape=inputShape, padding="same",
			kernel_regularizer=l2(reg)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		
		# Block #2: second CONV => RELU => POOL layer set
		model.add(Conv2D(32, (3, 3), strides=(1, 1),
			padding="same", kernel_regularizer=l2(reg)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(3, 3), stride=(2, 2)))
		
		# Block #3: third CONV => RELU => POOL layer set
		model.add(Conv2D(32, (3, 3), strides=(1, 1),
			padding="same", kernel_regularizer=l2(reg)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(3, 3), stride=(2, 2)))
		
		# Block #4: forth CONV => RELU => POOL layer set
		model.add(Conv2D(32, (3, 3), strides=(1, 1),
			padding="same", kernel_regularizer=l2(reg)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(3, 3), stride=(2, 2)))
		
		# Block #5: fifth CONV => RELU => POOL layer set
		model.add(Conv2D(32, (3, 3), strides=(1, 1),
			padding="same", kernel_regularizer=l2(reg)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(3, 3), stride=(2, 2)))
		
		# Block #6: sixth CONV => RELU => POOL layer set
		model.add(Conv2D(32, (3, 3), strides=(1, 1),
			padding="same", kernel_regularizer=l2(reg)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(3, 3), stride=(2, 2)))
		
		# Block #7: FC => RELU layers
		model.add(Flatten())
		model.add(Dense(classes, ????
