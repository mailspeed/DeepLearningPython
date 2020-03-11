import tensorflow as tf
if tf.test.gpu_device_name():
	print('Defaut GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
	print("Please install GPU version of TF")