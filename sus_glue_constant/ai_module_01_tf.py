import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import cv2

import os
import zmq
import time
import datetime
import sys
import io
from imageio import imread
import base64

def normalization(X):
	return  X / 127.5 - 1

def inverse_normalization(X):
	return (X + 1.) / 2.

# Get the current absolute path
cur_path = os.path.abspath(".")
model_path = cur_path + "\\ParamFile\\alexnet_sus_glue_constant.model"

# load the pre-trained model
print("[INFO] loading pre-trained model...")
model = load_model(model_path)

url = 'tcp://127.0.0.1:5555'

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind(url)
print(datetime.datetime.now(), "[INFO] service started...", url)
while True:
	msg = socket.recv()
	img = imread(io.BytesIO(base64.b64decode(msg)))
	im = np.array(img)
	print(datetime.datetime.now(), "<< requst in")
	x.d = np.reshape(im, (1, imgDepth, imgHeight, imgWidth)) / 255.
	# Build network for inference

	
	
	
	
	print(datetime.datetime.now(), ">> predict result: {}".format(float(y.d)))
	# time.sleep(0.1)
	# socket.send_string("1")
	socket.send_string(str(float(y.d)))


for (i, imagePath) in enumerate(imagePaths):
	# load the images and do prediction
	print("[INFO] load image " + imagePath)	
	image_org = cv2.imread(imagePath)
	# load the image to fit the size of model input
	img = load_img(imagePath, target_size=(64, 64))
	imgarray = img_to_array(img)
	
	org_imgs = []
	org_imgs.append(imgarray)
	data = np.array(org_imgs)
	data = data[:].astype(np.float32)
	# normalization and inverse_normalization is critical
	data = normalization(data)
	# do the prediction
	image_gen = model.predict(data, batch_size=32)
	
	image_gen = inverse_normalization(image_gen)
	# plt.imshow(image_gen[0])
	# plt.savefig("image.png")
	# plt.clf()
	# plt.close()
	
	image_gen = np.array(image_gen[0] * 255, dtype = np.uint8)	
	image_gen = cv2.resize(image_gen, (image_org.shape[1], image_org.shape[0]))
	
	# display original image and generated image
	cv2.imshow("org", image_org)
	cv2.imshow("gen", image_gen)
	cv2.waitKey(0)


