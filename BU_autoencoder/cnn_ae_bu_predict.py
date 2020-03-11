import cv2
import argparse
import os
import numpy as np
from imutils import paths
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

col = 500
row = 500
ch = 3
def normalization(X):
	return  X / 127.5 - 1

def inverse_normalization(X):
	return (X + 1.) / 2.

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default='test',
	help="path to dataset")
ap.add_argument("-m", "--model", default='model/model.model',
	help="path to pre-trained model")
args = vars(ap.parse_args())

# grab the list of images in the dataset 
print("[INFO] sampling images...")
imagePaths = np.array(list(paths.list_images(args["dataset"])))

# load the pre-trained model
print("[INFO] loading pre-trained model...")
model = load_model(args["model"])

for (i, imagePath) in enumerate(imagePaths):
	# load the images and do prediction
	print("[INFO] load image " + imagePath)	
	image_org = cv2.imread(imagePath)
	# load the image to fit the size of model input
	img = load_img(imagePath, target_size=(col, row))
	imgarray = img_to_array(img)
	
	org_imgs = []
	org_imgs.append(imgarray)
	data = np.array(org_imgs)
	data = data[:].astype(np.float32)
	# normalization and inverse_normalization is critical
	# data = normalization(data)
	data = data / 255.0
	# do the prediction
	image_gen = model.predict(data, batch_size=32)
	
	# image_gen = inverse_normalization(image_gen)
	plt.imshow(image_gen[0].reshape(col, row, ch))
	plt.savefig("image.png")
	plt.clf()
	plt.close()
	
	image_gen = np.array(image_gen[0] * 255, dtype = np.uint8)
	image_gen = cv2.resize(image_gen, (image_org.shape[1], image_org.shape[0]))
	
	# display original image and generated image
	cv2.imshow("org", image_org)
	cv2.imshow("gen", image_gen)
	
	fileName = imagePath.split(os.path.sep)[-1]
	cv2.imwrite(os.path.sep.join(["output",fileName]), image_gen)
	cv2.waitKey(0)
