import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from imutils import paths
import argparse
import cv2
import matplotlib.pylab as plt

def normalization(X):
	return  X / 127.5 - 1

def inverse_normalization(X):
	return (X + 1.) / 2.

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to pre-trained model")
ap.add_argument("-o", "--output", required=True,
	help="path to output predict result")
args = vars(ap.parse_args())

# grab the list of images in the dataset 
print("[INFO] sampling images...")
imagePaths = np.array(list(paths.list_images(args["dataset"])))

# then randomly sample indexes into the image paths list
# idxs = np.random.randint(0, len(imagePaths), size=(200,))
# imagePaths = imagePaths[idxs]

# load the pre-trained model
print("[INFO] loading pre-trained model...")
model = load_model(args["model"])

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


