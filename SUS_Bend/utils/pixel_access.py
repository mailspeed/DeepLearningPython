import numpy as np
from imutils import paths
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to dataset")
args = vars(ap.parse_args())

# grab the list of images in the dataset 
print("[INFO] sampling images...")
imagePaths = np.array(list(paths.list_images(args["dataset"])))

for (i, imagePath) in enumerate(imagePaths):
	# load the images and do prediction
	print("[INFO] load image " + imagePath)	
	img = cv2.imread(imagePath)
	img[:, 1133:1311] = 255
	cv2.imwrite(imagePath, img)