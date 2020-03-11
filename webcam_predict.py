# import the necessary packages
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception # TensorFlow ONLY
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

import cv2
import numpy as np
import argparse


def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
	# initialize the dimensions of the image to be resized and
	# grab the image size
	dim = None
	(h, w) = image.shape[:2]

	# if both the width and height are None, then return the
	# original image
	if width is None and height is None:
		return image

	# check to see if the width is None
	if width is None:
		# calculate the ratio of the height and construct the
		# dimensions
		r = height / float(h)
		dim = (int(w * r), height)

	# otherwise, the height is None
	else:
		# calculate the ratio of the width and construct the
		# dimensions
		r = width / float(w)
		dim = (width, int(h * r))
		if height is not None:
			dim = (width, height)
		
	# resize the image
	resized = cv2.resize(image, dim, interpolation = inter)

	# return the resized image
	return resized

# construct the argument parse ad parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="vgg16",
	help="name of pre-trained network to use")
args = vars(ap.parse_args())

# define a dictionary that maps model names to their classes
# inside Keras
MODELS = {
	"vgg16": VGG16,
	"vgg19": VGG19,
	"inception": InceptionV3,
	"xception": Xception,	# TensorFlow ONLY
	"resnet": ResNet50
}

# ensure a valid model name was supplied via command line argument
if args["model"] not in MODELS.keys():
	raise AssertionError("The --model command line argument should "
		"be a key in the 'MODELS' dictionary")

# initialize the input image shape (224x224 pixels) along with
# the pre-processing function (this might need to be changed
# based on which model we use to classify our image)
inputShape = (224, 224)
preprocess = imagenet_utils.preprocess_input

# if we are using the InceptionV3 or Xception networks, then we
# need to set the input shape to (299x299) [rather than (224x224)]
# and use a different image processing function
if args["model"] in ("inception", "xception"):
	inputShape = (299, 299)
	preprocess = preprocess_input

# load our the network weights from disk (NOTE: if this is the
# first time you are running this script for a given network, the
# weights will need to be downloaded first -- depending on which
# network you are using, the weights can be 90-575MB, so be
# patient; the weights will be cached and subsequent runs of this
# script will be *much* faster)
print("[INFO] loading {}...".format(args["model"]))
Network = MODELS[args["model"]]
model = Network(weights="imagenet")

# define camera device using webcam
camera = cv2.VideoCapture(0)
# keep looping
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()

	# show image
	# cv2.imshow("capture", frame)
	
	# resize the frame and convert it to grayscale
	resized_frame = resize(frame, width=inputShape[0], height=inputShape[1])
	
	image = img_to_array(resized_frame)
	image = np.expand_dims(image, axis=0)
	# print(image.shape)
	
	# pre-process the image using th appropriate function based on the
	# model that has been loaded (i.e., mean subtraction, scaling, etc.)
	image = preprocess(image)
	
	# classify the image
	# print("[INFO] classifying image with '{}'...".format(args["model"]))
	preds = model.predict(image)
	P = imagenet_utils.decode_predictions(preds)

	# loop over the predictions and display the rank-5 predictions +
	# probabilities to our terminal
	# for (i, (imagenetID, label, prob)) in enumerate(P[0]):
		# print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))
		
	# loop the image via OpenCV, draw the top prediction on the image,
	# and display the image to our screen
	orig = frame.copy()
	(imagenetID, label, prob) = P[0][0]
	cv2.putText(orig, "Label: {}".format(label), (10, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
	cv2.imshow("Classification", orig)
	
	# if the 'q' key is pressed, stop the loop
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()