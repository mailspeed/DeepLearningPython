# import the necessary packages
from config import sus_glue_constant_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from pyimagesearch.io.hdf5datasetwriter import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import json
import cv2
import os

# grab the paths to the images
trainPaths = list(paths.list_images(config.IMAGES_PATH))
# convert to absolute path so the HDF5DatasetWriter would not be confused
trainPaths = [os.path.abspath(p) for p in trainPaths]

trainLabels = [p.split(os.path.sep)[-1].split("_")[0]
	for p in trainPaths]

le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)

# perform stratified sampling from the training set to build the
# testing split from the training data
split = train_test_split(trainPaths, trainLabels,
	test_size=config.TEST_SIZE, stratify=trainLabels,
	random_state=42)
(trainPaths, testPaths, trainLabels, testLabels) = split

# perform another stratified sampling, this time to build the
# validation data
split = train_test_split(trainPaths, trainLabels,
	test_size=config.VAL_SIZE, stratify=trainLabels,
	random_state=42)
(trainPaths, valPaths, trainLabels, valLabels) = split

# construct a list pairing the training, validation, and testing
# image paths along with their corresponding labels and output HDF5
# files
datasets = [
	("train", trainPaths, trainLabels, config.TRAIN_HDF5),
	("val", valPaths, valLabels, config.VAL_HDF5),
	("test", testPaths, testLabels, config.TEST_HDF5)]

# initialize the image preprocessor and the lists of RGB channel
# averages
aap = AspectAwarePreprocessor(256, 256)
(R, G, B) = ([], [], [])

# define image width(as 'w') and height(as 'h')
w = 230
h = 106

# loop over the dataset tuples
for (dType, paths, labels, outputPath) in datasets:
	# create HDF5 writer
	print("[INFO] building {}...".format(outputPath))
	writer = HDF5DatasetWriter((len(paths), h, w, 3), os.path.abspath(outputPath))
	
	# initialize the progress bar
	widgets = ["Building Dataset: ", progressbar.Percentage(), " ",
		progressbar.Bar(), " ", progressbar.ETA()]
	pbar = progressbar.ProgressBar(maxval=len(paths),
		widgets=widgets).start()
	
	# loop over the image paths
	for (i, (path, label)) in enumerate(zip(paths, labels)):
		# load the image and process it
		image = cv2.imread(path)
		# image = aap.preprocess(image)
		
		# if we are building the training dataset, then compute the
		# mean of each channel in the image, then update the
		# respective lists
		if dType == "train":
			(b, g, r) = cv2.mean(image)[:3]
			R.append(r)
			G.append(g)
			B.append(b)
			
		# add the image and label # to the HDF5 dataset
		writer.add([image], [label])
		pbar.update(i)
		
	# close the HDF5 writer
	pbar.finish()
	writer.close()
	
# construct a dictionary of averages, then serialize the means to a
# JSON file
print("[INFO] serializing means...")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open(config.DATASET_MEAN, "w")
f.write(json.dumps(D))
f.close()






















