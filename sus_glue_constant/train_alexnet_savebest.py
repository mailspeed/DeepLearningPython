# import the necessary packages
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from config import sus_glue_constant_config as config
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from pyimagesearch.preprocessing.patchpreprocessor import PatchPreprocessor
from pyimagesearch.preprocessing.meanpreprocessor import MeanPreprocessor
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from pyimagesearch.io.hdf5datasetgenerator import HDF5DatasetGenerator
from pyimagesearch.nn.conv.alexnet import AlexNet
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import argparse
import json
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=int, default=100,
	help="epochs to train")
args = vars(ap.parse_args())

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")

# load the RGB means of for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
# sp = SimplePreprocessor(227, 227)
# pp = PatchPreprocessor(227, 227)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

# initialize the training and validation datset generators
# trainGen = HDF5DatasetGenerator(os.path.abspath(config.TRAIN_HDF5), 128, aug=aug,
	# preprocessors=[pp, mp, iap], classes=2)
# valGen = HDF5DatasetGenerator(os.path.abspath(config.VAL_HDF5), 128,
	# preprocessors=[sp, mp, iap], classes=2)
trainGen = HDF5DatasetGenerator(os.path.abspath(config.TRAIN_HDF5), 128, aug=aug,
	preprocessors=[mp, iap], classes=2)
valGen = HDF5DatasetGenerator(os.path.abspath(config.VAL_HDF5), 128,
	preprocessors=[mp, iap], classes=2)
	
	
# initialize the optimizer
print("[INFO] compiling model...")
opt = Adam(lr=1e-3)
model = AlexNet.build(width=230, height=106, depth=3,
	classes=2, reg=0.0002)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# construct the set of callbacks to save only the *best* model to disk
# based on the validation loss
checkpoint = ModelCheckpoint(config.MODEL_PATH, monitor="val_loss",
	save_best_only=True, verbose=1)
path = os.path.sep.join([os.path.abspath(config.OUTPUT_PATH), "{}.png".format(
	os.getpid())])
callbacks = [checkpoint, TrainingMonitor(path)]

# train the network
print("[INFO] training network...")
model.fit_generator(
	trainGen.generator(),
	steps_per_epoch=trainGen.numImages // 128,
	validation_data=valGen.generator(),
	validation_steps=valGen.numImages // 128,
	epochs=args["epochs"],
	max_queue_size=128 * 2,
	callbacks=callbacks, verbose=1)

# save the model to file
print("[INFO] serializing model...")
model.save(config.MODEL_PATH, overwrite=True)

# close the HDF5 datasets
trainGen.close()
valGen.close()








