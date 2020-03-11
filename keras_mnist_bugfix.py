# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import argparse
	
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

# grab the MNIST dataset (if this is your first time running this
# script, the download may take a minute -- then 55MB MNIST dataset
# will be downloaded)
print("[INFO] loading MNIST (full) dataset...")
''' METHOD ########1
try:
	dataset = datasets.fetch_mldata("MNIST Original")
except Exeption as ex:
	from six.moves import urllib
	from scipy.io import loadmat
	import os
	
	mnist_path = os.path.join(".", "datasets", "mnist-original.mat")
	
	# download dataset from github
	mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
	response = urllib.request.urlopen(mnist_alternative_url)
	with open(mnist_path, "wb") as f:
		content = response.read()
		f.write(content)
	
	mnist_raw = loadmat(mnist_path)
	dataset = {
		"data": mnist_raw["data"].T,
		"target": mnist_raw["label"][0],
		"COL_NAMES": ["label", "data"],
		"DESCR": "mldata.org dataset: mnist-original",
	}
	print("Done!")
'''

''' METHOD ########2
from scipy.io import loadmat
import os
mnist_path = os.path.join(".", "datasets", "mnist-original.mat")
mnist_raw = loadmat(mnist_path)
dataset = {
	"data": mnist_raw["data"].T,
	"target": mnist_raw["label"][0],
	"COL_NAMES": ["label", "data"],
	"DESCR": "mldata.org dataset: mnist-original",
}	
'''

''' METHOD ########3
import tensorflow.examples.tutorials.mnist.input_data as input_data
dataset = input_data.read_data_sets("MNIST")
'''

''' METHOD ########4
from shutil import copyfileobj
from six.moves import urllib
from sklearn.datasets.base import get_data_home
import os

def fetch_mnist(data_home=None):
    mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
    data_home = get_data_home(data_home=data_home)
    data_home = os.path.join(data_home, 'mldata')
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    mnist_save_path = os.path.join(data_home, "mnist-original.mat")
    if not os.path.exists(mnist_save_path):
        mnist_url = urllib.request.urlopen(mnist_alternative_url)
        with open(mnist_save_path, "wb") as matlab_file:
            copyfileobj(mnist_url, matlab_file)
'''

# METHOD ########5
# cause the original download web for method "fetch_mldat("MNIST Original")" (mddata.org) is down for good
# the script [ dataset = datasets.fetch_mldata("MNIST Original") ] wouldn't work at all
# alternatively, we use the script bellow. but first, you need to download the file "mnist-original.mat"
# from "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
# and copy it here ->  "working directory/files/mldata/mnist-original.mat" 
dataset = datasets.fetch_mldata("MNIST Original", transpose_data=True, data_home="files")

# scale the raw pixel intensities to the range [0, 1.0], then
# construct the training and testing splits
data = dataset.data.astype("float") / 255.0
(trainX, testX, trainY, testY) = train_test_split(data,
	dataset.target, test_size=0.25)

print(trainY)
print(testY)

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# define the 784-256-128-10 architecture using Keras
model = Sequential()
model.add(Dense(256, input_shape=(784,), activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(10, activation="softmax"))

# train the model using SGD
print("[INFO] training network...")
sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd,
	metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	epochs=100, batch_size=128)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1),
	target_names=[str(x) for x in lb.classes_]))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])


	
	
	
	
	
	
	
	
	