import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
from keras import backend as K
# from keras.utils import plot_model
from keras.callbacks import TensorBoard

from imutils import paths
from sklearn.model_selection import train_test_split
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from pyimagesearch.datasets.simpledatasetloader import SimpleDatasetLoader

import argparse
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default="dataset",
	help="path to the dataset")
ap.add_argument("-m", "--model", default="model.model",
	help="path to the dataset")
args = vars(ap.parse_args())

w = 500 #2208
h = 500 #1656
ch = 3

"""
	モデルの生成
"""
input_img = Input(shape=(w, h, ch))

# Encode_Conv1
x = Conv2D(32, (3,3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2,2), padding='same')(x)
# Encode_Conv2
x = Conv2D(16, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2), padding='same')(x)
# Encode_Conv3
x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2), padding='same')(x)

# Decode_Conv1
x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
# Decode_Conv2
x = Conv2D(16, (3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
# Decode_Conv3
x = Conv2D(32, (3,3), activation='relu')(x)
x = UpSampling2D((2,2))(x)
# Decode_Conv4
decoded = Conv2D(3, (3,3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# アーキテクチャの可視化
# plot_model(autoencoder, to_file="architecture.png")

"""
	データの読み込み
"""
# grab the list of images that we'll be describling
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the image processors
sp = SimplePreprocessor(w, h)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities
# to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) =  sdl.load(imagePaths, verbose=100)
data = data.astype("float") / 255.0

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(x_train, x_test, _, _) = train_test_split(data, labels,
	test_size=0.1, random_state=42)

x_train = np.reshape(x_train, (len(x_train), w, h, ch))
x_test = np.reshape(x_test, (len(x_test), w, h, ch))

epochs = 1000
batch_size = 32

autoencoder.fit(x_train, x_train,
				epochs=epochs,
				batch_size=batch_size,
				shuffle=True,
				validation_data=(x_test, x_test),
				callbacks=[TensorBoard(log_dir='./autoencoder')])

autoencoder.save(os.path.sep.join(["model",args["model"]]))

"""
	グラフへ可視化
"""
decoded_imgs = autoencoder.predict(x_test)

# 表示数
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
	# テスト画像の表示
	ax = plt.subplot(2, n, i + 1)
	plt.imshow(x_test[i].reshape(w, h, ch))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	
	#　変換された画像の表示
	ax = plt.subplot(2, n, i + n + 1)
	plt.imshow(decoded_imgs[i].reshape(w, h, ch))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
plt.show()













