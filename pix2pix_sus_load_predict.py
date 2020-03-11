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

def weightCtr(x):
	if np.count_nonzero(x) == 0:
		return int((x.shape[0] + 1) / 2)
	else:
		return int(np.flatnonzero(x).mean())

def drawMarks(org, gen):
	# load as greyscale
	# im = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
	im = cv2.cvtColor(gen, cv2.COLOR_BGR2GRAY)
	print(im.shape)
	# invert
	im = 255 - im
	# do binariziation
	im = cv2.threshold(im, 200, 255, cv2.THRESH_BINARY)[1]
	# calculate vertical projection
	proj = np.sum(im, 0)

	# search for key points
	m = np.max(proj)
	proj = proj / m
	th = 0.2

	x0 = 0
	y0 = 0
	x1 = 0
	y1 = 0
	x2 = 0
	y2 = 0
	x3 = 0
	y3 = 0

	p = 0
	for col in range(proj.shape[0]):
		if col >= 1000 and col <= 1400:
			continue
			
		if p == 0:
			if proj[col] >= th:
				x0 = col
				y0 = weightCtr(im[:, col])
				p = 1
		elif p == 1:
			if proj[col] < th:
				if col > 0:
					x1 = col - 1
					y1 = weightCtr(im[:, col - 1])
				else:
					x1 = col
					y1 = weightCtr(im[:, col])
				p = 2
		elif p == 2:
			if proj[col] >= th:
				x2 = col
				y2 = weightCtr(im[:, col])
				p = 3
		elif p == 3:
			if proj[col] < th:
				if col > 0:
					x3 = col - 1
					y3 = weightCtr(im[:, col - 1])
				else:
					x3 = col
					y3 = weightCtr(im[:, col])
				p = 4

	print(x0, y0, x1, y1, x2, y2, x3, y3)
	# cv2.line(im, (x0, y0), (x1, y1), (128, 128, 128), 5)
	# cv2.line(im, (x2, y2), (x3, y3), (128, 128, 128), 5)

	marks = np.zeros(shape=(128, 2), dtype=np.integer)
	marks[0] = [x0, y0]
	marks[15] = [x1, y1]
	marks[16] = [x2, y2]
	marks[127] = [x3, y3]

	marks_base = np.zeros(shape=(128, 2), dtype=np.integer)
	marks_base[0] = [x0, y0]
	marks_base[15] = [x1, y1]
	marks_base[16] = [x2, y2]
	marks_base[127] = [x3, y3]

	step = (x1 - x0) / 15
	stepY = (y1 - y0) / 15
	for i in range(1, 15):
		newX = x0 + int(i * step)
		marks[i] = [newX, weightCtr(im[:, newX])]
		marks_base[i] = [newX, y0 + int(i * stepY)]
		# print(marks[i])
		# cv2.line(im, (marks[i-1][0], marks[i-1][1]), (marks[i][0], marks[i][1]), (128, 128, 128), 5)
		
	step = (x3 - x2) / 111
	stepY = (y3 - y2) / 111
	for i in range(17, 127):
		newX = x2 + int((i-16) * step)
		marks[i] = [newX, weightCtr(im[:, newX])]
		marks_base[i] = [newX, y2 + int(i * stepY)]
		# print(marks[i])
		# cv2.line(im, (marks[i-1][0], marks[i-1][1]), (marks[i][0], marks[i][1]), (128, 128, 128), 5)

	for i in range(0, 16):
		cv2.circle(org, (marks[i][0], marks[i][1]), 4, (255, 0, 0), 2)
		cv2.line(org, (marks[i][0], marks[i][1]), (marks_base[i][0], marks_base[i][1]), (0, 255, 0), 2)
		# cv2.imshow("ORG", org)
		# cv2.imshow("GEN", gen)
		# cv2.waitKey(0)
	for i in range(16, 128):
		if i >= 96 and i <= 112:
			continue
		cv2.circle(org, (marks[i][0], marks[i][1]), 4, (255, 0, 0), 2)
		cv2.line(org, (marks[i][0], marks[i][1]), (marks_base[i][0], marks_base[i][1]), (0, 255, 0), 2)
		# cv2.imshow("ORG", org)
		# cv2.imshow("GEN", gen)
		# cv2.waitKey(0)
		
	cv2.imshow("ORG", org)
	cv2.imshow("GEN", gen)
	cv2.waitKey(0)
	# cv2.imwrite('result.png', org)

def drawMarks_Cut(org, gen):
	# load as greyscale
	# im = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
	im = cv2.cvtColor(gen, cv2.COLOR_BGR2GRAY)

	# invert
	im = 255 - im

	# print(im.shape)
	# print(im.dtype)
	# cv2.imshow("INV", im)
	# cv2.waitKey(0)
	im = cv2.threshold(im, 180, 255, cv2.THRESH_BINARY)[1]
	# cv2.imshow("BIN", im)
	im = cv2.dilate(im, np.ones((5, 30)))
	im = cv2.erode(im, np.ones((5, 30)))
	# cv2.imshow("ERODE", im)
	# cv2.waitKey(0)


	# calculate vertical projection
	proj = np.sum(im, 0)

	# search for key points
	m = np.max(proj)
	proj = proj / m
	th = 0.2

	x0 = 0
	y0 = 0
	x1 = 0
	y1 = 0
	x2 = 0
	y2 = 0
	x3 = 0
	y3 = 0
	x4 = 0
	y4 = 0
	x5 = 0
	y5 = 0

	p = 0
	for col in range(proj.shape[0]):	
		if p == 0:
			if proj[col] >= th:
				x0 = col
				y0 = weightCtr(im[:, col])
				p = 1
				
		elif p == 1:
			if proj[col] < th:
				if col > 0:
					x1 = col - 1
					y1 = weightCtr(im[:, col - 1])
				else:
					x1 = col
					y1 = weightCtr(im[:, col])
				
				p = 2
				
		elif p == 2:
			if proj[col] >= th:
				x2 = col
				y2 = weightCtr(im[:, col])
				p = 3
				
		elif p == 3:
			if proj[col] < th:
				if col > 0:
					x3 = col - 1
					y3 = weightCtr(im[:, col - 1])
				else:
					x3 = col
					y3 = weightCtr(im[:, col])
				p = 4
				
		elif p == 4:
			if proj[col] >= th:
				x4 = col
				y4 = weightCtr(im[:, col])
				p = 5
				
		elif p == 5:
			if proj[col] < th:
				if col > 0:
					x5 = col - 1
					y5 = weightCtr(im[:, col - 1])
				else:
					x5 = col
					y5 = weightCtr(im[:, col])
				
				p = 6

	print(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5)
	# cv2.line(im, (x0, y0), (x1, y1), (128, 128, 128), 5)
	# cv2.line(im, (x2, y2), (x3, y3), (128, 128, 128), 5)

	marks = np.zeros(shape=(127, 2), dtype=np.integer)
	marks[0] = [x0, y0]
	marks[12] = [x1, y1]
	marks[13] = [x2, y2]
	marks[108] = [x3, y3]
	marks[109] = [x4, y4]
	marks[126] = [x5, y5]

	marks_base = np.zeros(shape=(127, 2), dtype=np.integer)
	marks_base[0] = [x0, y0]
	marks_base[12] = [x1, y1]
	marks_base[13] = [x2, y2]
	marks_base[126] = [x5, y5]

	# calc marks of seg 1
	stepX = (x1 - x0) / 12
	stepY = 0
	if (x1 - x0) > 0:
		stepY = (y1 - y0) / (x1 - x0)
	for i in range(1, 12):
		newX = x0 + int(i * stepX)
		marks[i] = [newX, weightCtr(im[:, newX])]
		marks_base[i] = [newX, y0 + int((newX - x0) * stepY)]
		# print(marks[i])
		# cv2.line(im, (marks[i-1][0], marks[i-1][1]), (marks[i][0], marks[i][1]), (128, 128, 128), 5)
	
	# calc marks of seg 2
	stepX = (x3 - x2) / 95
	stepY = 0
	if (x5 - x2) > 0:
		stepY = (y5 - y2) / (x5 - x2)
	for i in range(14, 108):
		newX = x2 + int((i-13) * stepX)
		marks[i] = [newX, weightCtr(im[:, newX])]
		marks_base[i] = [newX, y2 + int((newX-x2) * stepY)]
		# print(marks[i])
		# cv2.line(im, (marks[i-1][0], marks[i-1][1]), (marks[i][0], marks[i][1]), (128, 128, 128), 5)

	marks_base[108] = [x3, y2 + int((x3 - x2) * stepY)]
	
	# calc marks of seg 3
	marks_base[109] = [x4, y2 + int((x4 - x2) * stepY)]
	
	stepX = (x5 - x4) / 17
	for i in range(110, 126):
		newX = x4 + int((i-109) * stepX)
		marks[i] = [newX, weightCtr(im[:, newX])]
		marks_base[i] = [newX, y2 + int((newX - x2) * stepY)]
		# print(marks[i])
		# cv2.line(im, (marks[i-1][0], marks[i-1][1]), (marks[i][0], marks[i][1]), (128, 128, 128), 5)
	
	# draw marking lines and find the maximum y diff
	maxDiff = 0
	maxIndex = 0
	for i in range(0, 127):
		diff = abs(marks[i][1] - marks_base[i][1])
		if diff > maxDiff:
			maxDiff = diff
			maxIndex = i
		cv2.circle(org, (marks[i][0], marks[i][1]), 4, (255, 0, 0), 1)
		cv2.line(org, (marks[i][0], marks[i][1]), (marks_base[i][0], marks_base[i][1]), (0, 255, 0), 1)
	
	# draw max y diff lines
	cv2.putText(org, str(maxDiff), (marks_base[maxIndex][0], 22), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
	for i in range(0, 127):
		diff = abs(marks[i][1] - marks_base[i][1])
		if diff >= maxDiff:	
			cv2.arrowedLine(org, (marks_base[i][0], marks_base[i][1]), (marks[i][0], marks[i][1]), (0, 0, 255), 2)
	
	cv2.imshow("ORG", org)
	cv2.imshow("GEN", gen)
	cv2.waitKey(0)

#====================== MAIN START ============================
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to pre-trained model")
args = vars(ap.parse_args())

# col = 256
# row = 256
col = 1024
row = 64

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
	img = load_img(imagePath, target_size=(row, col))
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
	# cv2.imshow("org", image_org)
	# cv2.imshow("gen", image_gen)
	# cv2.waitKey(0)
	drawMarks_Cut(image_org, image_gen)
	# cv2.imwrite('test.jpg', image_gen)

