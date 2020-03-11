import os
import cv2
import numpy as np
from imutils import paths

def calcMean(imagePath, imgResize=100):
	img = cv2.imread(imagePath)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = cv2.resize(img, (imgResize, imgResize))
	mean = img.mean()
	return mean

def calcMax(imagePath, imgResize=100):
	img = cv2.imread(imagePath)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = cv2.resize(img, (imgResize, imgResize))
	val = img.max()
	return val
	
removeTh_bd = 25
removeTh_dvd = 20
# removeTh_cd = 20
removeTh_cd_max = 150
imageFolder = "D:\\201905_auto_capture - org\\cd"

print("[INFO] loading images from " + imageFolder + "...")
imagePaths = list(paths.list_images(imageFolder))

'''
for (i, imgPath) in enumerate(imagePaths):    
	m = calcMean(imgPath)
	if m < removeTh_cd:
		print("[INFO] {} | AVG = {} | REMOVED".format(imgPath, m))
		os.remove(imgPath)
'''
for (i, imgPath) in enumerate(imagePaths):    
	m = calcMax(imgPath)
	if m < removeTh_cd_max:
		print("[INFO] {} | MAX = {} | REMOVED".format(imgPath, m))
		os.remove(imgPath)