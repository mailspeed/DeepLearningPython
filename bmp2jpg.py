import os
from PIL import Image
from imutils import paths

imagefolder = "CD-JPEG"
imagePaths = list(paths.list_images(imagefolder))

for imgPath in imagePaths:
	print(imgPath)
	img = Image.open(imgPath)
	newPath = imgPath.split('.')[0] + ".jpg"
	img.save(newPath, 'jpeg')
