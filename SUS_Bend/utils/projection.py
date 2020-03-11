import cv2
import numpy as np

def weightCtr(x):
	# print(x)
	if np.count_nonzero(x) == 0:
		return int((x.shape[0] + 1) / 2)
	else:
		return int(np.flatnonzero(x).mean())

# load as greyscale
im = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)

# invert
im = 255 - im

# print(im.shape)
# print(im.dtype)
# cv2.imshow("INV", im)
# cv2.waitKey(0)
im = cv2.threshold(im, 200, 255, cv2.THRESH_BINARY)[1]
# cv2.imshow("BIN", im)
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
	cv2.circle(im, (marks[i][0], marks[i][1]), 4, (128, 128, 128), 2)
	cv2.line(im, (marks[i][0], marks[i][1]), (marks_base[i][0], marks_base[i][1]), (64, 64, 64), 2)
	cv2.imshow("gen", im)
	cv2.waitKey(0)
for i in range(16, 128):
	cv2.circle(im, (marks[i][0], marks[i][1]), 4, (128, 128, 128), 2)
	cv2.line(im, (marks[i][0], marks[i][1]), (marks_base[i][0], marks_base[i][1]), (64, 64, 64), 2)
	cv2.imshow("gen", im)
	cv2.waitKey(0)
	
cv2.imwrite('result.png', im)
'''


# create output image same height as text, 500 px wide
m = np.max(proj)
h = im.shape[0]
result = np.zeros((h, proj.shape[0]))

# draw a line for each row
for col in range(proj.shape[0]):
	print(proj[col] / m)
	cv2.line(result, (col, 0), (col, int(proj[col] * h / m)), (255, 255, 255), 1)

# save result
cv2.imwrite('result.png', result)
'''