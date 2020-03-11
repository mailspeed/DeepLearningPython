import cv2
import numpy as np

def weightCtr(x):
	# print(x)
	if np.count_nonzero(x) == 0:
		return int((x.shape[0] + 1) / 2)
	else:
		return int(np.flatnonzero(x).mean())

# load as greyscale
im = cv2.imread('test_cut.jpg', cv2.IMREAD_GRAYSCALE)

# invert
im = 255 - im

# print(im.shape)
# print(im.dtype)
# cv2.imshow("INV", im)
# cv2.waitKey(0)
im = cv2.threshold(im, 200, 255, cv2.THRESH_BINARY)[1]
im = cv2.dilate(im, np.ones((5, 30)))
im = cv2.erode(im, np.ones((5, 30)))
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
	cv2.circle(im, (marks[i][0], marks[i][1]), 4, (255, 0, 0), 1)
	cv2.line(im, (marks[i][0], marks[i][1]), (marks_base[i][0], marks_base[i][1]), (0, 255, 0), 1)

# draw max y diff lines
cv2.putText(im, str(maxDiff), (marks_base[maxIndex][0], 22), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
for i in range(0, 127):
	diff = abs(marks[i][1] - marks_base[i][1])
	if diff >= maxDiff:	
		cv2.arrowedLine(im, (marks_base[i][0], marks_base[i][1]), (marks[i][0], marks[i][1]), (0, 0, 255), 2)
	
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