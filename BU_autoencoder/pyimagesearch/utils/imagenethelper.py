# import the necessary packages
import numpy as np
import os

class ImageNetHelper:
	def __init__(self, config):
		# store the configuration object
		self.config = config
		
		# build the label mappings and validation blacklist
		self.labelMappings = self.buildClassLabels()
		self.valBlacklist = self.buildBlacklist()
		
	def buildClassLabels(self):
		# load the contents of the file that maps the WordNet IDs
		# to integers, then initialize the label mappings dictionary
		rows = open(slef.config.WORD_IDS).read().strip().split("\n")
		labelMappings = {}
		
		# loop over the labels
		for row in rows:
			# split the row into the WordNet ID, label integer, and
			# human readable label
			(wordID, label, hrLabel) = row.split(" ")
			
			# update the label mappings dictionary using the word ID
			# as the key and the labels as the value, subtracting '1'
			# from the label since MATLAB is one-indexed while Python
			# is zero-indexed
			labelMappings[wordID] = int(label) - 1
			
		# return the label mappings dictionary
		return labelMappings
		
	def buildBlacklist(self):
		# load the list of blacklisted image IDs and convert them to
		# a set
		rows = open(self.config.VAL_BLACKLIST).read()
		rows = set(rows.strip().split("\n")
		
		# return the blacklisted image IDs
		return rows
		
	def buildTrainingSet(self):
		# ----- config.TRAIN_LIST: train_cls.txt ---------------------------
		# n01440764/n01440764_10026 1
		# n01440764/n01440764_10027 2
		# n01440764/n01440764_10029 3
		# n01440764/n01440764_10040 4
		# n01440764/n01440764_10042 5
		# n01440764/n01440764_10043 6
		# n01440764/n01440764_10048 7
		# n01440764/n01440764_10066 8
		# n01440764/n01440764_10074 9
		# n01440764/n01440764_10095 10
		# -------------------------------------------------------------------
		
		# --- self.labelMappings: map_clsloc.txt ---------------------------
		# n02119789 1 kit_fox
		# n02100735 2 English_setter
		# n02110185 3 Siberian_husky
		# n02096294 4 Australian_terrier
		# n02102040 5 English_springer
		# n02066245 6 grey_whale
		# n02509815 7 lesser_panda
		# n02124075 8 Egyptian_cat
		# n02417914 9 ibex
		# n02123394 10 Persian_cat	
		# -------------------------------------------------------------------
	
		# load the contents of the training input file that lists
		# the partial image ID and image number, then initialize
		# the list of image paths and class labels
		rows = open(self.config.TRAIN_LIST).read().strip()
		rows = rows.split("\n")
		paths = []
		labels = []
	
		# loop over the rows in the input training file
		for row in rows:
			# break the row into the partial path and image
			# number (the image number is sequential and is
			# essentially useless to us)
			(partialPath, imageNum) = row.strip().split(" ")
			
			# construct the full path to the training image, then
			# grab the word ID from the path and use it to determine
			# the integer class label
			path = os.path.sep.join([self.config.IMAGES_PATH,
				"train", "{}.JPEG".format(partialPath)])
			wordID = partialPath.split("/")[0]
			label = self.labelMappings[wordID]
			
			# update the respective paths and label lists
			paths.append(path)
			labels.append(label)
		
		# return a tuple of image paths and associated integer class
		# labels
		return (np.array(paths), np.array(labels))
	
	def buildValidationSet(self):
		# -------- config.VAL_LIST: val.txt DEMO ----------------------------
		# ILSVRC2012_val_00000001 1
		# ILSVRC2012_val_00000002 2
		# ILSVRC2012_val_00000003 3
		# ILSVRC2012_val_00000004 4
		# ILSVRC2012_val_00000005 5
		# ILSVRC2012_val_00000006 6
		# ILSVRC2012_val_00000007 7
		# ILSVRC2012_val_00000008 8
		# ILSVRC2012_val_00000009 9
		# ILSVRC2012_val_00000010 10
		# -------------------------------------------------------------------
		
		# - cofig.VAL_LABELS: ILSVRC2015_clsloc_validation_ground_truth.txt -
		# 490
		# 361
		# 171
		# 822
		# 297
		# 482
		# 13
		# 704
		# 599
		# 164
		# -------------------------------------------------------------------
		
		# - self.valBlacklist: ILSVRC2015_clsloc_validation_blacklist.txt ---
		# 36
		# 50
		# 56
		# 103
		# 127
		# 195
		# 199
		# 226
		# 230
		# 235
		# -------------------------------------------------------------------

		# initialize the list of image paths and class labels
		paths = []
		labels = []
		
		# load the contents of the file that lists the partial
		# validation image filenames
		valFileNames = open(self.config.VAL_LIST).read()
		valFileNames = valFileNames.strip().split("\n")
		
		# load the contents of the file that contains the *actual*
		# ground-truth integer class labels for the validation set
		valLabels = open(self.config.VAL_LABELS).read()
		valLabels = valLabels.strip().split("\n")
		
		# loop over the validation data
		for (row, label) in zip(valFilenames, valLabels):
			# break the row into the partial path and image number
			(partialPath, imageNum) = row.strip().split(" ")
			
			# if the image number is in the blacklist set then we
			# should ignore this validation image
			if imageNum in self.valBlacklist:
				continue
			
			# construct the full path to the validation image, then
			# update the respective paths and the labels lists
			path = os.path.sep.join([self.config.IMAGES_PATH, "val",
				"{}.JPEG".format(partialPath)])
			paths.append(path)
			labels.append(int(label) - 1)
			
		# return a tuple of image paths and associated integer class
		# labels
		return (np.array(paths), np.array(labels))
			
		
			
			
			
			
			
			
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		