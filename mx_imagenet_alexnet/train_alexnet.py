# import the necessary packages
from config import imagenet_alexnet_config as config
from pyimagesearch.nn.mxconv import MxAlexNet
import mxnet as mx
import argparse
import logging
import json
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True,
	help="path to output checkpoint directory")
ap.add_argument("-p", "--prefix", required=True,
	help="name of model prefix")
ap.add_argument("-s", "--start_epoch", type=int, default=0,
	help="epoch to restart training at")
args = vars(ap.parse_args())