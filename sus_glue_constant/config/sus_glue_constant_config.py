# define the paths to the images directory
IMAGES_PATH = "datasets/sus_glue_constant/train"

# since we do not have validation data or access to the testing
# labels we need to take a number of images from the training
# data and use them instead
NUM_CLASSES = 2
# NUM_VAL_IMAGES = 1250 * NUM_CLASSES
# NUM_TEST_IMAGES = 1250 * NUM_CLASSES
TEST_SIZE = 0.05
VAL_SIZE = 0.1

# define the path to the output training, validation, and testing
# HDF5 files
TRAIN_HDF5 = "datasets/sus_glue_constant/hdf5/train.hdf5"
VAL_HDF5 = "datasets/sus_glue_constant/hdf5/val.hdf5"
TEST_HDF5 = "datasets/sus_glue_constant/hdf5/test.hdf5"

# path to the output model file
MODEL_PATH = "output/alexnet_sus_glue_constant.model"

# define the path to the dataset mean
DATASET_MEAN = "output/sus_glue_constant_mean.json"

# define the path to the output directory used for storing plots,
# classfication reports, etc.
OUTPUT_PATH = "output"