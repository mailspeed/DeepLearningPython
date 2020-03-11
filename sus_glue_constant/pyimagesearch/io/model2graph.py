import os
import h5py
import tensorflow as tf
import keras.backend as Backend
    
from keras.models import load_model
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

class Model2Graph(object):
	@staticmethod
	def Convert(ModelPath, OutputPath = "Output", Graph = "Model.pb"):
		# Load the model from file
		Model = load_model(filepath = ModelPath)
		
		Backend.set_learning_phase(0)
		Session = Backend.get_session()

		OutputCount = len(Model.outputs)
		Temp = [None] * OutputCount
		NodeNames = [None] * OutputCount
		for i in range(OutputCount):
			NodeNames[i] = "output_node" + str(i)
			Temp[i] = tf.identity(Model.outputs[i], name = NodeNames[i])
		print("Output Nodes: {}".format(NodeNames))

		constant_graph = graph_util.convert_variables_to_constants(Session, Session.graph.as_graph_def(), NodeNames)    
		graph_io.write_graph(constant_graph, OutputPath, Graph, as_text = False)
		
		# Load the class labels from the model
		with h5py.File(ModelPath, "r") as File:
			if("ClassLabels" in File):
				print("[INFO] Save classlabels...")
				ClassLabels = File["ClassLabels"].value
				ClassLabels = [x.decode("ascii") for x in ClassLabels]
				with open(ModelPath.rsplit(os.path.sep, 1)[0] + os.path.sep + "Label" + ".txt", "w") as LabelFile:
					for Nr, Label in enumerate(ClassLabels):
						LabelFile.write(str(Nr) + " " + Label + "\n")
