from nnabla.utils.nnp_graph import NnpLoader
import numpy as np
from PIL import Image

nnpPath = 'model.nnp'
networkName = 'MainRuntime'

# read a .nnp file
nnp = NnpLoader(nnpPath)

# for nt in nnp.get_network_names():
	# print(nt)

# assume a graph 'graph_a' is in the nnp file
net = nnp.get_network(networkName, batch_size=1)

# prepare input of the graph
x = net.inputs['Input']
# prepare output of the graph
y = net.outputs['Sigmoid']

imgPath = 'test.png'
im = np.array(Image.open(imgPath))
x.d = np.reshape(im,(1,1,28,28))/255 #將im的形状改爲輸入形状: 張,色,高度,寛度
# Execute inference
y.forward(clear_buffer=True)

print(float(y.d))