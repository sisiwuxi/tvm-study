from tflite import Model
import pdb

buf = open('mobilenet_v1_1.0_224.tflite', 'rb').read()
pdb.set_trace()
model = Model.Model.GetRootAsModel(buf, 0)
subgraph = model.Subgraphs(0)
tensor_idx = 0
tensor = subgraph.Tensors(tensor_idx)
buffer_idx = tensor.Buffer()
buffer = model.Buffers(buffer_idx)
print(buffer.Data(0))
