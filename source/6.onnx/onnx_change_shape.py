import onnx

def change_input_dim(model):
	sym_batch_dim = '1'.encode('utf-8')
	sym_tensor_size = '512'.encode('utf-8')

	inputs = model.graph.input
	for input in inputs:
		dim1 = input.type.tensor_type.shape.dim[0]
		dim1.dim_param = sym_batch_dim

		dim2 = input.type.tensor_type.shape.dim[2]
		dim2.dim_param = sym_tensor_size

		dim3 = input.type.tensor_type.shape.dim[3]
		dim3.dim_param = sym_tensor_size

	output_node_names = ["detection_boxes", "detection_classes", "detection_scores"]
	outputs = model.graph.output
	for output in outputs:
		if output.name in output_node_names:
			dim1 = output.type.tensor_type.shape.dim[0]
			dim1.dim_param = sym_batch_dim

def apply(transform, infile, outfile):
    model = onnx.load(infile)
    transform(model)
    onnx.save(model, outfile)

model_path = "/data/Models/efficientdet/2021_09_27/export/model.onnx"
output_path = "/data/Models/efficientdet/2021_09_27/export/test.onnx"
apply(change_input_dim, model_path, output_path)