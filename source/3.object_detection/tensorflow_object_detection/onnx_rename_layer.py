import onnx

onnx_model = onnx.load("/home/barcelona/test/test.onnx")

layer_names = ['serving_default_images:0']

for i in range(len(onnx_model.graph.node)):
	for j in range(len(onnx_model.graph.node[i].input)):
		if onnx_model.graph.node[i].input[j] in layer_names:
			print('-'*60)
			print(onnx_model.graph.node[i].name)
			print(onnx_model.graph.node[i].input)
			print(onnx_model.graph.node[i].output)

			onnx_model.graph.node[i].input[j] = "input_2"

	for j in range(len(onnx_model.graph.node[i].output)):
		if onnx_model.graph.node[i].output[j] in layer_names:
			print('-'*60)
			print(onnx_model.graph.node[i].name)
			print(onnx_model.graph.node[i].input)
			print(onnx_model.graph.node[i].output)

			onnx_model.graph.node[i].output[j] = "input_2"

for i in range(len(onnx_model.graph.input)):
	if onnx_model.graph.input[i].name in layer_names:
		print('-'*60)
		print(onnx_model.graph.input[i])
		onnx_model.graph.input[i].name = "input_2"

for i in range(len(onnx_model.graph.output)):
	if onnx_model.graph.output[i].name in layer_names:
		print('-'*60)
		print(onnx_model.graph.output[i])
		onnx_model.graph.output[i].name = "input_2"


layer_names = ['StatefulPartitionedCall:31']

for i in range(len(onnx_model.graph.node)):
	for j in range(len(onnx_model.graph.node[i].input)):
		if onnx_model.graph.node[i].input[j] in layer_names:
			print('-'*60)
			print(onnx_model.graph.node[i].name)
			print(onnx_model.graph.node[i].input)
			print(onnx_model.graph.node[i].output)

			onnx_model.graph.node[i].input[j] = "Identity_locations"

	for j in range(len(onnx_model.graph.node[i].output)):
		if onnx_model.graph.node[i].output[j] in layer_names:
			print('-'*60)
			print(onnx_model.graph.node[i].name)
			print(onnx_model.graph.node[i].input)
			print(onnx_model.graph.node[i].output)

			onnx_model.graph.node[i].output[j] = "Identity_locations"

for i in range(len(onnx_model.graph.input)):
	if onnx_model.graph.input[i].name in layer_names:
		print('-'*60)
		print(onnx_model.graph.input[i])
		onnx_model.graph.input[i].name = "Identity_locations"

for i in range(len(onnx_model.graph.output)):
	if onnx_model.graph.output[i].name in layer_names:
		print('-'*60)
		print(onnx_model.graph.output[i])
		onnx_model.graph.output[i].name = "Identity_locations"


layer_names = ['StatefulPartitionedCall:32']

for i in range(len(onnx_model.graph.node)):
	for j in range(len(onnx_model.graph.node[i].input)):
		if onnx_model.graph.node[i].input[j] in layer_names:
			print('-'*60)
			print(onnx_model.graph.node[i].name)
			print(onnx_model.graph.node[i].input)
			print(onnx_model.graph.node[i].output)

			onnx_model.graph.node[i].input[j] = "Identity_classes"

	for j in range(len(onnx_model.graph.node[i].output)):
		if onnx_model.graph.node[i].output[j] in layer_names:
			print('-'*60)
			print(onnx_model.graph.node[i].name)
			print(onnx_model.graph.node[i].input)
			print(onnx_model.graph.node[i].output)

			onnx_model.graph.node[i].output[j] = "Identity_classes"

for i in range(len(onnx_model.graph.input)):
	if onnx_model.graph.input[i].name in layer_names:
		print('-'*60)
		print(onnx_model.graph.input[i])
		onnx_model.graph.input[i].name = "Identity_classes"

for i in range(len(onnx_model.graph.output)):
	if onnx_model.graph.output[i].name in layer_names:
		print('-'*60)
		print(onnx_model.graph.output[i])
		onnx_model.graph.output[i].name = "Identity_classes"


layer_names = ['StatefulPartitionedCall:33']

for i in range(len(onnx_model.graph.node)):
	for j in range(len(onnx_model.graph.node[i].input)):
		if onnx_model.graph.node[i].input[j] in layer_names:
			print('-'*60)
			print(onnx_model.graph.node[i].name)
			print(onnx_model.graph.node[i].input)
			print(onnx_model.graph.node[i].output)

			onnx_model.graph.node[i].input[j] = "Identity_scores"

	for j in range(len(onnx_model.graph.node[i].output)):
		if onnx_model.graph.node[i].output[j] in layer_names:
			print('-'*60)
			print(onnx_model.graph.node[i].name)
			print(onnx_model.graph.node[i].input)
			print(onnx_model.graph.node[i].output)

			onnx_model.graph.node[i].output[j] = "Identity_scores"

for i in range(len(onnx_model.graph.input)):
	if onnx_model.graph.input[i].name in layer_names:
		print('-'*60)
		print(onnx_model.graph.input[i])
		onnx_model.graph.input[i].name = "Identity_scores"

for i in range(len(onnx_model.graph.output)):
	if onnx_model.graph.output[i].name in layer_names:
		print('-'*60)
		print(onnx_model.graph.output[i])
		onnx_model.graph.output[i].name = "Identity_scores"


layer_names = ['StatefulPartitionedCall:34']

for i in range(len(onnx_model.graph.node)):
	for j in range(len(onnx_model.graph.node[i].input)):
		if onnx_model.graph.node[i].input[j] in layer_names:
			print('-'*60)
			print(onnx_model.graph.node[i].name)
			print(onnx_model.graph.node[i].input)
			print(onnx_model.graph.node[i].output)

			onnx_model.graph.node[i].input[j] = "Identity_box_num"

	for j in range(len(onnx_model.graph.node[i].output)):
		if onnx_model.graph.node[i].output[j] in layer_names:
			print('-'*60)
			print(onnx_model.graph.node[i].name)
			print(onnx_model.graph.node[i].input)
			print(onnx_model.graph.node[i].output)

			onnx_model.graph.node[i].output[j] = "Identity_box_num"

for i in range(len(onnx_model.graph.input)):
	if onnx_model.graph.input[i].name in layer_names:
		print('-'*60)
		print(onnx_model.graph.input[i])
		onnx_model.graph.input[i].name = "Identity_box_num"

for i in range(len(onnx_model.graph.output)):
	if onnx_model.graph.output[i].name in layer_names:
		print('-'*60)
		print(onnx_model.graph.output[i])
		onnx_model.graph.output[i].name = "Identity_box_num"

onnx.save(onnx_model, "/home/barcelona/test/test_rename.onnx")