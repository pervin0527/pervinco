import onnx

model_path = "/home/barcelona/test/model/efdet/float.onnx"
model = onnx.load(model_path)

# inputs = model.graph.input
# outputs = model.graph.output
# print(inputs)
# print(outputs)
# print("====================================================================================================================================")

# for input in inputs:
#     elem_type = input.type.tensor_type.elem_type
#     print(elem_type)

#     input.type.tensor_type.elem_type = 1

# onnx.save(model, "/home/barcelona/test/efdet/float.onnx")

output =[node.name for node in model.graph.output]
print(output)

endpoint_names = ['serving_default_images:0_dequant']
for i in range(len(model.graph.node)):
    for j in range(len(model.graph.node[i].input)):
        if model.graph.node[i].input[j] in endpoint_names:
            print(model.graph.node[i].name)
            print(model.graph.node[i].input)
            print(model.graph.node[i].dtype)
            print(model.graph.node[i].output)