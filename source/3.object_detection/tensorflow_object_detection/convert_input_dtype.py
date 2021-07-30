import onnx

model_path = "/home/barcelona/test/efdet/fire_efdet_d0.onnx"
model = onnx.load(model_path)

inputs = model.graph.input
outputs = model.graph.output
print(inputs)
print(outputs)
print("====================================================================================================================================")

for input in inputs:
    elem_type = input.type.tensor_type.elem_type
    print(elem_type)

    input.type.tensor_type.elem_type = 1

onnx.save(model, "/home/barcelona/test/efdet/float.onnx")