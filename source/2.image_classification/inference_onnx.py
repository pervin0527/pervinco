import onnx

MODEL_PATH = "/data/Models/model.onnx"
model = onnx.load(MODEL_PATH)

print(model)