import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

saved_input_dir = "/data/Models/ssd_mobilenet_v2_etri/2021_09_24/convert/saved_model"
dirname = '/data/Models/ssd_mobilenet_v2_etri/2021_09_24/convert/'
filename = 'tf2_frozen_inference_graph.pb'

model = tf.saved_model.load(saved_input_dir)
graph_func = model.signatures['serving_default']
frozen_func = convert_variables_to_constants_v2(graph_func)
frozen_func.graph.as_graph_def()

tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir=dirname,
                  name=filename,
                  as_text=True)