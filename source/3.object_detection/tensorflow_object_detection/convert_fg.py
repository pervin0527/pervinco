import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

dirname = '/home/barcelona/test/saved_model'

model = tf.saved_model.load(dirname)
graph_func = model.signatures['serving_default']
frozen_func = convert_variables_to_constants_v2(graph_func)
frozen_func.graph.as_graph_def()

tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir='/home/barcelona/test',
                  name='frozen_graph.pb',
                  as_text=True)