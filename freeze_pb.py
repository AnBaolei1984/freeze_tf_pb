import os
import sys
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.framework import graph_util

def freeze_graph(input_checkpoint, meta_path, output_node_names, output_graph):
    saver = tf.train.import_meta_graph(meta_path, clear_devices = True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        output_graph_def = graph_util.convert_variables_to_constants(  
            sess = sess,
            input_graph_def = input_graph_def,
            output_node_names = output_node_names.split(','))
 
        with tf.gfile.GFile(output_graph, 'wb') as f: 
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node)) 

if len(sys.argv) < 4:
    print('input error: input_checkpoint, output_node_names, output_graph')
    exit(1)

input_checkpoint = sys.argv[1]
output_node_names = sys.argv[2]
output_graph = sys.argv[3]
meta_path = input_checkpoint + '.meta'

reader = tf.train.NewCheckpointReader(input_checkpoint)

global_variables = reader.get_variable_to_shape_map()
for variable_name in global_variables:
    print(variable_name, global_variables[variable_name])

freeze_graph(input_checkpoint, meta_path, output_node_names, output_graph)