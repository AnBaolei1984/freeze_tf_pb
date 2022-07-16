import os
import sys
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.framework import graph_util

def freeze_graph(input_checkpoint, meta_path, output_node_names, output_graph):
    saver = tf.train.import_meta_graph(meta_path, clear_devices = True)
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        output_graph_def = graph_util.convert_variables_to_constants(  
            sess = sess,
            input_graph_def = sess.graph_def,
            output_node_names = output_node_names.split(','))
 
        with tf.gfile.GFile(output_graph, 'wb') as f: 
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node)) 

if len(sys.argv) < 5:
    print('input error: input_checkpoint, meta_path, output_node_names, output_graph')
    exit(1)

input_checkpoint = sys.argv[1]
meta_path = sys.argv[2]
output_node_names = sys.argv[3]
output_graph = sys.argv[4]

# print all nodes in network
reader = tf.train.NewCheckpointReader(input_checkpoint)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print ('tensor name :' key)


freeze_graph(input_checkpoint, meta_path, output_node_names, output_graph)