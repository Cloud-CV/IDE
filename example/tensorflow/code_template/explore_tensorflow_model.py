import tensorflow as tf
from google.protobuf import text_format
import sys

try:
    model_file = sys.argv[1]
except IndexError:
    print("Syntax: python explore_tensorflow_model.py <path/to/pbtxt>")

with open(model_file, 'r') as model_file:
    model_protobuf = text_format.Parse(model_file.read(),
                                       tf.GraphDef())

tf.import_graph_def(model_protobuf)

tensors = [n for n in tf.get_default_graph().as_graph_def().node]
print("Variables: ", [n.name for n in tensors if n.op == "VariableV2"])
print("Placeholders: ", [n.name for n in tensors if n.op == "Placeholder"])

# TODO: model usage sample when export model bug fixed
