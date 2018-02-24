import keras
import numpy as np
import torch
from torch.autograd import Variable
from layers_import_process import pytorch_to_keras

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-input_file', action="store",
                    dest='input_file', type=str, default='model.json')
parser.add_argument('-output_file', action="store",
                    dest='output_file', type=str, default='model.pbtxt')

args = parser.parse_args()
model_dir = args.input_file
output_dir = args.output_file


model = torch.load(model_dir)
for m in model.modules():
    m.training = False

dummy_model = keras.models.Sequential()

input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
input_var = Variable(torch.FloatTensor(input_np))
output = model(input_var)

keras_model = pytorch_to_keras((3, 224, 224,), output)
json_string = keras_model.to_json()

with open(output_dir, "w") as f:
    f.write(json_string)
