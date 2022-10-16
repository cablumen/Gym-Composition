import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from Architectures import Architectures

dir_path = os.path.dirname(os.path.abspath(__file__))
param_file = os.path.join(dir_path, "params.txt")

if os.path.exists(param_file):
    os.remove(param_file)

write_file = open(param_file, "w")

for architecture in Architectures:
    name = architecture.name
    model = architecture.value
    model_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    write_file.write(name + ": " + str(model_params) +"\n")

write_file.close()
