import subprocess
import sys


# Get the command line arguments
model_file = ''
try:
    model_file = sys.argv[1]
except IndexError:
    print('Usage: python caffe_sample.py PATH_TO_MODEL')
    exit()

# Create solver.prototxt
with open('solver.prototxt', 'w') as file:
    file.write('net: "{}"\n'.format(model_file))          # path to the network
    file.write('test_iter: 200\n')                        # how many mini-batches to test in each validation phase
    file.write('test_interval: 500\n')                    # how often do we call the test phase
    file.write('base_lr: 1e-5\n')                         # base learning rate
    file.write('lr_policy: "step"\n')                     # step means to decrease lr after a number of iterations
    file.write('gamma: 0.1\n')                            # ratio of decrement in each step
    file.write('stepsize: 5000\n')                        # how often do we step (should be called step_interval)
    file.write('display: 20\n')                           # how often do we print training loss
    file.write('max_iter: 450000\n')                      # maximum amount of iterations
    file.write('momentum: 0.9\n')
    file.write('weight_decay: 0.0005\n')                  # regularization!
    file.write('snapshot: 2000\n')                        # taking snapshot is like saving your progress in a game
    file.write('snapshot_prefix: "model/caffe_sample"\n') # path to saved model
    file.write('solver_mode: GPU\n')                      # choose CPU or GPU for processing, GPU is far faster, but CPU is more supported.

# Train the model
# subsystem.run(['caffe', 'train', '-gpu', '0', '-solver', 'solver.prototxt'])
