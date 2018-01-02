<h1> How a model which is be exported from Fabrik can be further used for training/testing </h1>

<b>Step-1</b>:Go the actions section and select the 2nd button from left(An arrow showing North-East Dirction.
<br><b>Step-2</b>:A Drop Down List Should Be Appearing now, Select The Caffe Button(First Button).
<br><b>Step-3</b>: Rename the File to `model.prototxt`.
<br><b>Step-4</b>:Create a file titled 'solver.prototxt' with the following:
 ```
  net: "path/to/model.prototxt"    # path to the network
  test_iter: 200                   # how many mini-batches to test in each validation phase
  test_interval: 500               # how often do we call the test phase
  base_lr: 1e-5                    # base learning rate
  lr_policy: "step"                # step means to decrease lr after a number of iterations
  gamma: 0.1                       # ratio of decrement in each step
  stepsize: 5000                   # how often do we step (should be called step_interval)
  display: 20                      # how often do we print training loss
  max_iter: 450000                 # maximum amount of iterations
  momentum: 0.9
  weight_decay: 0.0005             # regularization!
  snapshot: 2000                   # taking snapshot is like saving your progress in a game
  snapshot_prefix: "path/to/model" # path to saved model
  solver_mode: GPU                 # choose CPU or GPU for processing, GPU is far faster, but CPU is more supported.
  ```
<br><b>Step-5</b>:Execute the following using caffe. ```caffe``` is the executable in the caffe folder (./build/tools/caffe).           ```solver.prototxt``` should be the path to the file we just created.
```
  caffe train \
    -gpu 0 \
    -solver solver.prototxt
  ```
                                      
Now, You are Done.
