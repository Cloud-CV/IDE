# Layers supported in Caffe, Keras, and Tensorflow

###### The columns Caffe, Keras, and Tensorflow show which layers are supported in those libraries.
###### The columns Fabrik Caffe, Fabrik Keras, and Fabrik Tensorflow show which layers are currently supported by Fabrik in those libraries.


Below are tables showing which layers are supported by Caffe, Keras, and Tensorflow:
 ### Core Layers
| Layer                     | Caffe         | Keras        | Tensorflow  | Fabrik Caffe  | Fabrik Keras    | Fabrik Tensorflow  |
| :-----------------------: | :-----------: | :----------: | :---------: | :-----------: | :--------------: | :---------------: | 
| Activation                | ×             | √            | √           | ×             | √                | √                 | 
| ActivityRegularization    | ×             | √            | √           | ×             | √                | ×                 |
| Inner Product             | √             | √ (Dense)     | √           | √             | √                | ×                 |
| Dropout                   | √             | √            | √           | √             | √                | √                 |
| Flatten                   | √             | √            | √           | √             | √                | ×                 |
| Lambda                    | ×             | √            | √           | ×             | ×                | ×                 |
| Masking                   | √             | √            | √           | ×             | √                | ×                 |
| Permute                   | ×             | √            | √           | ×             | √                | ×                 |
| Repeat Vector             | ×             | √            | √           | ×             | √                | √                 |
| Reshape                   | √             | √            | √           | √             | √                | ×                 |
| Spatial Dropout 1D        | ×             | √            | √           | ×             | ×                | ×                 |
| Spatial Dropout 2D        | ×             | √            | √           | ×             | ×                | ×                 |
| Spatial Dropout 3D        | ×             | √            | √           | ×             | ×                | ×                 |
 #### Convolutional Layers
| Layer                     | Caffe         | Keras        | Tensorflow  | Fabrik Caffe  | Fabrik Keras    | Fabrik Tensorflow  |
| :-----------------------: | :-----------: | :----------: | :---------: | :-----------: | :--------------: | :-----------------: | 
| Conv1D                    | √             | √            | √           | √             | √                | √                    |
| Conv2D                    | √             | √            | √           | √             | √                | √                    |
| DepthwiseConv2D           | ×             | ×            | √           | ×             | √                | √                    |
| SeperableConv1D           | ×             | √            | √           | ×             | √                | ×                    |
| SeperableConv2D           | ×             | √            | √           | ×             | √                | ×                    |
| Conv2DTranspose           | √             | √            | √           | ×             | √                | ×                    |
| Conv3D                    | √             | √            | √           | √             | √                | √                    |
| Conv3DTranspose           | √             | √            | √           | ×             | ×                | ×                    |
| Deconvolution             | ×             | √            | √           | √             | ×                | √                    |
| Cropping1D                | √             | √            | √           | √             | ×                | ×                    |
| Cropping2D                | √             | √            | √           | √             | ×                | ×                    |
| Cropping3D                | √             | √            | √           | √             | ×                | ×                    |
| Upsampling 1D             | √             | √            | √           | ×             | √                | ×                    |
| Upsampling 2D             | √             | √            | √           | ×             | √                | ×                    |
| Upsampling 3D             | √             | √            | √           | ×             | √                | ×                    |
| ZeroPadding 1D            | ×             | √            | √           | ×             | √                | √                    |
| ZeroPadding 2D            | ×             | √            | √           | ×             | √                | √                    |
| ZeroPadding 3D            | ×             | √            | √           | ×             | √                | √                    |
| Im2Col                    | √             | ×            | ×           | ×             | ×                | ×                    |
| Spatial Pyramid Pooling   | √             | ×            | ×           | √             | ×                | ×                    |
* Upsampling in Caffe can be done by using methods shown [here](https://gist.github.com/tnarihi/54744612d35776f53278) 
 ### Pooling Layers
| Layer                     | Caffe         | Keras        | Tensorflow  | Fabrik Caffe  | Fabrik Keras    | Fabrik Tensorflow  |
| :-----------------------: | :-----------: | :----------: | :---------: | :-----------: | :--------------: | :-----------------: | 
| MaxPooling1D              | √             | √            | √           | √             | √               | ×                    |
| MaxPooling2D              | √             | √            | √           | √             | √               | √                    |
| MaxPooling3D              | √             | √            | √           | √             | √               | √                    |
| AveragePooling1D          | √             | √            | √           | ×             | √               | ×                    |
| AveragePooling2D          | √             | √            | √           | ×             | √               | ×                    |
| AveragePooling3D          | √             | √            | √           | ×             | √               | ×                    |
| GlobalMaxPooling1D        | ×             | √            | √           | ×             | ×               | ×                    |
| GlobalAveragePooling1D    | ×             | √            | √           | ×             | ×               | ×                    |
| GlobalMaxPooling2D        | ×             | √            | √           | ×             | ×               | ×                    |
| GlobalAveragePooling2D    | ×             | √            | √           | ×             | ×               | ×                    |
| GlobalMaxPooling3D        | ×             | √            | √           | ×             | ×               | ×                    |
| GlobalAveragePooling3D    | ×             | √            | √           | ×             | ×               | ×                    |
| Stochastic Pooling        | √             | ×            | ×           | √             | ×               | ×                    |
 
 ### Data Layers
| Layer                     | Caffe         | Keras        | Tensorflow  | Fabrik Caffe  | Fabrik Keras    | Fabrik Tensorflow  |
| :-----------------------: | :-----------: | :----------: | :---------: | :-----------: | :--------------: | :-----------------: | 
| Image Data                | √             | √            | √           | √             | ×               | ×                    |
| Data                      | √             | √            | √           | √             | ×               | ×                    |
| HDF5 Data                 | √             | √            | √           | √             | ×               | ×                    |
| HDF5 Output Data          | √             | √            | √           | √             | ×               | ×                    |
| Input                     | √             | √            | √           | √             | √               | √                    |
| Window Data               | √             | ×            | ×           | √             | ×               | ×                    |
| Memory Data               | √             | √            | √           | √             | ×               | ×                    |
| Dummy Data                | √             | ×            | ×           | √             | ×               | ×                    |
| Python                    | √             | √            | √           | √             | ×               | ×                    |
 
### Locally-connected Layers
| Layer                     | Caffe         | Keras        | Tensorflow  | Fabrik Caffe  | Fabrik Keras    | Fabrik Tensorflow  |
| :-----------------------: | :-----------: | :----------: | :---------: | :-----------: | :--------------: | :-----------------: | 
| LocallyConnected1D        | ×             | √            | √           | ×             | √                | √×                  |
| LocallyConnected2D        | ×             | √            | √           | ×             | √                | √×                  |
 ### Recurrent Layers
| Layer                     | Caffe         | Keras        | Tensorflow  | Fabrik Caffe  | Fabrik Keras    | Fabrik Tensorflow   |
| :-----------------------: | :-----------: | :----------: | :---------: | :-----------: | :--------------:| :-----------------: | 
| RNN                       | √             | √            | √           | √             | ×                | ×                   |
| SimpleRNN                 | ×             | √            | √           | ×             | √                | ×                   |
| GRU                       | ×             | √            | √           | ×             | √                | ×                   |
| LSTM                      | √             | √            | √           | √             | √                | ×                   |
| ConvLSTM2D                | ×             | √            | √           | ×             | ×                | ×                   |
| SimpleRNNCell             | ×             | √            | √           | ×             | ×                | ×                   |
| GRUCell                   | ×             | √            | √           | ×             | ×                | ×                   |
| LSTMCell                  | ×             | √            | √           | ×             | ×                | ×                   |
| CuDDNGRU                  | ×             | √            | √           | ×             | ×                | ×                   |
| CuDDNLSTM                 | ×             | √            | √           | ×             | ×                | ×                   |
| StackedRNNCell            | ×             | ×            | √           | ×             | ×                | ×                   |
 ### Embedding Layers
| Layer                     | Caffe         | Keras        | Tensorflow  | Fabrik Caffe  | Fabrik Keras    | Fabrik Tensorflow  |
| :-----------------------: | :-----------: | :----------: | :---------: | :-----------: | :--------------: | :-----------------: | 
| Embedding                 | √             | √            | √           | ×             | √                | ×                   |
 ### Merge Layers
| Layer                     | Caffe         | Keras        | Tensorflow  | Fabrik Caffe  | Fabrik Keras    | Fabrik Tensorflow  |
| :-----------------------: | :-----------: | :----------: | :---------: | :-----------: | :--------------: | :-----------------: | 
| Add                       | ×             | √            | √           | √             | √                | √                   |
| Subtract                  | ×             | √            | √           | ×             | ×                | ×                   |
| Multiply                  | ×             | √            | √           | √             | √                | √                   |
| Average                   | ×             | √            | √           | √             | √                | √                   |
| Minium                    | ×             | ×            | √           | ×             | ×                | ×                   |
| Maximum                   | ×             | √            | √           | √             | √                | ×                   |
| Concatenate               | √             | √            | √           | ×             | √                | √                   |
| Dot                       | ×             | √            | √           | √             | √                | √                   |
 ### Activations Layers
| Layer                     | Caffe         | Keras        | Tensorflow  | Fabrik Caffe  | Fabrik Keras    | Fabrik Tensorflow  |
| :-----------------------: | :-----------: | :----------: | :---------: | :-----------: | :--------------: | :-----------------: | 
| ReLu                      | √             | √            | √           | √             | √                | √                   |
| LeakyReLu                 | √             | √            | √           | ×             | √                | √                   |
| PReLU                     | √             | √            | √           | √             | √                | ×                   |
| ELU                       | √             | √            | √           | √             | √                | √                   |
| ThresholdedReLU           | √             | √            | √           | √             | √                | ×                   |
| Softmax                   | √             | √            | √           | ×             | √                | √                   |
| Argmax                    | √             | ×            | ×           | √             | ×                | ×                   |
| Sigmoid                   | √             | √            | √           | √             | √                | √                   |
| Hard Sigmoid              | √             | √            | √           | ×             | √                | ×                   |
| TanH                      | √             | √            | √           | √             | √                | ×                   |
| SELU                      | ×             | √            | √           | ×             | ×                | √                   |
| Absolute Value            | √             | ×            | ×           | √             | ×                | ×                   |
| Power                     | √             | √            | ×           | √             | ×                | ×                   |
| Exp                       | √             | √            | ×           | √             | ×                | ×                   |
| Linear                    | ×             | √            | √           | ×             | ×                | ×                   |
| Log                       | √             | √            | ×           | √             | ×                | ×                   |
| BNLL                      | √             | ×            | ×           | √             | ×                | ×                   |
| Bias                      | √             | ×            | ×           | √             | √                | √                   |
| Scale                     | √             | ×            | ×           | √             | ×                | ×                   |
 ### Utility Layers
| Layer                     | Caffe         | Keras        | Tensorflow  | Fabrik Caffe  | Fabrik Keras    | Fabrik Tensorflow  |
| :-----------------------: | :-----------: | :----------: | :---------: | :-----------: | :--------------: | :---------------: | 
| Slicing                   | √             | ×            | ×           | √             | ×                | ×                 |
| Eltwise                   | √             | ×            | ×           | √             | √                | √                 |
| Parameter                 | √             | ×            | ×           | √             | ×                | √                 |
| Reduction                 | √             | ×            | ×           | √             | ×                | ×                 |
| Silence                   | √             | ×            | ×           | √             | ×                | ×                 |
 ### Loss Layers
| Layer                     | Caffe         | Keras        | Tensorflow  | Fabrik Caffe  | Fabrik Keras    | Fabrik Tensorflow  |
| :-----------------------: | :-----------: | :----------: | :---------: | :-----------: | :--------------: | :-----------------: | 
| Multinomial Logistic Loss | √             | ×            | ×           | √             | ×                | ×                   |
| Infogain Loss             | √             | ×            | ×           | √             | ×                | ×                   |
| Softmax with Loss         | √             | ×            | √           | √             | ×                | ×                   |
| Sum-of-Squares/Euclidean  | √             | ×            | ×           | √             | ×                | ×                   |
| Hinge / Margin            | √             | √            | √           | √             | ×                | ×                   |
| Sigmoid Cross-Entropy Loss| √             | ×            | √           | √             | ×                | ×                   |
| Accuracy / Top-k layer    | √             | ×            | ×           | √             | ×                | ×                   |
| Contrastive Loss          | √             | ×            | ×           | √             | ×                | ×                   |
 ### Normalization Layers
| Layer                     | Caffe         | Keras        | Tensorflow  | Fabrik Caffe  | Fabrik Keras    | Fabrik Tensorflow  |
| :-----------------------: | :-----------: | :----------: | :---------: | :-----------: | :--------------: | :-----------------: | 
| BatchNormalization        | √             | √            | √           | √             | √                | √                   |
| MVN                       | √             | ×            | ×           | √             | ×                | ×                   |
 ### Noise Layers
| Layer                     | Caffe         | Keras        | Tensorflow  | Fabrik Caffe  | Fabrik Keras    | Fabrik Tensorflow  |
| :-----------------------: | :-----------: | :----------: | :---------: | :-----------: | :--------------: | :-----------------: | 
| GaussianNoise             | ×             | √            | √           | ×             | √                | ×                   |
| GaussianDropout           | √             | √            | √           | ×             | √                | ×                   |
| AlphaDropout              | √             | √            | √           | ×             | √                | √                   |
 ### Layer Wrappers
| Layer                     | Caffe         | Keras        | Tensorflow  | Fabrik Caffe  | Fabrik Keras    | Fabrik Tensorflow  |
| :-----------------------: | :-----------: | :----------: | :---------: | :-----------: | :--------------: | :-----------------: | 
| TimeDistributed           | ×             | √            | √           | ×             | √                | ×                   |
| Bidirectional             | ×             | √            | √           | ×             |  √               | ×                   |
 ### Custom Layers
| Layer         | Caffe        | Keras       | Tensorflow  | Fabrik Caffe  | Fabrik Keras    | Fabrik Tensorflow  |
| :-----------: | :----------: | :---------: | :---------: | :-----------: | :--------------: | :-----------------: | 
|               | √            | √           | Use Keras API for custom layers | NA           | NA               | NA|
| LRN           | √            | √           | √           | √            | √                | √                   |

 ## Additional Notes:
* Keras does not support the LRN layer used in Alexnet & many other models. To use the LRN layer refer to [here](https://github.com/Cloud-CV/Fabrik/blob/master/tutorials/keras_custom_layer_usage.md.)
* Documentation for writing your own Keras layers is found [here](https://keras.io/layers/writing-your-own-keras-layers/) 
 ## Documentation for Caffe, Keras, and Tensorflow layers
* Documentation for all Keras Layers is found [here](https://keras.io/layers/about-keras-layers/)
* Documentation for all Caffe Layers is found [here](http://caffe.berkeleyvision.org/tutorial/layers.html)
* Documentation for all Tensorflow Layers is found [here](https://www.tensorflow.org/api_docs/python/tf/layers)

#### * This documentation is subject to change as more layers are supported.
