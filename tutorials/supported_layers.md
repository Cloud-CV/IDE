# Layers supported in Caffe, Keras, and Tensorflow
Below is are table showing which layers are supported by Caffe, Keras, and Tensorflow:

#### Core Layers
| Layer                     | Caffe         | Keras        | Tensorflow    |
| :-----------------------: | :-----------: | :----------: | :-----------: |
| Activation                | √             | √            | √             |
| ActivityRegularization    | √             | √            | √             |
| Dense                     | √             | √            | √             |
| Dropout                   | √             | √            | √             |
| Flatten                   | √             | √            | √             |
| Lambda                    | √             | √            | √             |
| Masking                   | √             | √            | √             |
| Permute                   | √             | √            | √             |
| Repeat Vector             | √             | √            | √             |
| Reshape                   | √             | √            | √             |
| Spatial Dropout 1D        | √             | √            | √             |
| Spatial Dropout 2D        | √             | √            | √             |
| Spatial Dropout 3D        | √             | √            | √             |

#### Convolutional Layers
| Layer                     | Caffe         | Keras        | Tensorflow    |
| :-----------------------: | :-----------: | :----------: | :-----------: |
| Conv1D                    | √             | √            | √             |
| Conv2D                    | √             | √            | √             |
| SeperableConv1D           | √             | √            | √             |
| SeperableConv2D           | √             | √            | √             |
| Conv2DTranspose           | √             | √            | √             |
| Conv3D                    | √             | √            | √             |
| Conv3DTranspose           | √             | √            | √             |
| Cropping1D                | √             | √            | √             |
| Cropping2D                | √             | √            | √             |
| Cropping3D                | √             | √            | √             |
| Upsampling 1D             | √             | √            | √             |
| Upsampling 2D             | √             | √            | √             |
| Upsampling 3D             | √             | √            | √             |
| ZeroPadding 1D            | √             | √            | √             |
| ZeroPadding 2D            | √             | √            | √             |
| ZeroPadding 3D            | √             | √            | √             |

#### Pooling Layers
| Layer                     | Caffe         | Keras        | Tensorflow    |
| :-----------------------: | :-----------: | :----------: | :-----------: |
| MaxPooling1D              | √             | √            | √             |
| MaxPooling2D              | √             | √            | √             |
| MaxPooling3D              | √             | √            | √             |
| AveragePooling1D          | √             | √            | √             |
| AveragePooling2D          | √             | √            | √             |
| AveragePooling3D          | √             | √            | √             |
| GlobalMaxPooling1D        | √             | √            | √             |
| GlobalAveragePooling1D    | √             | √            | √             |
| GlobalMaxPooling2D        | √             | √            | √             |
| GlobalAveragePooling2D    | √             | √            | √             |
| GlobalMaxPooling3D        | √             | √            | √             |
| GlobalAveragePooling3D    | √             | √            | √             |

#### Locally-connected Layers
| Layer                     | Caffe         | Keras        | Tensorflow    |
| :-----------------------: | :-----------: | :----------: | :-----------: |
| LocallyConnected1D        | √             | √            | √             |
| LocallyConnected2D        | √             | √            | √             |

#### Recurrent Layers
| Layer                     | Caffe         | Keras        | Tensorflow    |
| :-----------------------: | :-----------: | :----------: | :-----------: |
| RNN                       | √             | √            | √             |
| SimpleRNN                 | √             | √            | √             |
| GRU                       | √             | √            | √             |
| LSTM                      | √             | √            | √             |
| ConvLSTM2D                | √             | √            | √             |
| SimpleRNNCell             | √             | √            | √             |
| GRUCell                   | √             | √            | √             |
| LSTMCell                  | √             | √            | √             |
| CuDDNGRU                  | √             | √            | √             |
| CuDDNLSTM                 | √             | √            | √             |

#### Embedding Layers
| Layer                     | Caffe         | Keras        | Tensorflow    |
| :-----------------------: | :-----------: | :----------: | :-----------: |
| Embedding                 | √             | √            | √             |

#### Merge Layers
| Layer                     | Caffe         | Keras        | Tensorflow    |
| :-----------------------: | :-----------: | :----------: | :-----------: |
| Add              | √             | √            | √             |
| Subtract              | √             | √            | √             |
| Multiply              | √             | √            | √             |
| Average          | √             | √            | √             |
| Maximum          | √             | √            | √             |
| Concatenate          | √             | √            | √             |
| Dot        | √             | √            | √             |

#### Advanced Activations Layers
| Layer                     | Caffe         | Keras        | Tensorflow    |
| :-----------------------: | :-----------: | :----------: | :-----------: |
| ReLu                      | √             | √            | √             |
| LeakyReLu                 | √             | √            | √             |
| PReLU                     | √             | √            | √             |
| ELU                       | √             | √            | √             |
| ThresholdedReLU           | √             | √            | √             |
| Softmax                   | √             | √            | √             |

#### Normalization Layers
| Layer                     | Caffe         | Keras        | Tensorflow    |
| :-----------------------: | :-----------: | :----------: | :-----------: |
| BatchNormalization        | √             | √            | √             |

#### Noise Layers
| Layer                     | Caffe         | Keras        | Tensorflow    |
| :-----------------------: | :-----------: | :----------: | :-----------: |
| GaussianNoise             | √             | √            | √             |
| GaussianDropout           | √             | √            | √             |
| AlphaDropout              | √             | √            | √             |

#### Layer Wrappers
| Layer                     | Caffe         | Keras        | Tensorflow    |
| :-----------------------: | :-----------: | :----------: | :-----------: |
| TimeDistributed           | √             | √            | √             |
| Bidirectional             | √             | √            | √             |


## Additional Notes:
* Keras doesn't support the LRN layer used in Alexnet & many other models out of the box. To use the LRN layer refer to here: https://github.com/Cloud-CV/Fabrik/blob/master/tutorials/keras_custom_layer_usage.md.
* Documentation for writing your own Keras layers is found here: https://keras.io/layers/writing-your-own-keras-layers/ 
