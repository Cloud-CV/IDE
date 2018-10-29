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



## Additional Notes:
* Keras doesn't support the LRN layer used in Alexnet & many other models out of the box. To use the LRN layer refer to here: https://github.com/Cloud-CV/Fabrik/blob/master/tutorials/keras_custom_layer_usage.md.
