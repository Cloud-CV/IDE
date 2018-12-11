# Layers in Keras

Keras is second popular deep learning framework and layers is key component of the framework.

### Core Layers
| Class Name      | **Description** |
|-----------------| ----------------|
| Dense      | Regular densely-connected Neural Network layer|
| Activation | Applies an activation function to an output. |
| Dropout    | Applies Dropout to the input.|
| Flatten  | Flattens the input. Does not affect the batch size.|
| Input | Used to instantiate a Keras tensor. |
| Reshape | Reshapes an output to a certain shape.|
| Permute | Permutes the dimensions of the input according to a given pattern.|
| RepeatVector | Repeats the input n times.|
| Lambda | Wraps arbitrary expression as a Layer object. |
| ActivityRegularization | Layer that applies an update to the cost function based input activity.|
| Masking | Masks a sequence by using a mask value to skip timesteps.|
| SpatialDropout1D | Spatial 1D version of Dropout. |
| SpatialDropout2D | Spatial 2D version of Dropout. |
| SpatialDropout3D | Spatial 3D version of Dropout. |



### Convolutional Layers
| Class Name      | **Description** |
|-----------------| ----------------|
| Conv1D      | Spatial 3D version of Dropout.|
| Conv2D | 2D convolution layer (e.g. spatial convolution over images).|
| SeparableConv1D | Depthwise separable 1D convolution. |
| SeparableConv2D | Depthwise separable 2D convolution. |
| DepthwiseConv2D | Depthwise separable 2D convolution |
| Conv2DTranspose | Transposed convolution layer (sometimes called Deconvolution). |
| Conv3D | 3D convolution layer (e.g. spatial convolution over volumes). |
| Conv3DTranspose | Transposed convolution layer (sometimes called Deconvolution). |
| Cropping1D | Cropping layer for 1D input (e.g. temporal sequence). | 
| Cropping2D | Cropping layer for 2D input (e.g. picture). |
| Cropping3D | Cropping layer for 3D data (e.g. spatial or spatio-temporal). |
| UpSampling1D | Upsampling layer for 1D inputs.|
| UpSampling2D | Upsampling layer for 2D inputs. |
| UpSampling3D | Upsampling layer for 3D inputs. |
| ZeroPadding1D | Zero-padding layer for 1D input (e.g. temporal sequence). | 
| ZeroPadding2D | Zero-padding layer for 2D input (e.g. picture).|
| ZeroPadding3D | Zero-padding layer for 3D data (spatial or spatio-temporal). |



### Pooling Layers
| Class Name      | **Description** |
|-----------------| ----------------|
| MaxPooling1D      | Max pooling operation for temporal data.|
| MaxPooling2D | Max pooling operation for spatial data.
| MaxPooling3D | Max pooling operation for 3D data (spatial or spatio-temporal).|
| AveragePooling1D | Average pooling for temporal data. |
| AveragePooling2D | Average pooling operation for spatial data.|
| AveragePooling3D | Average pooling operation for 3D data (spatial or spatio-temporal).|
| GlobalMaxPooling1D | Global max pooling operation for temporal data. |
| GlobalAveragePooling1D | Global average pooling operation for temporal data. |
| GlobalMaxPooling2D | Global max pooling operation for spatial data.|
| GlobalAveragePooling2D | Global average pooling operation for spatial data. |
| GlobalMaxPooling3D | Global Max pooling operation for 3D data. |
| GlobalAveragePooling3D | Global Average pooling operation for 3D data.|


### Locally Connected Layers
| Class Name      | **Description** |
|-----------------| ----------------|
| LocallyConnected1D      | Locally-connected layer for 1D inputs.|
| LocallyConnected2D | Locally-connected layer for 2D inputs. |
|RNN | Base class for recurrent layers.|
| SimpleRNN | Fully-connected RNN where the output is to be fed back to input.|
| GRU | Gated Recurrent Unit |
| LSTM | Long Short-Term Memory layer |
| ConvLSTM2D | Convolutional LSTM. It is similar to an LSTM layer, but the input transformations and recurrent transformations are both convolutional.|
| SimpleRNNCell | Cell class for SimpleRNN. |
| GRUCell | Cell class for the GRU layer. |
| LSTMCell | Cell class for the LSTM layer. |
| CuDNNGRU | Fast GRU implementation backed by CuDNN. -It can only be run on GPU, with the TensorFlow backend.- |
| CuDNNLSTM | Fast LSTM implementation with CuDNN. |


### Embedding Layers
| Class Name      | **Description** |
|-----------------| ----------------|
| Embedding | Turns positive integers (indexes) into dense vectors of fixed size.|


### Merge Layers
 |Class Name      | **Description** |
|-----------------| ----------------|
| Add | Layer that adds a list of inputs.|
| Subtract | Layer that subtracts two inputs. |
| Multiply | Layer that multiplies (element-wise) a list of inputs. |
| Average | Layer that averages a list of inputs. |
| Maximum | Layer that computes the maximum (element-wise) a list of inputs. |
| Concatenate | Layer that concatenates a list of inputs. |
| Dot | Layer that computes a dot product between samples in two tensors.|


### Advanced Activations Layers
 |Class Name      | **Description** |
|-----------------| ----------------|
| LeakyRELU | Leaky version of a Rectified Linear Unit. |
| PReLU | Parametric Rectified Linear Unit. |
| ELU | Exponential Linear Unit. |
| ThresholdedReLU | Thresholded Rectified Linear Unit.|
| Softmax | Softmax activation function. |
| ReLU | Rectified Linear Unit activation function. |


### Normalization Layers
 |Class Name      | **Description** |
|-----------------| ----------------|
| BatchNormalization | Batch normalization layer |


### Noise Layers
 |Class Name      | **Description** |
|-----------------| ----------------|
| GaussianNoise | Apply additive zero-centered Gaussian noise.|
| GaussianDropout | Apply multiplicative 1-centered Gaussian noise. |
| AlphaDropout | Applies Alpha Dropout to the input.|


### Layer Wrappers
 |Class Name      | **Description** |
|-----------------| ----------------|
| TimeDistributed | This wrapper applies a layer to every temporal slice of an input.|
| Bidirectional | Bidirectional wrapper for RNNs.|
 
