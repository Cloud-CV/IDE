# Layers in Tensorflow 

Following table lists classes available in TensorFlow:



| Class Name      | **Description** |
|-----------------| ----------------|
| AveragePooling1D | Average Pooling layer for 1D inputs |
| AveragePooling2D | Average pooling layer for 2D inputs (e.g. images) |
| AveragePooling3D | Average pooling layer for 3D inputs (e.g. volumes) |
| BatchNormalization | Accelerates deep network training by normalizing layer inputs |
| Conv1D | 1D convolution layer (e.g. temporal convolution). This layer creates a convolution kernel that is convolved (actually cross-correlated) with the layer input to produce a tensor of outputs.|
| Conv2D | 2D convolution layer (e.g. spatial convolution over images). This layer creates a convolution kernel that is convolved (actually cross-correlated) with the layer input to produce a tensor of outputs. | 
| Conv2DTranspose | Transposed 2D convolution layer (sometimes called 2D Deconvolution). The need for transposed convolutions generally arises from the desire to use a transformation going in the opposite direction of a normal convolution, i.e., from something that has the shape of the output of some convolution to something that has the shape of its input while maintaining a connectivity pattern that is compatible with said convolution. |
| Conv3D | 3D convolution layer (e.g. spatial convolution over volumes). This layer creates a convolution kernel that is convolved (actually cross-correlated) with the layer input to produce a tensor of outputs. |
| Dense | Densely-connected layer class. |
| Dropout | Applies Dropout to the input. |
| Flatten | Flattens an input tensor while preserving the batch axis (axis 0).|
| Layer | Base layer class. **It is considered legacy.**|
| MaxPooling1D | Max Pooling layer for 1D inputs. |
| MaxPooling2D | Max pooling layer for 2D inputs (e.g. images).|
| MaxPooling3D | Max pooling layer for 3D inputs (e.g. volumes).|
| SeparableConv1D | Depthwise separable 1D convolution. This layer performs a depthwise convolution that acts separately on channels, followed by a pointwise convolution that mixes channels.|
| SeparableConv2D | Depthwise separable 2D convolution. This layer performs a depthwise convolution that acts separately on channels, followed by a pointwise convolution that mixes channels.|

