# Layers in Caffe

Keras is third-most popular deep learning framework and layers is key component of the framework.

### Data Layers
| Class Name      | **Description** |
|-----------------| ----------------|
| ImageData | Read raw images |
| Database | Read data from LEVELDB or LMDB. |
| HDF5 Input | Read HDF5 data, allows data of arbitrary dimensions.|
| HDF5 Output | Write data as HDF5.|
| Input | Typically used for networks that are being deployed.|
| Window Data | Read window data file| 
| Memory Data | Read data directly from memory|
| Dummy Data| For static data and debugging|

### Vision Layers
| Class Name      | **Description** |
|-----------------| ----------------|
| Convolution Layer | Convolves the input image with a set of learnable filters, each producing one feature map in the output image. |
| Pooling Layer | Max, average, or stochastic pooling. |
| Spatial Pyramid Pooling (SPP) | |
| Crop | Perform cropping transformation| 
| Deconvolution Layer | Transposed convolution.| 

### Recurrent Layer
| Class Name      | **Description** |
|-----------------| ----------------|
| Recurrent ||
| RNN ||
| LSTM ||

### Common Layer
| Class Name      | **Description** |
|-----------------| ----------------|
| Inner Product | Fully connected layer |
| Dropout | |
| Embed | For learning embeddings of one-hot encoded vector (takes index as input). |

### Normalization Layer
| Class Name      | **Description** |
|-----------------| ----------------|
| Local Response Normalization (LRN) | Performs a kind of “lateral inhibition” by normalizing over local input regions.|
| Mean Variance Normalization (MVN) | Performs contrast normalization / instance normalization. |
| Batch Normalization | Performs normalization over mini-batches. |

### Activation / Neuron Layer
| Class Name      | **Description** |
|-----------------| ----------------|
| ReLU | ReLU and Leaky-ReLU rectification. |
| PReLU | Parametric ReLU |
| ELU | Exponential Linear Rectification. |
| Sigmoid | |
| TanH | |
| Absolute Value ||
| Power | f(x) = (shift + scale * x) ^ power.|
| Exp | f(x) = base ^ (shift + scale * x). |
| Log | f(x) = log(x). |
| BNLL | f(x) = log(1 + exp(x)).|
| Threshold | Performs step function at user defined threshold.|
| Bias | Adds a bias to a blob that can either be learned or fixed.|
| Scale | Scales a blob by an amount that can either be learned or fixed.|

### Utility Layers
| Class Name      | **Description** |
|-----------------| ----------------|
| Flatten | |
| Reshape ||
| Batch Reindex ||
| Split ||
| Concat ||
| Slicing ||
| Eltwise ||
| Filter / Mask ||
| Parameter ||
| Reduction ||
| Silence ||
| ArgMax ||
| Softmax ||

### Loss Layers
| Class Name      | **Description** |
|-----------------| ----------------|
| Multinomial Logistic Loss | |
| Infogain Loss | A generalization of MultinomialLogisticLossLayer. |
| Softmax with Loss | Computes the multinomial logistic loss of the softmax of its inputs. It’s conceptually identical to a softmax layer followed by a multinomial logistic loss layer, but provides a more numerically stable gradient.|
| Sum-of-Squares / Euclidean | Computes the sum of squares of differences of its two inputs |
| Hinge / Margin | The hinge loss layer computes a one-vs-all hinge (L1) or squared hinge loss (L2). |
| Sigmoid Cross-Entropy Loss | Computes the cross-entropy (logistic) loss, often used for predicting targets interpreted as probabilities.|
| Accuracy / Top-k layer | Scores the output as an accuracy with respect to target – it is not actually a loss and has no backward step.|
| Contrastive Loss | |
