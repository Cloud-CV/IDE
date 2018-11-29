# Adding New Layers

- For setup instructions, check [README](https://github.com/Cloud-CV/Fabrik/blob/master/README.md).
- Add your new layer(s) to the [data.js](https://github.com/Cloud-CV/Fabrik/blob/master/ide/static/js/data.js) file.


## Basics for adding a new layer

- Open the ```data.js``` file in any text other.

<img src="https://raw.githubusercontent.com/Cloud-CV/Fabrik/master/tutorials/layercategory.png" />

- You should see the line ```/* ********** Data Layers ********** */```, it is the category of the layer. There are many categories in the file as mentioned below:
    - Data Layers
    - Vision Layers
    - Recurrent Layers
    - Activation/Neuron Layers
    - Normalization Layers
    - Noise Layers
    - Common Layers
    - Loss Layers
    - Utility Layers
    - Python Layers
- You should add the new layer below the category it belongs to.
- Moving to the next line in the image, we create a new json element (layer). The line ```// Only Caffe``` tells that this layer is only for Caffe and not for Keras.
- Add the suitable comment for the new layer or leave it if there is no such need.


## Detailed overview of a layer

<img src="https://raw.githubusercontent.com/Cloud-CV/Fabrik/master/tutorials/layerdetails.png" />

- Here is a whole layer shown named ```ReLU```. It is a ```Activation/Neuron Layer```, that's why it is kept below the line ```/* ********** Activation/Neron Layers ********** */```.
- Then add the suitable comment for you layer or leave it empty if it is not for any specific framework.
- Keywords' explanation:
    - name: Name of the layer.
    - color: Color of the layer to be shown in frontend.
    - endpoint: Endpoints of the layer.
        - src: Source endpoint of the layer.
        - trg: Target endpoint of the layer.
    - params: Parameters for the layer.
        - inplace: Checkbox input for the layer.
        - negative_slope: Numerical input for the layer.
        - caffe: Availibility of Caffe (Checkbox input).
    - props: It defines the properties of the layer.
    - learn: This declares if the layer can be used for learning.
- We can define different parameters for a layer and it is not limited to ```inplace``` & ```negative_slope```.


## Making the layer visible in Fabrik

- Open [pane.js](https://github.com/Cloud-CV/Fabrik/blob/master/ide/static/js/pane.js) in a text editor, and you should see something like this.

<img src="https://raw.githubusercontent.com/Cloud-CV/Fabrik/master/tutorials/layerpanel.png" />

- Now, add a new line for the layer you just added in ```data.js``` in the section of Activation/Neuron Layer, because this layer belongs to this category.
- ```<PaneElement handleClick={this.props.handleClick} id="your_layer_id">your_layer_name</PaneElement>``` this line will make your layer visible in Fabrik.
- Open [filterbar.js](https://github.com/Cloud-CV/Fabrik/blob/master/ide/static/js/filterbar.js) in a text editor, add ```"your_layer_id"``` to 1(or more) of 3 framework filter array ```var KerasLayers = [...]```, ```var TensorFlowLayers = [...]``` or ```var CaffeLayers = [...]```. This should be like this ```var KerasLayers = ["RNN_Button", "GRU_Button", "your_layer_id"]```. This arrays are placed inside ```changeEvent() {}``` function.


## Adding layer handling to the backend

### 1. Caffe

#### Import

- Open [import_prototxt.py](https://github.com/Cloud-CV/Fabrik/blob/master/caffe_app/views/import_prototxt.py) file in a text editor.

    <img src="https://raw.githubusercontent.com/Cloud-CV/Fabrik/master/tutorials/layerImportPrototxt1.png" />

- Add a function for the new layer below the category of this layer.
- Load the parameters, do the calculations for your layer in pyhton and return the value of ```params``` (parameters).
- Move down in the file.

    <img src="https://raw.githubusercontent.com/Cloud-CV/Fabrik/master/tutorials/layerImportPrototxt2.png" />

- Add your defined layer in the ```layer_dict``` array, as shown above.

#### Export

- Now, open [jsonToPrototxt.py](https://github.com/Cloud-CV/Fabrik/blob/master/ide/utils/jsonToPrototxt.py) in a text editor.

    <img src="https://raw.githubusercontent.com/Cloud-CV/Fabrik/master/tutorials/layerJSONtoPrototxt1.png" />

- Add an export function for training and testing of the new layer.
- There you need to load parameters, then train & test values and at last return the trained and tested data.
- Move down in this file as well.

    <img src="https://raw.githubusercontent.com/Cloud-CV/Fabrik/master/tutorials/layerJSONtoPrototxt2.png" />

- Add the export function in the ```layer_map``` array.


### Keras

#### Importing a layer

- Open [keras_app/views/layers_import.py](https://github.com/Cloud-CV/Fabrik/blob/master/keras_app/views/layers_import.py).
    - Add a function to process the new layer.
        - The function should take a Keras `Layer` object, called `layer`.
        - You can get the layer parameters from `layer`, and create a json layer with the function  [jsonLayer](https://github.com/Cloud-CV/Fabrik/blob/master/keras_app/views/layers_import.py#L529). 
        - Return the json layer you just created.

        ```
        def Activation(layer):
            activationMap = {
                'softmax': 'Softmax',
                'relu': 'ReLU',
                'tanh': 'TanH',
                'sigmoid': 'Sigmoid',
                'selu': 'SELU',
                'softplus': 'Softplus',
                'softsign': 'Softsign',
                'hard_sigmoid': 'HardSigmoid',
                'linear': 'Linear'
            }
            if (layer.__class__.__name__ == 'Activation'):
                return jsonLayer(activationMap[layer.activation.func_name], {}, layer)
            else:
                tempLayer = {}
                tempLayer['inbound_nodes'] = [
                    [[layer.name + layer.__class__.__name__]]]
            return jsonLayer(activationMap[layer.activation.func_name], {}, tempLayer)
        ```

- Open [keras_app/views/import_json.py](https://github.com/Cloud-CV/Fabrik/blob/master/keras_app/views/import_json.py) in a text editor.
    - From `layers_import`, import the function you just defined.

        ```
        from layers_import import Activation
        ```

    - Map your layer name to your function in [`layer_map`](https://github.com/Cloud-CV/Fabrik/blob/master/keras_app/views/import_json.py#L53).

        ```diff
            layer_map = {
                'InputLayer': Input,
                'Dense': Dense,
                ...
        +       'relu': Activation
            }
        ```

#### Exporting a layer

- Open [keras_app/views/layers_export.py](https://github.com/Cloud-CV/Fabrik/blob/master/keras_app/views/layers_export.py).
    - Add a function to process your layer that takes in the following arguments:
        - `layer` : a json string, same as the one created in layers_import.py
        - `layer_in`: a Tensor, output of the previous layer
        - `layerId`: a string id of layer
        - `tensor`: a bool
    - Inside the function, get layer parameters from `layer` and build a Keras layer using them.
    - If `tensor==True`, call the Keras layer on `layer` and return the output tensor. Else, return the `layer` itself.

    ```
    def activation(layer, layer_in, layerId, tensor=True):
        out = {}
        if (layer['info']['type'] == 'ReLU'):
            if ('negative_slope' in layer['params'] and layer['params']['negative_slope'] != 0):
                out[layerId] = LeakyReLU(alpha=layer['params']['negative_slope'])
            else:
                out[layerId] = Activation('relu')
        elif (layer['info']['type'] == 'PReLU'):
            out[layerId] = PReLU()
        elif (layer['info']['type'] == 'ELU'):
            out[layerId] = ELU(alpha=layer['params']['alpha'])
        elif (layer['info']['type'] == 'ThresholdedReLU'):
            out[layerId] = ThresholdedReLU(theta=layer['params']['theta'])
        elif (layer['info']['type'] == 'Sigmoid'):
            out[layerId] = Activation('sigmoid')
        elif (layer['info']['type'] == 'TanH'):
            out[layerId] = Activation('tanh')
        elif (layer['info']['type'] == 'Softmax'):
            out[layerId] = Activation('softmax')
        elif (layer['info']['type'] == 'SELU'):
            out[layerId] = Activation('selu')
        elif (layer['info']['type'] == 'Softplus'):
            out[layerId] = Activation('softplus')
        elif (layer['info']['type'] == 'Softsign'):
            out[layerId] = Activation('softsign')
        elif (layer['info']['type'] == 'HardSigmoid'):
            out[layerId] = Activation('hard_sigmoid')
        elif (layer['info']['type'] == 'Linear'):
            out[layerId] = Activation('linear')
        if tensor:
            out[layerId] = out[layerId](*layer_in)
    return out
    ```

- Open [ide/tasks.py](https://github.com/Cloud-CV/Fabrik/blob/master/ide/tasks.py).
    - From `layers_export`, import the function you just defined.

    ```
    from keras_app.views.layers_export import activation
    ```

    - Add your function to [`layer_map`](https://github.com/Cloud-CV/Fabrik/blob/master/ide/tasks.py#L65).

    ```diff
        layer_map = {
            'InputLayer': Input,
            'Dense': Dense,
                ...
    +       'ReLU': activation
        }
    ```

- Open [keras_app/views/export_json.py](https://github.com/Cloud-CV/Fabrik/blob/master/keras_app/views/export_json.py)
    - From `layers_export`, import the function you just defined.

    ```
    from layers_export import activation
    ```

    - Add your function to [`layer_map`](https://github.com/Cloud-CV/Fabrik/blob/master/keras_app/views/export_json.py#L36).

    ```diff
        layer_map = {
            'InputLayer': Input,
            'Dense': Dense,
                ...
    +       'ReLU': activation
        }
    ```

    Fabrik exports models using the celery task [export_keras_json](https://github.com/Cloud-CV/Fabrik/blob/master/ide/tasks.py#L60), but we keep `export_json.py` updated to test the export code.


### Tensorflow

#### Importing a layer

- Open [import_graphdef.py](https://github.com/Cloud-CV/Fabrik/blob/master/tensorflow_app/views/import_graphdef.py)
    - Add your layer to one of these four dictionaries that map Tensorflow ops to Caffe layers:
        - `op_layer_map` : if the op has can be mapped to a Caffe layer directly
        - `activation_map` : if the op is a simple activation
        - `name_map` : if the op type can only be inferred from the name of the op
        - `initializer_map` : if the op is an initializer

        ```diff
        activation_map = {'Sigmoid': 'Sigmoid', 'Softplus': 'Softplus', 'Softsign': 'Softsign',
                          ...
        +                 'Relu': 'ReLU'}
        ```

    - Inside the loop [`for node in graph.get_operations()`](https://github.com/Cloud-CV/Fabrik/blob/master/tensorflow_app/views/import_graphdef.py#L169), write code to get any layer parameters needed and build the layer.

#### Exporting a layer

Fabrik exports Tensorflow models using Keras. [See the guide above](#exporting-a-layer) for exporting a layer for Keras.


## Testing and pushing the new layer.

- Run the fabrik application on you local machine by following the instructions in [README](https://github.com/Cloud-CV/Fabrik/blob/master/README.md) file.

<img src="https://raw.githubusercontent.com/Cloud-CV/Fabrik/master/tutorials/layertesting.png" />

- Check the new layer inside the category you added it. See if all the parameters are properly displayed and usable as you wanted.
- If everything is working fine commit your changes and push it to your fork then make a Pull Request.
- Congratulations! Happy contributing :-)
