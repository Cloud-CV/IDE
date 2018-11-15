## Using an Exported Tensorflow Model

In order to export a Tensorflow model from Fabrik:

1. First, select the 2nd button from the left in the Actions section of the sidebar.
<img src="https://raw.githubusercontent.com/Cloud-CV/Fabrik/master/tutorials/exportbutton.png">

2. A drop-down list should appear. Select Tensorflow.
    * This should download a ```.meta``` file to your computer.  
    <img src="https://raw.githubusercontent.com/Cloud-CV/Fabrik/master/tutorials/export_tensorflow.png">

3. Rename the file to ```model.meta```.

4. Load the model from ```model.meta``` using the following code:

    ```
    import tensorflow as tf
    from google.protobuf import text_format

    # read the graphdef from the model file
    with open('model.meta', 'r') as model_file:
	    model_protobuf = text_format(model_file.read(),
		    	                 tf.MetaGraphDef())
    
    # import the graphdef into the default graph
    tf.train.import_meta_graph(model_protobuf)
    ```

### Code template

[The code template](../example/tensorflow/code_template/tensorflow_sample.py) loads the model from a ```.meta``` file into the default graph. Additional operations like layers and optimizers can be then built onto the graph as required.

To run the code, run:

```
python tensorflow_sample.py model.meta
```

Replace ```model.meta``` with the model file that you want to use.
