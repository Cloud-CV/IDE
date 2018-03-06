# PyTorch Documentation for import and export functionality

## Models built in PyTorch can currently be saved in two formats:
- '.pt' format (torch.save)
- '.proto' format (torch.onnx.export)

## Building the graph:
The operations of a graph can be extracted by back-tracing from the output variable's gradient function, all the way back to the input. Since there is no official graph tracing release yet, a rudimentary functionality is introduced in pytorch_app/views/**layers_import.py**, based on [this](https://github.com/szagoruyko/pytorchviz).

## Issues:

- A dummy input variable is required to perform graph tracing (Even for onnx format). This means that we need to know the size of the input. Problem is, torch's sequential models don't contain the input layer's shape.

- Torch's '.pt' format saves the model along with the actual variables. This can lead to severe memory overhead. The resnet-101 model's pt file is 170MB for instance.  

- For importing in other frameworks, we first open the file, and then send it to the import functionality in the backend via Django. Problem is, '.pt' files are in a binary format, there is no human readable format availabe. Python's default 'open' function doesn't work, even in 'rb' mode for torch '.pt' files. We would need to use torch.load or pickle to open it.

## Required Solutions:

- Get the input shape from the user (Easier, Also necessary for onnx support) (or) Calculate a dummy shape by backtracing from the last layer (Assuming we can access the other shapes without graph tracing)

- A method to transfer PyTorch binaries from backend to frontend (Can first try without using torch.load or torch.save)

- A more efficient method to save PyTorch models. Ideally, human readable (json-like) and less size hungry (without the actual variables) 
