# Automated Neural Network Drawing Procedure

### Step 1:
Add the ability to export the internal json of the model to the end of the following javascript files: import_prototxt.py, import_json.py, and import_graphdef.py

Use the following code:
```sh
with open('data.json', 'w') as outfile:
    json.dump({'result': 'success', 'net': net, 'net_name': model.name}, outfile)
```

### Step 2:
Create code to find the dimension of the model using the json produced in step 1.
See this link for a working example: [Github](https://github.com/rpalakkal/neuralustrate/blob/master/getDimensions.py) 

### Step 3:
Modify [NN-SVG](https://github.com/zfrenchee/NN-SVG) to generate a network diagram by passing through the network dimensions rather than by manually inputting the dimensions. Look at the [FCNN-SVG](https://github.com/zfrenchee/NN-SVG/blob/master/index.html) and the redraw function. Change the architecture variable to be equal to the output of the code from step 2(right now it is equal to the user input).

### Step 4:
Add a button on the Fabrik page to run the code from Step 2 and open the modified NN-SVG in a new tab.

Right now we are exporting the internal json representation of the model(for testing purposes), but this means people who don't have Fabrik installed on local machine can run it. Ideally, make the getDimensions code into a function, and it should be able to be called from the import function so that there is no need to create a json model file.

