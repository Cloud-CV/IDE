<h1>Adding New Model - Caffe</h1>

1. For Setup instructions, look at the [README](https://github.com/Cloud-CV/Fabrik/blob/master/README.md).
2. Open the <b>[Example folder](https://github.com/Cloud-CV/Fabrik/tree/master/example)</b>.
3. Open the <b>[Caffe folder](https://github.com/Cloud-CV/Fabrik/tree/master/example/caffe)</b> and add the new model in the form of .prototxt file.
4. Then add your entry to the front-end by refering to the following example <b>[topbar.js](https://github.com/Cloud-CV/Fabrik/blob/master/ide/static/js/topBar.js)</b>, in this the ```id``` should be the name of your prototxt without the extension.
```
<li><ModelElement importNet={this.props.importNet} framework="caffe" id="sample">Sample</ModelElement></li>

```
5. After making these changes, test if loading the model and exporting it to both or at least one framework is working fine and document it accordingly in your pull request.
6. Create a pull request for the same and get reviewed by the mentors.
Cheers!

<h1>Adding New Model - Keras </h1>

1. For Setup instructions, look at the [README](https://github.com/Cloud-CV/Fabrik/blob/master/README.md).
2. Open the <b>[Example folder](https://github.com/Cloud-CV/Fabrik/tree/master/example)</b>.
3. Open the <b>[Keras folder](https://github.com/Cloud-CV/Fabrik/tree/master/example/keras)</b> and add the new model in the form of .json file.
4. Then add your entry to the front-end by refering to the following example <b>[topbar.js](https://github.com/Cloud-CV/Fabrik/blob/master/ide/static/js/topBar.js)</b>, in this the ```id``` should be the name of your json without the extension.
```
<li><ModelElement importNet={this.props.importNet} framework="keras" id="Sample">sample</ModelElement></li> 
```
5. After making these changes, test if loading the model and exporting it to both or at least one framework is working fine and document it accordingly in your pull request.
6. Create a pull request for the same and get reviewed by the mentors.
Cheers!
