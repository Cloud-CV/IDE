<h1>Adding New Model - Caffe</h1>

1. For Setup Instructions. - https://github.com/Cloud-CV/Fabrik/blob/master/README.md
2. Open the <b>Example folder</b>.
3. Open the <b>Caffe folder</b> and add the new model in the form of .prototxt file.
4. Then add your entry to the frontend by adding the following line to <b>topbar.js</b>, in this the id should be the name of your prototxt without the extension.
     - The location of topbar.js is <b>ide/static/js/topbar.js</b> 
```
<li><ModelElement importNet={this.props.importNet} framework="caffe" id="sample">sample</ModelElement></li>

```
5. After making these changes, test if loading the model and exporting it to both or at least one framework is working fine.
6. Create a pull request for the same and get reviewed by the mentors.
Cheers!

<h1>Adding New Model - Keras </h1>

1. For Setup Instructions. - https://github.com/Cloud-CV/Fabrik/blob/master/README.md
2. Open the <b>Example folder.</b>
3. Open the <b>Keras folder</b> and add the new model in the form of .json file.
4. Then add your entry to the frontend by adding the following line to <b>topbar.js</b>, in this the id should be the name of your json without the extension.
    - The location of topbar.js is <b>ide/static/js/topbar.js</b>
```
<li><ModelElement importNet={this.props.importNet} framework="keras" id="sample">sample</ModelElement></li> 
```
5. After making these changes, test if loading the model and exporting it to both or at least one framework is working fine.
6. Create a pull request for the same and get reviewed by the mentors.
Cheers!
