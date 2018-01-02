<h1>Adding instructions to use exported model in Caffe</h1>

<b>Step-1</b>: See These Setup Instructions at [README](https://github.com/Cloud-CV/Fabrik/blob/master/README.md) to Know how to setup.
<br><b>Step-2</b>: You must a prototxt file to ensure that your model works.You Should add your prototxt file at
<b>[Example/Caffe](https://github.com/Cloud-CV/Fabrik/tree/master/example/caffe)</b> 
<br><b>Step-3</b>: Now, Add your Model to the front-end by after looking at this following example
in <b>[modelZoo.js](https://github.com/Cloud-CV/Fabrik/blob/master/ide/static/js/modelZoo.js)</b>,in this the ```id```
should be the name of your prototxt without the extension.
```
<li><ModelElement importNet={this.props.importNet} framework="caffe" id="sample">Sample</ModelElement></li>

```
<br><b>Last Step</b>:Check If Your Model is Working Properly and also if it is Exporting Properly, if it works then cheers you have 
successfulyexported a model in caffe!!!

