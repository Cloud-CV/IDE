1) Go to Github and fork the Fabrik repository.
2) Open the example folder.
3) Open the keras folder and add the new model in the form of .json file.
4) Then add your entry to the frontend by:-
i) Adding a line <li><ModelElement importNet={this.props.importNet} framework="keras" id="sample">sample</ModelElement></li> to topbar.js, in this the id should be the name of your json without the extension.
ii) Location of topbar.js is ide/static/js/topbar.js 
5) After making these changes, test if loading the model and exporting it to both or at least one framework is working fine.
6) Create a pull request for the same and get reviewed by the mentors.
Cheers!