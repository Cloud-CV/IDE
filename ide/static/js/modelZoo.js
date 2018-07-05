import React from 'react';
import ModelElement from './modelElement';

class ModelZoo extends React.Component {

  constructor() {
      super();
      this.state = {
        toAddPerClick:5,
        open:0,
        text:"Load More",
        arrow:"glyphicon  glyphicon glyphicon-chevron-down",
        Recognition:{ 
                      toDisplay:6,
                      models:[
                              ["keras","v3","Inception V3"],
                              ["caffe","GoogleNet","GoogLeNet"],
                              ["caffe","alexnet","AlexNet"],
                              ["caffe","vgg16","VGG 16"],
                              ["caffe","densenet","DenseNet"],
                              ["caffe","lenet","MNIST LeNet"],
                              ["caffe","cifar10_full","Cifar10 CNN"],
                              ["keras","imdb_cnn_lstm","IMDB CNN LSTM"],
                              ["caffe","All_CNN","All CNN"],
                              ["caffe","Caffenet","Caffenet"],
                              ["caffe","speech_net","Speech Net"],
                              ["caffe","CNDS","CNDS"],
                              ["caffe","Gender","Gender"],
                              ["caffe","Age","Age"],
                              ["caffe","network_in_network","Network in Network"],
                              ["caffe","resnet101","ResNet 101"],
                              ["caffe","Squeezenet","Squeezenet"],
                              ["caffe","SENet","SENet"]
                            ] 
                  },
        Detection:{ 
                    toDisplay:3,
                    models: [
                              ["caffe","yolo_net","YOLONet"],
                              ["caffe","vanilla","Vanilla CNN"],
                              ["caffe","R-CNN","R-CNN"],
                              ["caffe","fcn","FCN32 Pascal"],
                              ["caffe","48Net","48Net"]
                            ]
                  },
        Seq2Seq:{
                  toDisplay:3,
                  models:[
                          ["keras","textGeneration","Text Generation"],
                          ["keras","seq2seq_lang","Seq2Seq Translation"],
                          ["caffe","pix2pix","Pix2Pix"]
                        ]

                },
        VQA    :{
                  toDisplay:3,
                  models:[
                            ["keras","VQA","VQA"],
                            ["keras","VQA2","VQA2"],
                            ["caffe","mlpVQA","VQS"]
                          ]
                },
        Segmentation:{
                    toDisplay:2,
                    models:[
                              ["caffe","fcn2","Semantic Segmentation"],
                              ["keras","ZF_UNET_224","UNET"]
                            ]
                }
      };
      this.handleClick = this.handleClick.bind(this);
    
  }

  handleClick(event){
    if(this.state.open){
      this.setState({
        text:"Load More",
        arrow:"glyphicon  glyphicon glyphicon-chevron-down"
      });
    }
    else{
      this.setState({
        text:"Load Less",
        arrow:"glyphicon  glyphicon glyphicon-chevron-up"
      }); 
    }
    if (event.target.id =="change")
      this.setState({open:1-this.state.open});
  }

  render() {

    const startIndex = 0;

    var category = this.state.Recognition;
    var finalIndex = category.toDisplay*(1 - this.state.open)+this.state.open*category.models.length;
    var sliced = category.models.slice(startIndex, finalIndex);
    const renderRecognition = sliced.map((category) => {
      return (  
        <div>
          <ModelElement importNet = {this.props.importNet} framework =
                      {category[0]} id = {category[1]}> {category[2]} </ModelElement>
          <br/>
        </div>
        );
    });

    category = this.state.Detection;
    finalIndex = category.toDisplay*(1 - this.state.open)+this.state.open*category.models.length;
    sliced = category.models.slice(startIndex, finalIndex);
    const renderDetection = sliced.map((category) => {
      return (  
        <div>
          <ModelElement importNet = {this.props.importNet} framework =
                      {category[0]} id = {category[1]}> {category[2]} </ModelElement>
          <br/>
        </div>
        );
    });

    category = this.state.Seq2Seq;
    finalIndex = category.toDisplay*(1 - this.state.open)+this.state.open*category.models.length;
    sliced = category.models.slice(startIndex, finalIndex);
    const renderSeq2Seq = sliced.map((category) => {
      return (  
        <div>
          <ModelElement importNet = {this.props.importNet} framework =
                      {category[0]} id = {category[1]}> {category[2]} </ModelElement>
          <br/>
        </div>
        );
    });

    category = this.state.Segmentation;
    finalIndex = category.toDisplay*(1 - this.state.open)+this.state.open*category.models.length;
    sliced = category.models.slice(startIndex, finalIndex);
    const renderSegmentation = sliced.map((category) => {
      return (  
        <div>
          <ModelElement importNet = {this.props.importNet} framework =
                      {category[0]} id = {category[1]}> {category[2]} </ModelElement>
          <br/>
        </div>
        );
    });

    category = this.state.VQA;
    finalIndex = category.toDisplay*(1 - this.state.open)+this.state.open*category.models.length;
    sliced = category.models.slice(startIndex, finalIndex);
    const renderVQA = sliced.map((category) => {
      return (  
        <div>
          <ModelElement importNet = {this.props.importNet} framework =
                      {category[0]} id = {category[1]}> {category[2]} </ModelElement>
          <br/>
        </div>
        );
    });


    return (
      <div className="zoo-modal">
        <div className="centered-zoo-modal">
          <div className="zoo-modal-model">
            <h3 className="zoo-modal-text">Recognition</h3>

            {renderRecognition}

          </div>

          <div className="zoo-modal-model">
            <h3 className="zoo-modal-text">Detection</h3>
              {renderDetection}

            <h3 className="zoo-modal-text">Retrieval</h3>
              <ModelElement importNet={this.props.importNet} framework="caffe" id="siamese_mnist">MNIST Siamese</ModelElement>
          </div>


          <div className="zoo-modal-model">
            <h3 className="zoo-modal-text">Seq2Seq</h3>
              {renderSeq2Seq}

          </div>

          <div className="zoo-modal-model">
          
            <h3 className="zoo-modal-text">Caption</h3>
            <ModelElement importNet={this.props.importNet} framework="caffe" id="CoCo_Caption">CoCo Caption</ModelElement>
            
            <h3 className="zoo-modal-text">Segmentation</h3>
                {renderSegmentation}
  
            <h3 className="zoo-modal-text">VQA</h3>
              {renderVQA}
          </div>

          <button className="import-textbox-button btn btn-default col-md-2 col-md-offset-5"  
              id="change" onClick={this.handleClick}>{this.state.text} 
            &nbsp;<span className={this.state.arrow}  aria-hidden="true"></span>
          </button>   
        </div>
      </div>
    );
  }
}


ModelZoo.propTypes = {
  importNet: React.PropTypes.func
};

export default ModelZoo;
