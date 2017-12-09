import React from 'react';
import PaneElement from './paneElement';

class Pane extends React.Component {
  constructor(props) {
        super(props);
        this.toggleClass= this.toggleClass.bind(this);
        this.state = {
            data: false,
            vision: false,
            recurrent: false,
            utility: false,
            activation: false,
            normalization: false,
            common: false,
            noise: false,
            loss: false
        };
    }
    toggleClass(layer) {
        var obj = {};
        obj[layer] = !this.state[layer];
        this.setState(obj);
    }
    render(){
      return (
        <div className="panel-group" id="menu" role="tablist" aria-multiselectable="true">
              <div className="panel panel-default">
                <div className="panel-heading" role="tab">
                    <span className="badge sidebar-badge" id="dataLayers"> </span>
                      Data
                    <a data-toggle="collapse" data-parent="#menu" href="#data" aria-expanded="false" aria-controls="data">
                      <span className={this.state.data ? 'glyphicon sidebar-dropdown glyphicon-menu-down': 
                      'glyphicon sidebar-dropdown glyphicon-menu-right'} onKeyUp={() => this.toggleClass('data')}></span>
                    </a>
                </div>
                <div id="data" className="panel-collapse collapse" role=" tabpanel">
                  <div className="panel-body">
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="ImageData">Image Data</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="Data">Data</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="HDF5Data">HDF5 Data</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="HDF5Output">HDF5 Output</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="Input">Input</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="WindowData">Window Data</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="MemoryData">Memory Data</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="DummyData">Dummy Data</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="Python">Python</PaneElement>
                  </div>
                </div>
              </div>
              <div className="panel panel-default">
                <div className="panel-heading" role="tab">
                    <span className="badge sidebar-badge" id="visionLayers"> </span>
                      Vision
                    <a data-toggle="collapse" data-parent="#menu" href="#vision" aria-expanded="false" aria-controls="vision">
                      <span className={this.state.vision ? 'glyphicon sidebar-dropdown glyphicon-menu-down': 
                      'glyphicon sidebar-dropdown glyphicon-menu-right'} onKeyUp={() => this.toggleClass('vision')}></span>
                    </a>
                </div>
                <div id="vision" className="panel-collapse collapse" role="tabpanel">
                  <div className="panel-body">
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="Convolution">Convolution</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="Pooling">Pool</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="Upsample">Upsample</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="LocallyConnected">Locally Connected</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="Crop">Crop</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="SPP">SPP</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="Deconvolution">Deconvolution</PaneElement>
                  </div>
                </div>
              </div>
              <div className="panel panel-default">
                <div className="panel-heading" role="tab">
                    <span className="badge sidebar-badge" id="recurrentLayers"> </span>
                      Recurrent
                    <a data-toggle="collapse" data-parent="#menu" href="#recurrent" aria-expanded="false" aria-controls="recurrent">
                      <span className={this.state.recurrent ? 'glyphicon sidebar-dropdown glyphicon-menu-down': 
                      'glyphicon sidebar-dropdown glyphicon-menu-right'} onKeyUp={() => this.toggleClass('recurrent')}></span>
                    </a>
                </div>
                <div id="recurrent" className="panel-collapse collapse" role="tabpanel">
                  <div className="panel-body">
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="Recurrent">Recurrent</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="RNN">RNN</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="GRU">GRU</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="LSTM">LSTM</PaneElement>
                  </div>
                </div>
              </div>
              <div className="panel panel-default">
                <div className="panel-heading" role="tab">
                    <span className="badge sidebar-badge" id="utilityLayers"> </span>
                      Utility
                    <a data-toggle="collapse" data-parent="#menu" href="#utility" aria-expanded="false" aria-controls="utility">
                      <span className={this.state.utility ? 'glyphicon sidebar-dropdown glyphicon-menu-down': 
                      'glyphicon sidebar-dropdown glyphicon-menu-right'} onKeyUp={() => this.toggleClass('utility')}></span>
                    </a>
                </div>
                <div id="utility" className="panel-collapse collapse" role="tabpanel">
                  <div className="panel-body">
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="Flatten">Flatten</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="Reshape">Reshape</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="BatchReindex">Batch Reindex</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="Split">Split</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="Concat">Concat</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="Eltwise">Eltwise</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="Filter">Filter</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="Reduction">Reduction</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="Silence">Silence</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="ArgMax">ArgMax</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="Softmax">Softmax</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="Permute">Permute</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="RepeatVector">Repeat Vector</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="Regularization">Regularization</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="Masking">Masking</PaneElement>
                  </div>
                </div>
              </div>
              <div className="panel panel-default">
                <div className="panel-heading" role="tab">
                    <span className="badge sidebar-badge" id="activationLayers"> </span>
                      Activation/Neuron
                    <a data-toggle="collapse" data-parent="#menu" href="#activation" aria-expanded="false" aria-controls="activation">
                      <span className={this.state.activation ? 'glyphicon sidebar-dropdown glyphicon-menu-down': 
                      'glyphicon sidebar-dropdown glyphicon-menu-right'} onKeyUp={() => this.toggleClass('activation')}></span>
                    </a>
                </div>
                <div id="activation" className="panel-collapse collapse" role="tabpanel">
                  <div className="panel-body">
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="ReLU">ReLU/Leaky-ReLU</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="PReLU">PReLU</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="ELU">ELU</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="ThresholdedReLU">Thresholded ReLU</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="SELU">SELU</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="Softplus">Softplus</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="Softsign">Softsign</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="Sigmoid">Sigmoid</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="TanH">TanH</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="HardSigmoid">Hard Sigmoid</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="AbsVal">Absolute Value</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="Power">Power</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="Exp">Exp</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="Log">Log</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="BNLL">BNLL</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="Threshold">Threshold</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="Bias">Bias</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="Scale">Scale</PaneElement>
                  </div>
                </div>
              </div>
              <div className="panel panel-default">
                <div className="panel-heading" role="tab">
                    <span className="badge sidebar-badge" id="normalizationLayers"> </span>
                      Normalization
                    <a data-toggle="collapse" data-parent="#menu" href="#normalization" aria-expanded="false" aria-controls="normalization">
                      <span className={this.state.normalization ? 'glyphicon sidebar-dropdown glyphicon-menu-down': 
                      'glyphicon sidebar-dropdown glyphicon-menu-right'} onKeyUp={() => this.toggleClass('normalization')}></span>
                    </a>
                </div>
                <div id="normalization" className="panel-collapse collapse" role="tabpanel">
                  <div className="panel-body">
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="LRN">LRN</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="MVN">MVN</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="BatchNorm">Batch Norm</PaneElement>
                  </div>
                </div>
              </div>
              <div className="panel panel-default">
                <div className="panel-heading" role="tab">
                    <span className="badge sidebar-badge" id="commonLayers"> </span>
                      Common
                    <a data-toggle="collapse" data-parent="#menu" href="#common" aria-expanded="false" aria-controls="common">
                      <span className={this.state.common ? 'glyphicon sidebar-dropdown glyphicon-menu-down': 
                      'glyphicon sidebar-dropdown glyphicon-menu-right'} onKeyUp={() => this.toggleClass('common')}></span>
                    </a>
                </div>
                <div id="common" className="panel-collapse collapse" role="tabpanel">
                  <div className="panel-body">
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="InnerProduct">Inner Product</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="Dropout">Dropout</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="Embed">Embed</PaneElement>
                  </div>
                </div>
              </div>
              <div className="panel panel-default">
                <div className="panel-heading" role="tab">
                    <span className="badge sidebar-badge" id="noiseLayers"> </span>
                      Noise
                    <a data-toggle="collapse" data-parent="#menu" href="#noise" aria-expanded="false" aria-controls="noise">
                      <span className={this.state.noise ? 'glyphicon sidebar-dropdown glyphicon-menu-down': 
                      'glyphicon sidebar-dropdown glyphicon-menu-right'} onKeyUp={() => this.toggleClass('noise')}></span>
                    </a>
                </div>
                <div id="noise" className="panel-collapse collapse" role="tabpanel">
                  <div className="panel-body">
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="GaussianNoise">Gaussian Noise</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="GaussianDropout">Gaussian Dropout</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="AlphaDropout">Alpha Dropout</PaneElement>
                  </div>
                </div>
              </div>
              <div className="panel panel-default">
                <div className="panel-heading" role="tab">
                    <span className="badge sidebar-badge" id="lossLayers"> </span>
                      Loss
                    <a data-toggle="collapse" data-parent="#menu" href="#loss" aria-expanded="false" aria-controls="loss">
                      <span className={this.state.loss ? 'glyphicon sidebar-dropdown glyphicon-menu-down': 
                      'glyphicon sidebar-dropdown glyphicon-menu-right'} onKeyUp={() => this.toggleClass('loss')}></span>
                    </a>
                </div>
                <div id="loss" className="panel-collapse collapse" role="tabpanel">
                  <div className="panel-body">
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="MultinomialLogisticLoss">Multinomial Logistic Loss</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="InfogainLoss">Infogain Loss</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="SoftmaxWithLoss">Softmax With Loss</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="EuclideanLoss">Euclidean Loss</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="HingeLoss">Hinge Loss</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="SigmoidCrossEntropyLoss">Sigmoid Cross Entropy Loss</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="Accuracy">Accuracy</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="ContrastiveLoss">Contrastive Loss</PaneElement>
                    <PaneElement onKeyUp={() => this.props.handleClick(event)} id="Python">Python</PaneElement>
                  </div>
                </div>
              </div>
        </div>


      );
  }
}
Pane.propTypes = {
  handleClick: React.PropTypes.func.isRequired
};
export default Pane;