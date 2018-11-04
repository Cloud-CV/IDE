import React from 'react';
import $ from 'jquery'

class FilterBar extends React.Component {
  constructor(props) {
        super(props);
        this.changeEvent= this.changeEvent.bind(this);
    }

changeEvent(cbid) {
      var KerasLayers = ["RNN_Button", "GRU_Button", "LSTM_Button", "Embed_Button", "ELU_Button", "Sigmoid_Button",
      "ThresholdedReLU", "ReLU_Button", "PReLU_Button", "Softmax_Button", "BatchNorm_Button", "SELU_Button",
      "GaussianNoise_Button", "GaussianDropout_Button", "AlphaDropout_Button", "TimeDistributed_Button", "TanH_Button",
      "Bidirectional_Button", "RepeatVector_Button", "Masking_Button", "Permute_Button", "InnerProduct_Button",
      "Deconvolution_Button", "Regularization_Button", "Softsign_Button", "Upsample_Button", "Pooling_Button",
      "LocallyConnected_Button", "Input_Button", "Convolution_Button", "LRN_Button", "DepthwiseConv_Button", "Flatten_Button",
      "Reshape_Button", "Concat_Button", "Softplus_Button", "HardSigmoid_Button", "Dropout_Button", "Eltwise_Button"];
      var TensorFlowLayers = ["Input_Button", "Convolution_Button", "Pooling_Button", "DepthwiseConv_Button", "InnerProduct_Button",
      "LRN_Button", "Concat_Button", "Flatten_Button", "BatchNorm_Button", "Deconvolution_Button", "Sigmoid_Button", 
      "Softplus_Button", "Softsign_Button", "ELU_Button", "ReLU_Button", "Softmax_Button", "TanH_Button", "SELU_Button",
      "Dropout_Button", "Eltwise_Button"];
      var CaffeLayers = ["ImageData_Button", "HDF5Data_Button", "HDF5Output_Button", "Input_Button", "WindowData_Button",
      "MemoryData_Button", "DummyData_Button", "Convolution_Button", "Pooling_Button", "SPP_Button", "Deconvolution_Button",
      "Recurrent_Button", "RNN_Button", "LSTM_Button", "LRN_Button", "MVN_Button", "BatchNorm_Button",
      "InnerProduct_Button", "Dropout_Button", "Embed_Button", "ReLU_Button", "PReLU_Button", "ELU_Button",
      "Sigmoid_Button", "TanH_Button", "AbsVal_Button", "Power_Button", "Exp_Button", "Log_Button", "BNLL_Button",
      "Threshold_Button", "Bias_Button", "Scale_Button", "Flatten_Button", "ThresholdedReLU", "Python_Button",
      "Reshape_Button", "BatchReindex_Button", "Split_Button", "Concat_Button", "Eltwise_Button", "Filter_Button",
      "Reduction_Button", "Silence_Button", "ArgMax_Button", "Softmax_Button", "MultinomialLogisticLoss_Button",
      "InfogainLoss_Button", "SoftmaxWithLoss_Button", "EuclideanLoss_Button", "HingeLoss_Button", "Slice_Button",
      "SigmoidCrossEntropyLoss_Button", "Accuracy_Button", "ContrastiveLoss_Button", "Data_Button", "Crop_Button"];
      var CheckBoxA = document.getElementById("CheckBoxA");
      var CheckBoxB = document.getElementById("CheckBoxB");
      var CheckBoxC = document.getElementById("CheckBoxC");
      var visible = [];

    let CheckBox = document.getElementById(cbid);
    CheckBox.checked = !CheckBox.checked;

      if(CheckBoxA.checked == false & CheckBoxB.checked == false & CheckBoxC.checked == false){
        for (let elem of $('.drowpdown-button')) {
        elem.classList.remove("hide");
        }            
      }
      if (CheckBoxA.checked == true){
         visible = visible.concat(KerasLayers);
      }
      if (CheckBoxB.checked == true){
         visible = visible.concat(TensorFlowLayers);
      }
      if (CheckBoxC.checked == true){
         visible = visible.concat(CaffeLayers);
      }
      
            for (let elem of $('.drowpdown-button')) {
                for (let j = 0; j < visible.length; j++){
                let id = elem.id;
                if(id == visible[j]){
                    elem.classList.remove("hide");
                    j = visible.length + 1;
                }else{
                elem.classList.add("hide");
                }
            }
        }

    }

    render() {
      return (
            <div>
              <div className="form-group pull-right filter-button">
                <div className="dropdown">
                  <button id="topbar-icon" className="btn btn-default dropdown-toggle form-control" data-toggle="dropdown">
                    <span className="glyphicon glyphicon-filter" aria-hidden="true"></span>
                  </button>
                  <ul className="dropdown-menu pull-right">
                    <li>
                        <a className="btn" onClick={this.changeEvent.bind(this, "CheckBoxA")}>
                        <label className="filter">Keras</label>
                        <input type="checkbox" id="CheckBoxA" value="A" onClick={this.changeEvent.bind(this, "CheckBoxA")} />
                        </a>
                    </li>
                    <li>
                        <a className="btn" onClick={this.changeEvent.bind(this, "CheckBoxB")}>
                        <label className="filter">Tensorflow</label>
                        <input type="checkbox" id="CheckBoxB" value="B" onClick={this.changeEvent.bind(this, "CheckBoxB")} />
                        </a>
                    </li>
                    <li>
                        <a className="btn" onClick={this.changeEvent.bind(this, "CheckBoxC")}>
                        <label className="filter">Caffe</label>
                        <input type="checkbox" id="CheckBoxC" value="C" onClick={this.changeEvent.bind(this, "CheckBoxC")} />
                        </a>
                    </li>
                  </ul>
                </div>
              </div>
              </div>
             )
    }
}
export default FilterBar;
