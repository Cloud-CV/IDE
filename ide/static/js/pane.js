import React from 'react';
import PaneElement from './paneElement';

function Pane() {
  return (

<li className="dropdown" id="pane-dropdown" style={{paddingTop:'4px'}}>
  <button data-toggle="dropdown" className="dropdown-toggle" aria-haspopup="true" 
  aria-expanded="true"><span className="glyphicon glyphicon-plus-sign" style={{fontSize:'24px'}}></span></button>

  <ul className="dropdown-menu" id="addLayerDropdown">
    <li className="dropdown-submenu">
      <a tabIndex="-1" href="#">Data Layers</a>
      <ul className="dropdown-menu">
        <li><PaneElement id="Data">Data</PaneElement></li>
        <li><PaneElement id="Input">Input</PaneElement></li>
        <li><PaneElement id="HDF5Data">HDF5Data</PaneElement></li>
      </ul>
    </li>
    <li className="dropdown-submenu">
      <a tabIndex="-1" href="#">Vision Layers</a>
      <ul className="dropdown-menu">
        <li><PaneElement id="Convolution">Convolution</PaneElement></li>
        <li><PaneElement id="Crop">Crop</PaneElement></li>
        <li><PaneElement id="Deconvolution">Deconvolution</PaneElement></li>
        <li><PaneElement id="Pooling">Pool</PaneElement></li>
      </ul>
    </li>
    <li className="dropdown-submenu">
      <a tabIndex="-1" href="#">Recurrent Layers</a>
      <ul className="dropdown-menu">
        <li><PaneElement id="LSTM">LSTM</PaneElement></li>
      </ul>
    </li>
    <li className="dropdown-submenu">
      <a tabIndex="-1" href="#">Common Layers</a>
      <ul className="dropdown-menu">
        <li><PaneElement id="Dropout">Dropout</PaneElement></li>
        <li><PaneElement id="Embed">Embed</PaneElement></li>
        <li><PaneElement id="InnerProduct">Inner Product</PaneElement></li>
      </ul>
    </li>
    <li className="dropdown-submenu">
      <a tabIndex="-1" href="#">Normalisation Layers</a>
      <ul className="dropdown-menu">
        <li><PaneElement id="BatchNorm">BatchNorm</PaneElement></li>
        <li><PaneElement id="LRN">LRN</PaneElement></li>
      </ul>
    </li>
    <li className="dropdown-submenu">
      <a tabIndex="-1" href="#">Activation/Neuron Layers</a>
      <ul className="dropdown-menu">
        <li><PaneElement id="ReLU">ReLU</PaneElement></li>
        <li><PaneElement id="Scale">Scale</PaneElement></li>
      </ul>
    </li>
    <li className="dropdown-submenu">
      <a tabIndex="-1" href="#">Utility Layers</a>
      <ul className="dropdown-menu">
        <li><PaneElement id="Concat">Concat</PaneElement></li>
        <li><PaneElement id="Eltwise">Eltwise</PaneElement></li>
        <li><PaneElement id="Reshape">Reshape</PaneElement></li>
        <li><PaneElement id="Softmax">Softmax</PaneElement></li>
      </ul>
    </li>
    <li className="dropdown-submenu">
      <a tabIndex="-1" href="#">Loss Layers</a>
      <ul className="dropdown-menu">
        <li><PaneElement id="Accuracy">Accuracy</PaneElement></li>
        <li><PaneElement id="SoftmaxWithLoss">Softmax With Loss</PaneElement></li>
      </ul>
    </li>
    <li><PaneElement id="python-dropdown">Python</PaneElement></li>    
  </ul>
</li>


  );
}

export default Pane;