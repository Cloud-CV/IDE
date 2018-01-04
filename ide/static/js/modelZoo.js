import React from 'react';
import ModelElement from './modelElement';

class ModelZoo extends React.Component {
  render() {
    var toDisplay = [];
    var zooData =  {
        'Recognition': [{
          'id': 'alexnet',
          'framework': 'caffe',
          'name': 'AlexNet'
        }, {
          'id': 'All_CNN',
          'framework': 'caffe',
          'name': 'All CNN'
        }, {
          'id': 'vgg16',
          'framework': 'caffe',
          'name': 'VGG 16'
        }, {
          'id': 'densenet',
          'framework': 'caffe',
          'name': 'DenseNet'
        }, {
          'id': 'GoogleNet',
          'framework': 'caffe',
          'name': 'GoogLeNet'
        }, {
          'id': 'resnet101',
          'framework': 'caffe',
          'name': 'ResNet 101'
        }, {
          'id': 'v3',
          'framework': 'keras',
          'name': 'Inception V3'
        }, {
          'id': 'Squeezenet',
          'framework': 'caffe',
          'name': 'Squeezenet'
        }],
        'Detection': [{
          'id': 'fcn',
          'framework': 'caffe',
          'name': 'FCN32 Pascal'
        }, {
          'id': 'yolo_net',
          'framework': 'caffe',
          'name': 'YOLONet'
        }, {
          'id': 'HED',
          'framework': 'caffe',
          'name': 'HED'
        }, {
          'id': 'siamese_mnist',
          'framework': 'caffe',
          'name': 'MNIST Siamese'
        }],
        'Seq2Seq': [{
          'id': 'textGeneration',
          'framework': 'keras',
          'name': 'Text Generation'
        }, {
          'id': 'seq2seq_lang',
          'framework': 'keras',
          'name': 'Seq2Seq Translation'
        }],
        'Caption': [{
          'id': 'CoCo_Caption',
          'framework': 'caffe',
          'name': 'CoCo Caption'
        }, {
          'id': 'fcn2',
          'framework': 'caffe',
          'name': 'Semantic Segmentation'
        }],
        'VQA': [{
          'id': 'VQA',
          'framework': 'keras',
          'name': 'VQA'
        }]
      }
    Object.keys(zooData).map(category =>
      toDisplay.push(
      <div key={category} className="zoo-modal-model">
          <h3 className="zoo-modal-text">{category}</h3>
            {Object.keys(zooData[category]).map(modelEl =>
              <div key={zooData[category][modelEl]['id']}>
              <ModelElement importNet={this.props.importNet} 
              framework={zooData[category][modelEl]['framework']} 
              id={zooData[category][modelEl]['id']}> 
              {zooData[category][modelEl]['name']} 
              </ModelElement>
              <br></br>
              </div>
            )}
      )
    )
    return (
      <div className="zoo-modal"><div className="centered-zoo-modal"> {toDisplay} </div></div>
      )
  }
}

ModelZoo.propTypes = {
  importNet: React.PropTypes.func
};

export default ModelZoo;
