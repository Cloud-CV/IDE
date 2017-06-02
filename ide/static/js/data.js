export default {
  /* ********** Data Layers ********** */
  Data: {
    name: 'data',
    color: '#673ab7',
    endpoint: {
      src: ['Bottom'],
      trg: []
    },
    params: {
      source: {
        name: 'Data source',
        value: '',
        type: 'text',
        required: true
      },
      batch_size: {
        name: 'Batch size',
        value: '',
        type: 'number',
        required: true
      },
      backend: {
        name: 'Backend',
        value: 'LMDB',
        type: 'select',
        options: ['LMDB', 'LEVELDB'],
        required: true
      },
      scale: {
        name: 'Scale',
        value: '',
        type: 'float',
        required: false
      }
    },
    props: {
      name: {
        name: 'Name',
        value: '',
        type: 'text'
      }
    },
    learn: false
  },
  HDF5Data: {
    name: 'hdf5data',
    color: '#673ab7',
    endpoint: {
      src: ['Bottom'],
      trg: []
    },
    params: {
      source: {
        name: 'HDF5 Data source',
        value: '',
        type: 'text',
        required: true
      },
      batch_size: {
        name: 'Batch size',
        value: '',
        type: 'number',
        required: true
      }
    },
    props: {
      name: {
        name: 'Name',
        value: '',
        type: 'text'
      }
    },
    learn: false
  },
  Input: {
    name: 'input',
    color: '#673ab7',
    endpoint: {
      src: ['Bottom'],
      trg: []
    },
    params: {
      dim: {
        name: 'Dim',
        value: '',
        type: 'text',
        required: true
      }
    },
    props: {
      name: {
        name: 'Name',
        value: '',
        type: 'text'
      }
    },
    learn: false
  },
  /* ********** Vision Layers ********** */
  Convolution: {
    name: 'conv',
    color: '#3f51b5',
    endpoint: {
      src: ['Bottom'],
      trg: ['Top']
    },
    params: {
      num_output: {
        name: 'No of outputs',
        value: '',
        type: 'number',
        required: true
      },
      kernel_h: {
        name: 'Kernel height',
        value: '',
        type: 'number',
        required: true
      },
      kernel_w: {
        name: 'Kernel width',
        value: '',
        type: 'number',
        required: true
      },
      stride_h: {
        name: 'Stride height',
        value: '',
        type: 'number',
        required: false
      },
      stride_w: {
        name: 'Stride width',
        value: '',
        type: 'number',
        required: false
      },
      pad_h: {
        name: 'Padding height',
        value: '',
        type: 'number',
        required: false
      },
      pad_w: {
        name: 'Padding width',
        value: '',
        type: 'number',
        required: false
      },
      weight_filler: {
        name: 'Weight filler',
        value: 'xavier',
        type: 'select',
        options: ['xavier', 'constant'],
        required: false
      },
      bias_filler: {
        name: 'Bias filler',
        value: 'constant',
        type: 'select',
        options: ['xavier', 'constant'],
        required: false
      }
    },
    props: {
      name: {
        name: 'Name',
        value: '',
        type: 'text'
      }
    },
    learn: true
  },
  Pooling: {
    name: 'pool',
    color: '#3f51b5',
    endpoint: {
      src: ['Bottom'],
      trg: ['Top']
    },
    params: {
      pad_h: {
        name: 'Padding height',
        value: '',
        type: 'number',
        required: false
      },
      pad_w: {
        name: 'Padding width',
        value: '',
        type: 'number',
        required: false
      },
      kernel_h: {
        name: 'Kernel height',
        value: '',
        type: 'number',
        required: true
      },
      kernel_w: {
        name: 'Kernel width',
        value: '',
        type: 'number',
        required: true
      },
      stride_h: {
        name: 'Stride height',
        value: '',
        type: 'number',
        required: false
      },
      stride_w: {
        name: 'Stride width',
        value: '',
        type: 'number',
        required: false
      },
      pool: {
        name: 'Pooling method',
        value: 'MAX',
        type: 'select',
        options: ['MAX', 'AVE', 'STOCHASTIC'],
        required: false
      }
    },
    props: {
      name: {
        name: 'Name',
        value: '',
        type: 'text'
      }
    },
    learn: false
  },
  Crop: {
    name: 'crop',
    color: '#3f51b5',
    endpoint: {
      src: ['Bottom'],
      trg: ['Top']
    },
    params: {
      axis: {
        name: 'axis',
        value: '2',
        type: 'number',
        required: false
      },
      offset: {
        name: 'offset',
        value: '0',
        type: 'number',
        required: false
      }
    },
    props: {
      name: {
        name: 'Name',
        value: '',
        type: 'text'
      }
    },
    learn: false
  },
  Deconvolution: {
    name: 'deconv',
    color: '#3f51b5',
    endpoint: {
      src: ['Bottom'],
      trg: ['Top']
    },
    params: {
      num_output: {
        name: 'No of outputs',
        value: '',
        type: 'number',
        required: true
      },
      kernel_h: {
        name: 'Kernel height',
        value: '',
        type: 'number',
        required: true
      },
      kernel_w: {
        name: 'Kernel width',
        value: '',
        type: 'number',
        required: true
      },
      stride_h: {
        name: 'Stride height',
        value: '',
        type: 'number',
        required: false
      },
      stride_w: {
        name: 'Stride width',
        value: '',
        type: 'number',
        required: false
      },
      pad_h: {
        name: 'Padding height',
        value: '',
        type: 'number',
        required: false
      },
      pad_w: {
        name: 'Padding width',
        value: '',
        type: 'number',
        required: false
      },
      weight_filler: {
        name: 'Weight filler',
        value: 'xavier',
        type: 'select',
        options: ['xavier', 'constant'],
        required: false
      },
      bias_filler: {
        name: 'Bias filler',
        value: 'constant',
        type: 'select',
        options: ['xavier', 'constant'],
        required: false
      }
    },
    props: {
      name: {
        name: 'Name',
        value: '',
        type: 'text'
      }
    },
    learn: true
  },
  /* ********** Recurrent Layers ********** */
  LSTM: {
    name: 'lstm',
    color: '#3f51b5',
    endpoint: {
      src: ['Bottom'],
      trg: ['Top']
    },
    params: {
      num_output: {
        name: 'No of outputs',
        value: '',
        type: 'number',
        required: true
      },
      weight_filler: {
        name: 'Weight filler',
        value: 'xavier',
        type: 'select',
        options: ['xavier', 'constant'],
        required: false
      },
      bias_filler: {
        name: 'Bias filler',
        value: 'constant',
        type: 'select',
        options: ['xavier', 'constant'],
        required: false
      }
    },
    props: {
      name: {
        name: 'Name',
        value: '',
        type: 'text'
      }
    },
    learn: true
  },
  /* ********** Common Layers ********** */
  InnerProduct: {
    name: 'fc',
    color: '#ff9800',
    endpoint: {
      src: ['Bottom'],
      trg: ['Top']
    },
    params: {
      num_output: {
        name: 'No of outputs',
        value: '',
        type: 'number',
        required: true
      },
      weight_filler: {
        name: 'Weight filler',
        value: 'xavier',
        type: 'select',
        options: ['xavier', 'constant'],
        required: false
      },
      bias_filler: {
        name: 'Bias filler',
        value: 'constant',
        type: 'select',
        options: ['xavier', 'constant'],
        required: false
      }
    },
    props: {
      name: {
        name: 'Name',
        value: '',
        type: 'text'
      }
    },
    learn: true
  },
  Dropout: {
    name: 'dropout',
    color: '#ff9800',
    endpoint: {
      src: ['Bottom'],
      trg: ['Top']
    },
    params: {
      inplace: {
        name: 'Inplace operation',
        value: true,
        type: 'checkbox',
        required: false
      }
    },
    props: {
      name: {
        name: 'Name',
        value: '',
        type: 'text'
      }
    },
    learn: false
  },
  Embed: {
    name: 'embed',
    color: '#ff9800',
    endpoint: {
      src: ['Bottom'],
      trg: ['Top']
    },
    params: {
      num_output: {
        name: 'No of outputs',
        value: '',
        type: 'number',
        required: true
      },
      weight_filler: {
        name: 'Weight filler',
        value: 'xavier',
        type: 'select',
        options: ['xavier', 'constant'],
        required: false
      },
      bias_term: {
        name: 'Bias Term',
        value: 'False',
        type: 'select',
        options: ['True', 'False'],
        required: false
      },
      input_dim: {
        name: 'Input Dimensions',
        value: '',
        type: 'number',
        required: true
      }
    },
    props: {
      name: {
        name: 'Name',
        value: '',
        type: 'text'
      }
    },
    learn: true
  },
  /* ********** Normalisation Layers ********** */
  BatchNorm: {
    name: 'batchnorm',
    color: '#ffeb3b',
    endpoint: {
      src: ['Bottom'],
      trg: ['Top']
    },
    params: {
      use_global_stats: {
        name: 'Use Global Stats',
        value: '',
        type: 'select',
        options: ['true', 'false'],
        required: true
      }
    },
    props: {
      name: {
        name: 'Name',
        value: '',
        type: 'text'
      }
    },
    learn: true
  },
  LRN: {
    name: 'lrn',
    color: '#ffeb3b',
    endpoint: {
      src: ['Bottom'],
      trg: ['Top']
    },
    params: {
    },
    props: {
      name: {
        name: 'Name',
        value: '',
        type: 'text'
      }
    },
    learn: false
  },
  /* ********** Activation/Neuron Layers ********** */
  ReLU: {
    name: 'relu',
    color: '#009688',
    endpoint: {
      src: ['Bottom'],
      trg: ['Top']
    },
    params: {
      inplace: {
        name: 'Inplace operation',
        value: true,
        type: 'checkbox',
        required: false
      },
      negative_slope: {
        name: 'Negative slope',
        value: 0,
        type: 'number',
        required: false
      }
    },
    props: {
      name: {
        name: 'Name',
        value: '',
        type: 'text'
      }
    },
    learn: false
  },
  PReLU: {
    name: 'prelu',
    color: '#009688',
    endpoint: {
      src: ['Bottom'],
      trg: ['Top']
    },
    params: {
      inplace: {
        name: 'Inplace operation',
        value: true,
        type: 'checkbox',
        required: false
      },
      channel_shared: {
        name: 'Channel Shared',
        value: false,
        type: 'checkbox',
        required: false
      }
    },
    props: {
      name: {
        name: 'Name',
        value: '',
        type: 'text'
      }
    },
    learn: true
  },
  ELU: {
    name: 'elu',
    color: '#009688',
    endpoint: {
      src: ['Bottom'],
      trg: ['Top']
    },
    params: {
      inplace: {
        name: 'Inplace operation',
        value: true,
        type: 'checkbox',
        required: false
      },
      alpha: {
        name: 'Alpha',
        value: 1,
        type: 'float',
        required: false
      }
    },
    props: {
      name: {
        name: 'Name',
        value: '',
        type: 'text'
      }
    },
    learn: false
  },
  Sigmoid: {
    name: 'sigmoid',
    color: '#009688',
    endpoint: {
      src: ['Bottom'],
      trg: ['Top']
    },
    params: {
      inplace: {
        name: 'Inplace operation',
        value: true,
        type: 'checkbox',
        required: false
      }
    },
    props: {
      name: {
        name: 'Name',
        value: '',
        type: 'text'
      }
    },
    learn: false
  },
  TanH: {
    name: 'tanh',
    color: '#009688',
    endpoint: {
      src: ['Bottom'],
      trg: ['Top']
    },
    params: {
      inplace: {
        name: 'Inplace operation',
        value: true,
        type: 'checkbox',
        required: false
      }
    },
    props: {
      name: {
        name: 'Name',
        value: '',
        type: 'text'
      }
    },
    learn: false
  },
  AbsVal: {
    name: 'absval',
    color: '#009688',
    endpoint: {
      src: ['Bottom'],
      trg: ['Top']
    },
    params: {
      inplace: {
        name: 'Inplace operation',
        value: true,
        type: 'checkbox',
        required: false
      }
    },
    props: {
      name: {
        name: 'Name',
        value: '',
        type: 'text'
      }
    },
    learn: false
  },
  Power: {
    name: 'power',
    color: '#009688',
    endpoint: {
      src: ['Bottom'],
      trg: ['Top']
    },
    params: {
      inplace: {
        name: 'Inplace operation',
        value: true,
        type: 'checkbox',
        required: false
      },
      power: {
        name: 'Power',
        value: 1.0,
        type: 'float',
        required: false
      },
      scale: {
        name: 'Scale',
        value: 1.0,
        type: 'float',
        required: false
      },
      shift: {
        name: 'Shift',
        value: 0.0,
        type: 'float',
        required: false
      }
    },
    props: {
      name: {
        name: 'Name',
        value: '',
        type: 'text'
      }
    },
    learn: false
  },
  Exp: {
    name: 'exp',
    color: '#009688',
    endpoint: {
      src: ['Bottom'],
      trg: ['Top']
    },
    params: {
      inplace: {
        name: 'Inplace operation',
        value: true,
        type: 'checkbox',
        required: false
      },
      base: {
        name: 'Base',
        value: -1.0,
        type: 'float',
        required: false
      },
      scale: {
        name: 'Scale',
        value: 1.0,
        type: 'float',
        required: false
      },
      shift: {
        name: 'Shift',
        value: 0.0,
        type: 'float',
        required: false
      }
    },
    props: {
      name: {
        name: 'Name',
        value: '',
        type: 'text'
      }
    },
    learn: false
  },
  Log: {
    name: 'log',
    color: '#009688',
    endpoint: {
      src: ['Bottom'],
      trg: ['Top']
    },
    params: {
      inplace: {
        name: 'Inplace operation',
        value: true,
        type: 'checkbox',
        required: false
      },
      base: {
        name: 'Base',
        value: -1.0,
        type: 'float',
        required: false
      },
      scale: {
        name: 'Scale',
        value: 1.0,
        type: 'float',
        required: false
      },
      shift: {
        name: 'Shift',
        value: 0.0,
        type: 'float',
        required: false
      }
    },
    props: {
      name: {
        name: 'Name',
        value: '',
        type: 'text'
      }
    },
    learn: false
  },
  BNLL: {
    name: 'bnll',
    color: '#009688',
    endpoint: {
      src: ['Bottom'],
      trg: ['Top']
    },
    params: {
      inplace: {
        name: 'Inplace operation',
        value: true,
        type: 'checkbox',
        required: false
      }
    },
    props: {
      name: {
        name: 'Name',
        value: '',
        type: 'text'
      }
    },
    learn: false
  },
  Threshold: {
    name: 'threshold',
    color: '#009688',
    endpoint: {
      src: ['Bottom'],
      trg: ['Top']
    },
    params: {
      inplace: {
        name: 'Inplace operation',
        value: true,
        type: 'checkbox',
        required: false
      },
      threshold: {
        name: 'threshold',
        value: 0,
        type: 'float',
        required: false
      }
    },
    props: {
      name: {
        name: 'Name',
        value: '',
        type: 'text'
      }
    },
    learn: false
  },
  Scale: {
    name: 'scale',
    color: '#009688',
    endpoint: {
      src: ['Bottom'],
      trg: ['Top']
    },
    params: {
      bias_term: {
        name: 'Bias term',
        value: '',
        type: 'select',
        options: ['true', 'false'],
        required: true
      }
    },
    props: {
      name: {
        name: 'Name',
        value: '',
        type: 'text'
      }
    },
    learn: true
  },
  /* ********** Utility Layers ********** */
  Reshape: {
    name: 'reshape',
    color: '#03a9f4',
    endpoint: {
      src: ['Bottom'],
      trg: ['Top']
    },
    params: {
      dim: {
        name: 'Dim',
        value: '',
        type: 'text',
        required: true
      }
    },
    props: {
      name: {
        name: 'Name',
        value: '',
        type: 'text'
      }
    },
    learn: false
  },
  Concat: {
    name: 'concat',
    color: '#03a9f4',
    endpoint: {
      src: ['Bottom'],
      trg: ['Top']
    },
    params: {
    },
    props: {
      name: {
        name: 'Name',
        value: '',
        type: 'text'
      }
    },
    learn: false
  },
  Eltwise: {
    name: 'eltwise',
    color: '#03a9f4',
    endpoint: {
      src: ['Bottom'],
      trg: ['Top']
    },
    params: {
      operation: {
        name: 'Eltwise method',
        value: 'SUM',
        type: 'select',
        options: ['SUM', 'PROD', 'Max'],
        required: false
      }
    },
    props: {
      name: {
        name: 'Name',
        value: '',
        type: 'text'
      }
    },
    learn: false
  },
  Softmax: {
    name: 'softmax',
    color: '#03a9f4',
    endpoint: {
      src: [],
      trg: ['Top']
    },
    params: {
    },
    props: {
      name: {
        name: 'Name',
        value: '',
        type: 'text'
      }
    },
    learn: false
  },
  /* ********** Loss Layers ********** */
  SoftmaxWithLoss: {
    name: 'loss',
    color: '#f44336',
    endpoint: {
      src: [],
      trg: ['Top']
    },
    params: {
    },
    props: {
      name: {
        name: 'Name',
        value: '',
        type: 'text'
      }
    },
    learn: false
  },
  Accuracy: {
    name: 'acc',
    color: '#f44336',
    endpoint: {
      src: [],
      trg: ['Top']
    },
    params: {
    },
    props: {
      name: {
        name: 'Name',
        value: '',
        type: 'text'
      }
    },
    learn: false
  }
};
