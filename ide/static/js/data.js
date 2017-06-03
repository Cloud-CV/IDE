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
  LRN: {
    name: 'lrn',
    color: '#ffeb3b',
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
      local_size: {
        name: 'Local Size',
        value: 5,
        type: 'number',
        required: false
      },
      alpha: {
        name: 'Alpha',
        value: 1.0,
        type: 'float',
        required: false
      },
      beta: {
        name: 'Beta',
        value: 0.75,
        type: 'float',
        required: false
      },
      k: {
        name: 'K',
        value: 1.0,
        type: 'float',
        required: false
      },
      norm_region: {
        name: 'Norm Region',
        value: 'ACROSS_CHANNELS',
        type: 'select',
        options: ['ACROSS_CHANNELS', 'WITHIN_CHANNEL'],
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
  MVN: {
    name: 'mvn',
    color: '#ffeb3b',
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
      normalize_variance: {
        name: 'Normalize Variance',
        value: true,
        type: 'select',
        options: ['true', 'false'],
        required: false
      },
      across_channels: {
        name: 'Across Channels',
        value: false,
        type: 'select',
        options: ['true', 'false'],
        required: false
      },
      eps: {
        name: 'Epsilon',
        value: 1e-9,
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
    learn: true
  },
  BatchNorm: {
    name: 'batchnorm',
    color: '#ffeb3b',
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
      use_global_stats: {
        name: 'Use Global Stats',
        value: '',
        type: 'select',
        options: ['true', 'false'],
        required: true
      },
      moving_average_fraction: {
        name: 'Moving Avg. Fraction',
        value: 0.999,
        type: 'float',
        required: false
      },
      eps: {
        name: 'Epsilon',
        value: 1e-5,
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
    learn: true
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
  Bias: {
    name: 'bias',
    color: '#009688',
    endpoint: {
      src: ['Bottom'],
      trg: ['Top']
    },
    params: {
      axis: {
        name: 'Axis',
        value: 1,
        type: 'number',
        required: false
      },
      num_axes: {
        name: 'Number of Axis',
        value: 1,
        type: 'number',
        required: false
      },
      filler: {
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
  Flatten: {
    name: 'flatten',
    color: '#03a9f4',
    endpoint: {
      src: ['Bottom'],
      trg: ['Top']
    },
    params: {
      axis: {
        name: 'Axis',
        value: 1,
        type: 'number',
        required: false
      },
      end_axis: {
        name: 'End Axis',
        value: -1,
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
  BatchReindex: {
    name: 'Batch Reindex',
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
  Split: {
    name: 'Split',
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
  Slice: {
    name: 'slice',
    color: '#03a9f4',
    endpoint: {
      src: ['Bottom'],
      trg : ['Top']
    },
    params: {
      slice_point: {
        name: 'Slice Point',
        value: '',
        type: 'text',
        required: true
      },
      axis: {
        name: 'Axis',
        value: 1,
        type: 'number',
        required: false
      },
      slice_dim: {
        name: 'Slice Dim',
        value: 1,
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
  Filter: {
    name: 'filter',
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
  Parameter: {
    name: 'parameter',
    color: '#03a9f4',
    endpoint: {
      src: ['Bottom'],
      trg : ['Top']
    },
    params: {
      shape: {
        name: 'Shape',
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
    learn: true
  },
  Reduction: {
    name: 'reduction',
    color: '#03a9f4',
    endpoint: {
      src: ['Bottom'],
      trg: ['Top']
    },
    params: {
      operation: {
        name: 'Reduction operation',
        value: 'SUM',
        type: 'select',
        options: ['SUM', 'ASUM', 'SUMSQ', 'MEAN'],
        required: false
      },
      axis: {
        name: 'Axis',
        value: 0,
        type: 'number',
        required: false
      },
      coeff: {
        name: 'Coefficient',
        value: 1.0,
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
  Silence: {
    name: 'silence',
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
  ArgMax: {
    name: 'argmax',
    color: '#03a9f4',
    endpoint: {
      src: ['Bottom'],
      trg: ['Top']
    },
    params: {
      out_max_val: {
        name: 'Output Max Value',
        value: false,
        type: 'select',
        options: ['true', 'false'],
        required: false
      },
      top_k: {
        name: 'Top-K',
        value: 1,
        type: 'number',
        required: false
      },
      axis: {
        name: 'Axis',
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
  MultinomialLogisticLoss: {
    name: 'multinomial logistic loss',
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
  InfogainLoss: {
    name: 'infogain loss',
    color: '#f44336',
    endpoint: {
      src: [],
      trg: ['Top']
    },
    params: {
      source: {
        name: 'source',
        value: '',
        type: 'text',
        required: true
      },
      axis: {
        name: 'Axis',
        value: 1,
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
  SoftmaxWithLoss: {
    name: 'softmax loss',
    color: '#f44336',
    endpoint: {
      src: [],
      trg: ['Top']
    },
    params: {
      axis: {
        name: 'Axis',
        value: 1,
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
  EuclideanLoss: {
    name: 'euclidean loss',
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
  HingeLoss: {
    name: 'hinge loss',
    color: '#f44336',
    endpoint: {
      src: ['Bottom'],
      trg: ['Top']
    },
    params: {
      norm: {
        name: 'Norm',
        value: 'L1',
        type: 'select',
        options: ['L1', 'L2'],
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
  SigmoidCrossEntropyLoss: {
    name: 'sigmoid cross entropy loss',
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
      top_k: {
        name: 'Top-K',
        value: 1,
        type: 'number',
        required: false
      },
      axis: {
        name: 'Axis',
        value: 1,
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
  ContrastiveLoss: {
    name: 'contrastive loss',
    color: '#f44336',
    endpoint: {
      src: ['Bottom'],
      trg: ['Top']
    },
    params: {
      legacy_version: {
        name: 'Legacy',
        value: false,
        type: 'select',
        options: ['true', 'false'],
        required: false
      },
      margin: {
        name: 'Margin',
        value: 1.0,
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
  }
};
