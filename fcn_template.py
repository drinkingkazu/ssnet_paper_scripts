
def get_template():
    return """
name: "FCN8AlexNet"

layer {
  name: "data"
    type: "ROOTData"
    top: "data"
    top: "label"

    root_data_param {
      use_thread: false
      batch_size: 150
      filler_config: "/data/vgenty/UBFCN/dlmc_mcc8_multipvtx_v01/fcn8_yesspweights_drop_pretrain/ana_filler.cfg"
      filler_name: "DataFiller"
    }
}

layer {
  name: "conv1"
    type: "Convolution"
    bottom: "data"
    top: "conv1"
    convolution_param {
      num_output: 96
      pad: 100
      kernel_size: 11
      group: 1
      stride: 4
    weight_filler {
      type: "msra"
    }
  }
}

layer {
  name: "relu1"
    type: "ReLU"
    bottom: "conv1"
    top: "conv1"
    }

layer {
  name: "pool1"
    type: "Pooling"
    bottom: "conv1"
    top: "pool1"
    pooling_param {
      pool: MAX
      kernel_size: 3
      stride: 2
    }
}

layer {
  name: "norm1"
    type: "LRN"
    bottom: "pool1"
    top: "norm1"
    lrn_param {
    local_size: 5
      alpha: 0.0001
      beta: 0.75
      }
}
layer {
  name: "conv2"
    type: "Convolution"
    bottom: "norm1"
    top: "conv2"
    convolution_param {
    num_output: 256
      pad: 2
      kernel_size: 5
      group: 2
      stride: 1
    weight_filler {
       type: "msra"
      }
    }
}
layer {
  name: "relu2"
    type: "ReLU"
    bottom: "conv2"
    top: "conv2"
    }
layer {
  name: "pool2"
    type: "Pooling"
    bottom: "conv2"
    top: "pool2"
    pooling_param {
    pool: MAX
      kernel_size: 3
      stride: 2
      }
}
layer {
  name: "norm2"
    type: "LRN"
    bottom: "pool2"
    top: "norm2"
    lrn_param {
    local_size: 5
      alpha: 0.0001
      beta: 0.75
      }
}
layer {
  name: "conv3"
    type: "Convolution"
    bottom: "norm2"
    top: "conv3"
    convolution_param {
    num_output: 384
      pad: 1
      kernel_size: 3
      group: 1
      stride: 1
      weight_filler {
       type: "msra"
      }
  }
}
layer {
  name: "relu3"
    type: "ReLU"
    bottom: "conv3"
    top: "conv3"
    }
layer {
  name: "conv4"
    type: "Convolution"
    bottom: "conv3"
    top: "conv4"
    convolution_param {
    num_output: 384
      pad: 1
      kernel_size: 3
      group: 2
      stride: 1
    weight_filler {
       type: "msra"
      }
      }
}
layer {
  name: "relu4"
    type: "ReLU"
    bottom: "conv4"
    top: "conv4"
    }
layer {
  name: "conv5"
    type: "Convolution"
    bottom: "conv4"
    top: "conv5"
    convolution_param {
    num_output: 256
      pad: 1
      kernel_size: 3
      group: 2
      stride: 1
    weight_filler {
       type: "msra"
      }
      }
}
layer {
  name: "relu5"
    type: "ReLU"
    bottom: "conv5"
    top: "conv5"
    }
layer {
  name: "pool5"
    type: "Pooling"
    bottom: "conv5"
    top: "pool5"
    pooling_param {
    pool: MAX
      kernel_size: 3
      stride: 2
      }
}
layer {
  name: "fc6_"
    type: "Convolution"
    bottom: "pool5"
    top: "fc6_"
    convolution_param {

    num_output: 4096
      pad: 0
      kernel_size: 6
      group: 1
      stride: 1
    weight_filler {
       type: "msra"
      }
  }
}
layer {
  name: "relu6_"
    type: "ReLU"
    bottom: "fc6_"
    top: "fc6_"
    }
layer {
  name: "drop6_"
    type: "Dropout"
    bottom: "fc6_"
  top: "fc6_"
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: "fc7_"
  type: "Convolution"
  bottom: "fc6_"
  top: "fc7_"
  convolution_param {
    num_output: 4096
    pad: 0
    kernel_size: 1
    group: 1
    stride: 1
    weight_filler {
       type: "msra"
      }
  }
}
layer {
  name: "relu7_"
  type: "ReLU"
  bottom: "fc7_"
  top: "fc7_"
}
layer {
  name: "drop7_"
  type: "Dropout"
  bottom: "fc7_"
  top: "fc7_"
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: "score_fr"
  type: "Convolution"
  bottom: "fc7_"
  top: "score_fr"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 3 
    pad: 0
    kernel_size: 1
    weight_filler {
       type: "msra"
      }
  }
}
layer {
  name: "upscore2"
  type: "Deconvolution"
  bottom: "score_fr"
  top: "upscore2"
 param {
   lr_mult: 1
   decay_mult: 1
 }

# param {
#   lr_mult: 2
#   decay_mult: 0
# }


  convolution_param {
    num_output: 3        
    bias_term: false
    #kernel_size: 63
    #stride: 32
    kernel_size: 4
    stride: 2
     
   weight_filler {
      type: "msra"
   }
  }
}


layer {
  name: "score_pool2"
  type: "Convolution"
  bottom: "norm2"
  top: "score_pool2"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 3
    pad: 0
    kernel_size: 1
    weight_filler {
      type: "msra"
   }
  }
}


layer {
  name: "score_pool2c"
  type: "Crop"
  bottom: "score_pool2"
  bottom: "upscore2"
  top: "score_pool2c"
  crop_param {
    axis: 2
    offset: 4#5
  }
}


layer {
  name: "fuse_pool2"
  type: "Eltwise"
  bottom: "upscore2"
  bottom: "score_pool2c"
  top: "fuse_pool2"
  eltwise_param {
    operation: SUM
  }
}


layer {
  name: "upscore_pool2"
  type: "Deconvolution"
  bottom: "fuse_pool2"
  top: "upscore_pool2"
  param {
    lr_mult: 1
  }
  convolution_param {
    num_output: 3
    bias_term: false
    kernel_size: 4
    stride: 2
    weight_filler {
      type: "msra"
    }
  }
}

layer {
  name: "score_pool1"
  type: "Convolution"
  bottom: "norm1"
  top: "score_pool1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 3
    pad: 0
    kernel_size: 1
       weight_filler {
      type: "msra"
   }
  }
}
layer {
  name: "score_pool1c"
  type: "Crop"
  bottom: "score_pool1"
  bottom: "upscore_pool2"
  top: "score_pool1c"
  crop_param {
    axis: 2
    offset: 7
  }
}
layer {
  name: "fuse_pool1"
  type: "Eltwise"
  bottom: "upscore_pool2"
  bottom: "score_pool1c"
  top: "fuse_pool1"
  eltwise_param {
    operation: SUM
  }
}
layer {
  name: "upscore8"
  type: "Deconvolution"
  bottom: "fuse_pool1"
  top: "upscore8"
  param {
    lr_mult: 1
  }
  convolution_param {
    num_output: 3
    bias_term: false
    kernel_size: 16
    stride: 8
       weight_filler {
      type: "msra"
   }
  }
}
layer {
  name: "score"
  type: "Crop"
  bottom: "upscore8"
  bottom: "data"
  top: "score"
  crop_param {
    axis: 2
    offset: 44#31 # --> will I be one row/column off? fuck
  }
}

layer {
  name: "softmax"
  type: "Softmax"
  bottom: "score"
  top: "softmax"
}

layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "score"
  bottom: "label"
  top: "accuracy"
   accuracy_param {
     top_k: 1
     ignore_label: 0
   }
}
"""
