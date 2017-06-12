
def get_template():
    return """
name: "SPPlainResNet20b"

layer {
  name: "segdata"
  type: "ROOTData"
  top: "segdata"
  top: "seglabel"
  root_data_param {
    use_thread: false
    batch_size: 5
    report_time: 100
    filler_config: "CONFIG_NAME"
    filler_name: "DataFiller"
  }
}

#
# conv0
#

layer {
	bottom: "segdata"
	top: "conv0"
	name: "conv0"
	type: "Convolution"
	convolution_param {
		num_output: 64
		kernel_size: 7
		pad: 3
		stride: 1
    		weight_filler {
		  type: "msra"
    		}
	}
}

layer {
	bottom: "conv0"
	top: "conv0"
	name: "bn_conv0"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "conv0"
	top: "conv0"
	name: "scale_conv0"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "conv0"
	top: "conv0"
	name: "conv0_relu"
	type: "ReLU"
}

layer {
	bottom: "conv0"
	top: "pool0"
	name: "pool0"
	type: "Pooling"
	pooling_param {
		kernel_size: 3
		stride: 2
		pool: MAX
	}
}

#
# res1a
#

layer {
	bottom: "pool0"
	top: "res1a_branch1"
	name: "res1a_branch1"
	type: "Convolution"
	convolution_param {
		num_output: 64
		kernel_size: 1
		pad: 0
		stride: 1
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res1a_branch1"
	top: "res1a_branch1"
	name: "bn1a_branch1"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res1a_branch1"
	top: "res1a_branch1"
	name: "scale1a_branch1"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "pool0"
	top: "res1a_branch2a"
	name: "res1a_branch2a"
	type: "Convolution"
	convolution_param {
		num_output: 64
		kernel_size: 3
		pad: 1
		stride: 1
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res1a_branch2a"
	top: "res1a_branch2a"
	name: "bn1a_branch2a"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res1a_branch2a"
	top: "res1a_branch2a"
	name: "scale1a_branch2a"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "res1a_branch2a"
	top: "res1a_branch2a"
	name: "res1a_branch2a_relu"
	type: "ReLU"
}

layer {
	bottom: "res1a_branch2a"
	top: "res1a_branch2b"
	name: "res1a_branch2b"
	type: "Convolution"
	convolution_param {
		num_output: 64
		kernel_size: 3
		pad: 1
		stride: 1
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res1a_branch2b"
	top: "res1a_branch2b"
	name: "bn1a_branch2b"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res1a_branch2b"
	top: "res1a_branch2b"
	name: "scale1a_branch2b"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "res1a_branch2b"
	top: "res1a_branch2b"
	name: "res1a_branch2b_relu"
	type: "ReLU"
}

layer {
	bottom: "res1a_branch1"
	bottom: "res1a_branch2b"
	top: "res1a"
	name: "res1a"
	type: "Eltwise"
}

layer {
	bottom: "res1a"
	top: "res1a"
	name: "res1a_relu"
	type: "ReLU"
}

#
# res1b
#

layer {
	bottom: "res1a"
	top: "res1b_branch2a"
	name: "res1b_branch2a"
	type: "Convolution"
	convolution_param {
		num_output: 64
		kernel_size: 3
		pad: 1
		stride: 1
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res1b_branch2a"
	top: "res1b_branch2a"
	name: "bn1b_branch2a"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res1b_branch2a"
	top: "res1b_branch2a"
	name: "scale1b_branch2a"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "res1b_branch2a"
	top: "res1b_branch2a"
	name: "res1b_branch2a_relu"
	type: "ReLU"
}

layer {
	bottom: "res1b_branch2a"
	top: "res1b_branch2b"
	name: "res1b_branch2b"
	type: "Convolution"
	convolution_param {
		num_output: 64
		kernel_size: 3
		pad: 1
		stride: 1
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res1b_branch2b"
	top: "res1b_branch2b"
	name: "bn1b_branch2b"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res1b_branch2b"
	top: "res1b_branch2b"
	name: "scale1b_branch2b"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "res1b_branch2b"
	top: "res1b_branch2b"
	name: "res1b_branch2b_relu"
	type: "ReLU"
}

layer {
	bottom: "res1a"
	bottom: "res1b_branch2b"
	top: "res1b"
	name: "res1b"
	type: "Eltwise"
}

layer {
	bottom: "res1b"
	top: "res1b"
	name: "res1b_relu"
	type: "ReLU"
}

#
# res2a
#

layer {
	bottom: "res1b"
	top: "res2a_branch1"
	name: "res2a_branch1"
	type: "Convolution"
	convolution_param {
		num_output: 128
		kernel_size: 1
		pad: 0
		stride: 2
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}
	}
}

layer {
	bottom: "res2a_branch1"
	top: "res2a_branch1"
	name: "bn2a_branch1"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res2a_branch1"
	top: "res2a_branch1"
	name: "scale2a_branch1"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "res1b"
	top: "res2a_branch2a"
	name: "res2a_branch2a"
	type: "Convolution"
	convolution_param {
		num_output: 128
		kernel_size: 3
		pad: 1
		stride: 2
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res2a_branch2a"
	top: "res2a_branch2a"
	name: "bn2a_branch2a"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res2a_branch2a"
	top: "res2a_branch2a"
	name: "scale2a_branch2a"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "res2a_branch2a"
	top: "res2a_branch2a"
	name: "res2a_branch2a_relu"
	type: "ReLU"
}

layer {
	bottom: "res2a_branch2a"
	top: "res2a_branch2b"
	name: "res2a_branch2b"
	type: "Convolution"
	convolution_param {
		num_output: 128
		kernel_size: 3
		pad: 1
		stride: 1
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res2a_branch2b"
	top: "res2a_branch2b"
	name: "bn2a_branch2b"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res2a_branch2b"
	top: "res2a_branch2b"
	name: "scale2a_branch2b"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "res2a_branch2b"
	top: "res2a_branch2b"
	name: "res2a_branch2b_relu"
	type: "ReLU"
}

layer {
	bottom: "res2a_branch1"
	bottom: "res2a_branch2b"
	top: "res2a"
	name: "res2a"
	type: "Eltwise"
}

layer {
	bottom: "res2a"
	top: "res2a"
	name: "res2a_relu"
	type: "ReLU"
}

#
# res2b
#

layer {
	bottom: "res2a"
	top: "res2b_branch2a"
	name: "res2b_branch2a"
	type: "Convolution"
	convolution_param {
		num_output: 128
		kernel_size: 3
		pad: 1
		stride: 1
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res2b_branch2a"
	top: "res2b_branch2a"
	name: "bn2b_branch2a"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res2b_branch2a"
	top: "res2b_branch2a"
	name: "scale2b_branch2a"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "res2b_branch2a"
	top: "res2b_branch2a"
	name: "res2b_branch2a_relu"
	type: "ReLU"
}

layer {
	bottom: "res2b_branch2a"
	top: "res2b_branch2b"
	name: "res2b_branch2b"
	type: "Convolution"
	convolution_param {
		num_output: 128
		kernel_size: 3
		pad: 1
		stride: 1
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res2b_branch2b"
	top: "res2b_branch2b"
	name: "bn2b_branch2b"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res2b_branch2b"
	top: "res2b_branch2b"
	name: "scale2b_branch2b"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "res2b_branch2b"
	top: "res2b_branch2b"
	name: "res2b_branch2b_relu"
	type: "ReLU"
}

layer {
	bottom: "res2a"
	bottom: "res2b_branch2b"
	top: "res2b"
	name: "res2b"
	type: "Eltwise"
}

layer {
	bottom: "res2b"
	top: "res2b"
	name: "res2b_relu"
	type: "ReLU"
}

#
# res3a
#

layer {
	bottom: "res2b"
	top: "res3a_branch1"
	name: "res3a_branch1"
	type: "Convolution"
	convolution_param {
		num_output: 256
		kernel_size: 1
		pad: 0
		stride: 2
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res3a_branch1"
	top: "res3a_branch1"
	name: "bn3a_branch1"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res3a_branch1"
	top: "res3a_branch1"
	name: "scale3a_branch1"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "res2b"
	top: "res3a_branch2a"
	name: "res3a_branch2a"
	type: "Convolution"
	convolution_param {
		num_output: 256
		kernel_size: 3
		pad: 1
		stride: 2
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res3a_branch2a"
	top: "res3a_branch2a"
	name: "bn3a_branch2a"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res3a_branch2a"
	top: "res3a_branch2a"
	name: "scale3a_branch2a"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "res3a_branch2a"
	top: "res3a_branch2a"
	name: "res3a_branch2a_relu"
	type: "ReLU"
}

layer {
	bottom: "res3a_branch2a"
	top: "res3a_branch2b"
	name: "res3a_branch2b"
	type: "Convolution"
	convolution_param {
		num_output: 256
		kernel_size: 3
		pad: 1
		stride: 1
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res3a_branch2b"
	top: "res3a_branch2b"
	name: "bn3a_branch2b"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res3a_branch2b"
	top: "res3a_branch2b"
	name: "scale3a_branch2b"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "res3a_branch2b"
	top: "res3a_branch2b"
	name: "res3a_branch2b_relu"
	type: "ReLU"
}

layer {
	bottom: "res3a_branch1"
	bottom: "res3a_branch2b"
	top: "res3a"
	name: "res3a"
	type: "Eltwise"
}

layer {
	bottom: "res3a"
	top: "res3a"
	name: "res3a_relu"
	type: "ReLU"
}

#
# res3b
#

layer {
	bottom: "res3a"
	top: "res3b_branch2a"
	name: "res3b_branch2a"
	type: "Convolution"
	convolution_param {
		num_output: 256
		kernel_size: 3
		pad: 1
		stride: 1
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res3b_branch2a"
	top: "res3b_branch2a"
	name: "bn3b_branch2a"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res3b_branch2a"
	top: "res3b_branch2a"
	name: "scale3b_branch2a"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "res3b_branch2a"
	top: "res3b_branch2a"
	name: "res3b_branch2a_relu"
	type: "ReLU"
}

layer {
	bottom: "res3b_branch2a"
	top: "res3b_branch2b"
	name: "res3b_branch2b"
	type: "Convolution"
	convolution_param {
		num_output: 256
		kernel_size: 3
		pad: 1
		stride: 1
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res3b_branch2b"
	top: "res3b_branch2b"
	name: "bn3b_branch2b"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res3b_branch2b"
	top: "res3b_branch2b"
	name: "scale3b_branch2b"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "res3b_branch2b"
	top: "res3b_branch2b"
	name: "res3b_branch2b_relu"
	type: "ReLU"
}

layer {
	bottom: "res3a"
	bottom: "res3b_branch2b"
	top: "res3b"
	name: "res3b"
	type: "Eltwise"
}

layer {
	bottom: "res3b"
	top: "res3b"
	name: "res3b_relu"
	type: "ReLU"
}

#
# res4a
#

layer {
	bottom: "res3b"
	top: "res4a_branch1"
	name: "res4a_branch1"
	type: "Convolution"
	convolution_param {
		num_output: 512
		kernel_size: 1
		pad: 0
		stride: 2
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res4a_branch1"
	top: "res4a_branch1"
	name: "bn4a_branch1"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res4a_branch1"
	top: "res4a_branch1"
	name: "scale4a_branch1"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "res3b"
	top: "res4a_branch2a"
	name: "res4a_branch2a"
	type: "Convolution"
	convolution_param {
		num_output: 512
		kernel_size: 3
		pad: 1
		stride: 2
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res4a_branch2a"
	top: "res4a_branch2a"
	name: "bn4a_branch2a"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res4a_branch2a"
	top: "res4a_branch2a"
	name: "scale4a_branch2a"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "res4a_branch2a"
	top: "res4a_branch2a"
	name: "res4a_branch2a_relu"
	type: "ReLU"
}

layer {
	bottom: "res4a_branch2a"
	top: "res4a_branch2b"
	name: "res4a_branch2b"
	type: "Convolution"
	convolution_param {
		num_output: 512
		kernel_size: 3
		pad: 1
		stride: 1
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res4a_branch2b"
	top: "res4a_branch2b"
	name: "bn4a_branch2b"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res4a_branch2b"
	top: "res4a_branch2b"
	name: "scale4a_branch2b"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "res4a_branch2b"
	top: "res4a_branch2b"
	name: "res4a_branch2b_relu"
	type: "ReLU"
}

layer {
	bottom: "res4a_branch1"
	bottom: "res4a_branch2b"
	top: "res4a"
	name: "res4a"
	type: "Eltwise"
}

layer {
	bottom: "res4a"
	top: "res4a"
	name: "res4a_relu"
	type: "ReLU"
}

#
# res4b
#

layer {
	bottom: "res4a"
	top: "res4b_branch2a"
	name: "res4b_branch2a"
	type: "Convolution"
	convolution_param {
		num_output: 512
		kernel_size: 3
		pad: 1
		stride: 1
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res4b_branch2a"
	top: "res4b_branch2a"
	name: "bn4b_branch2a"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res4b_branch2a"
	top: "res4b_branch2a"
	name: "scale4b_branch2a"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "res4b_branch2a"
	top: "res4b_branch2a"
	name: "res4b_branch2a_relu"
	type: "ReLU"
}

layer {
	bottom: "res4b_branch2a"
	top: "res4b_branch2b"
	name: "res4b_branch2b"
	type: "Convolution"
	convolution_param {
		num_output: 512
		kernel_size: 3
		pad: 1
		stride: 1
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res4b_branch2b"
	top: "res4b_branch2b"
	name: "bn4b_branch2b"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res4b_branch2b"
	top: "res4b_branch2b"
	name: "scale4b_branch2b"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "res4b_branch2b"
	top: "res4b_branch2b"
	name: "res4b_branch2b_relu"
	type: "ReLU"
}

layer {
	bottom: "res4a"
	bottom: "res4b_branch2b"
	top: "res4b"
	name: "res4b"
	type: "Eltwise"
}

layer {
	bottom: "res4b"
	top: "res4b"
	name: "res4b_relu"
	type: "ReLU"
}

#
# res5a
#

layer {
	bottom: "res4b"
	top: "res5a_branch1"
	name: "res5a_branch1"
	type: "Convolution"
	convolution_param {
		num_output: 1024
		kernel_size: 1
		pad: 0
		stride: 2
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res5a_branch1"
	top: "res5a_branch1"
	name: "bn5a_branch1"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res5a_branch1"
	top: "res5a_branch1"
	name: "scale5a_branch1"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "res4b"
	top: "res5a_branch2a"
	name: "res5a_branch2a"
	type: "Convolution"
	convolution_param {
		num_output: 1024
		kernel_size: 3
		pad: 1
		stride: 2
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res5a_branch2a"
	top: "res5a_branch2a"
	name: "bn5a_branch2a"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res5a_branch2a"
	top: "res5a_branch2a"
	name: "scale5a_branch2a"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "res5a_branch2a"
	top: "res5a_branch2a"
	name: "res5a_branch2a_relu"
	type: "ReLU"
}

layer {
	bottom: "res5a_branch2a"
	top: "res5a_branch2b"
	name: "res5a_branch2b"
	type: "Convolution"
	convolution_param {
		num_output: 1024
		kernel_size: 3
		pad: 1
		stride: 1
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res5a_branch2b"
	top: "res5a_branch2b"
	name: "bn5a_branch2b"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res5a_branch2b"
	top: "res5a_branch2b"
	name: "scale5a_branch2b"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "res5a_branch2b"
	top: "res5a_branch2b"
	name: "res5a_branch2b_relu"
	type: "ReLU"
}

layer {
	bottom: "res5a_branch1"
	bottom: "res5a_branch2b"
	top: "res5a"
	name: "res5a"
	type: "Eltwise"
}

layer {
	bottom: "res5a"
	top: "res5a"
	name: "res5a_relu"
	type: "ReLU"
}

#
# res5b
#

layer {
	bottom: "res5a"
	top: "res5b_branch2a"
	name: "res5b_branch2a"
	type: "Convolution"
	convolution_param {
		num_output: 1024
		kernel_size: 3
		pad: 1
		stride: 1
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res5b_branch2a"
	top: "res5b_branch2a"
	name: "bn5b_branch2a"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res5b_branch2a"
	top: "res5b_branch2a"
	name: "scale5b_branch2a"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "res5b_branch2a"
	top: "res5b_branch2a"
	name: "res5b_branch2a_relu"
	type: "ReLU"
}

layer {
	bottom: "res5b_branch2a"
	top: "res5b_branch2b"
	name: "res5b_branch2b"
	type: "Convolution"
	convolution_param {
		num_output: 1024
		kernel_size: 3
		pad: 1
		stride: 1
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res5b_branch2b"
	top: "res5b_branch2b"
	name: "bn5b_branch2b"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res5b_branch2b"
	top: "res5b_branch2b"
	name: "scale5b_branch2b"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "res5b_branch2b"
	top: "res5b_branch2b"
	name: "res5b_branch2b_relu"
	type: "ReLU"
}

layer {
	bottom: "res5a"
	bottom: "res5b_branch2b"
	top: "res5b"
	name: "res5b"
	type: "Eltwise"
}

layer {
	bottom: "res5b"
	top: "res5b"
	name: "res5b_relu"
	type: "ReLU"
}

#
# deconv0
#

layer {
  name: "deconv0_deconv"
  type: "Deconvolution"
  bottom: "res5b"
  top: "deconv0_deconv"
  param {
    name: "par_deconv0_deconv_w"
    lr_mult: 1.0
  }
  param {
    name: "par_deconv0_deconv_b"
    lr_mult: 0.0
  }
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 4
    group: 512
    stride: 2
    weight_filler {
      type: "bilinear"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "deconv0_concat"
  type: "Concat"
  bottom: "res4b"
  bottom: "deconv0_deconv"
  top: "deconv0_concat"
  concat_param {
    axis: 1
  }
}

#
# res6a
#

layer {
	bottom: "deconv0_concat"
	top: "res6a_branch1"
	name: "res6a_branch1"
	type: "Convolution"
	convolution_param {
		num_output: 512
		kernel_size: 1
		pad: 0
		stride: 1
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res6a_branch1"
	top: "res6a_branch1"
	name: "bn6a_branch1"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res6a_branch1"
	top: "res6a_branch1"
	name: "scale6a_branch1"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "deconv0_concat"
	top: "res6a_branch2a"
	name: "res6a_branch2a"
	type: "Convolution"
	convolution_param {
		num_output: 512
		kernel_size: 3
		pad: 1
		stride: 1
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res6a_branch2a"
	top: "res6a_branch2a"
	name: "bn6a_branch2a"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res6a_branch2a"
	top: "res6a_branch2a"
	name: "scale6a_branch2a"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "res6a_branch2a"
	top: "res6a_branch2a"
	name: "res6a_branch2a_relu"
	type: "ReLU"
}

layer {
	bottom: "res6a_branch2a"
	top: "res6a_branch2b"
	name: "res6a_branch2b"
	type: "Convolution"
	convolution_param {
		num_output: 512
		kernel_size: 3
		pad: 1
		stride: 1
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res6a_branch2b"
	top: "res6a_branch2b"
	name: "bn6a_branch2b"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res6a_branch2b"
	top: "res6a_branch2b"
	name: "scale6a_branch2b"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "res6a_branch2b"
	top: "res6a_branch2b"
	name: "res6a_branch2b_relu"
	type: "ReLU"
}

layer {
	bottom: "res6a_branch1"
	bottom: "res6a_branch2b"
	top: "res6a"
	name: "res6a"
	type: "Eltwise"
}

layer {
	bottom: "res6a"
	top: "res6a"
	name: "res6a_relu"
	type: "ReLU"
}

#
# res6b
#

layer {
	bottom: "res6a"
	top: "res6b_branch2a"
	name: "res6b_branch2a"
	type: "Convolution"
	convolution_param {
		num_output: 512
		kernel_size: 3
		pad: 1
		stride: 1
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res6b_branch2a"
	top: "res6b_branch2a"
	name: "bn6b_branch2a"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res6b_branch2a"
	top: "res6b_branch2a"
	name: "scale6b_branch2a"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "res6b_branch2a"
	top: "res6b_branch2a"
	name: "res6b_branch2a_relu"
	type: "ReLU"
}

layer {
	bottom: "res6b_branch2a"
	top: "res6b_branch2b"
	name: "res6b_branch2b"
	type: "Convolution"
	convolution_param {
		num_output: 512
		kernel_size: 3
		pad: 1
		stride: 1
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res6b_branch2b"
	top: "res6b_branch2b"
	name: "bn6b_branch2b"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res6b_branch2b"
	top: "res6b_branch2b"
	name: "scale6b_branch2b"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "res6b_branch2b"
	top: "res6b_branch2b"
	name: "res6b_branch2b_relu"
	type: "ReLU"
}

layer {
	bottom: "res6a"
	bottom: "res6b_branch2b"
	top: "res6b"
	name: "res6b"
	type: "Eltwise"
}

layer {
	bottom: "res6b"
	top: "res6b"
	name: "res6b_relu"
	type: "ReLU"
}

#
# deconv1
#

layer {
  name: "deconv1_deconv"
  type: "Deconvolution"
  bottom: "res6b"
  top: "deconv1_deconv"
  param {
    name: "par_deconv1_deconv_w"
    lr_mult: 1.0
  }
  param {
    name: "par_deconv1_deconv_b"
    lr_mult: 0.0
  }
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 4
    group: 256
    stride: 2
    weight_filler {
      type: "bilinear"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "deconv1_concat"
  type: "Concat"
  bottom: "res3b"
  bottom: "deconv1_deconv"
  top: "deconv1_concat"
  concat_param {
    axis: 1
  }
}

#
# res7a
#

layer {
	bottom: "deconv1_concat"
	top: "res7a_branch1"
	name: "res7a_branch1"
	type: "Convolution"
	convolution_param {
		num_output: 256
		kernel_size: 1
		pad: 0
		stride: 1
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res7a_branch1"
	top: "res7a_branch1"
	name: "bn7a_branch1"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res7a_branch1"
	top: "res7a_branch1"
	name: "scale7a_branch1"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "deconv1_concat"
	top: "res7a_branch2a"
	name: "res7a_branch2a"
	type: "Convolution"
	convolution_param {
		num_output: 256
		kernel_size: 3
		pad: 1
		stride: 1
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res7a_branch2a"
	top: "res7a_branch2a"
	name: "bn7a_branch2a"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res7a_branch2a"
	top: "res7a_branch2a"
	name: "scale7a_branch2a"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "res7a_branch2a"
	top: "res7a_branch2a"
	name: "res7a_branch2a_relu"
	type: "ReLU"
}

layer {
	bottom: "res7a_branch2a"
	top: "res7a_branch2b"
	name: "res7a_branch2b"
	type: "Convolution"
	convolution_param {
		num_output: 256
		kernel_size: 3
		pad: 1
		stride: 1
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res7a_branch2b"
	top: "res7a_branch2b"
	name: "bn7a_branch2b"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res7a_branch2b"
	top: "res7a_branch2b"
	name: "scale7a_branch2b"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "res7a_branch2b"
	top: "res7a_branch2b"
	name: "res7a_branch2b_relu"
	type: "ReLU"
}

layer {
	bottom: "res7a_branch1"
	bottom: "res7a_branch2b"
	top: "res7a"
	name: "res7a"
	type: "Eltwise"
}

layer {
	bottom: "res7a"
	top: "res7a"
	name: "res7a_relu"
	type: "ReLU"
}

#
# res7b
#

layer {
	bottom: "res7a"
	top: "res7b_branch2a"
	name: "res7b_branch2a"
	type: "Convolution"
	convolution_param {
		num_output: 256
		kernel_size: 3
		pad: 1
		stride: 1
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res7b_branch2a"
	top: "res7b_branch2a"
	name: "bn7b_branch2a"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res7b_branch2a"
	top: "res7b_branch2a"
	name: "scale7b_branch2a"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "res7b_branch2a"
	top: "res7b_branch2a"
	name: "res7b_branch2a_relu"
	type: "ReLU"
}

layer {
	bottom: "res7b_branch2a"
	top: "res7b_branch2b"
	name: "res7b_branch2b"
	type: "Convolution"
	convolution_param {
		num_output: 256
		kernel_size: 3
		pad: 1
		stride: 1
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res7b_branch2b"
	top: "res7b_branch2b"
	name: "bn7b_branch2b"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res7b_branch2b"
	top: "res7b_branch2b"
	name: "scale7b_branch2b"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "res7b_branch2b"
	top: "res7b_branch2b"
	name: "res7b_branch2b_relu"
	type: "ReLU"
}

layer {
	bottom: "res7a"
	bottom: "res7b_branch2b"
	top: "res7b"
	name: "res7b"
	type: "Eltwise"
}

layer {
	bottom: "res7b"
	top: "res7b"
	name: "res7b_relu"
	type: "ReLU"
}

#
# deconv2
#

layer {
  name: "deconv2_deconv"
  type: "Deconvolution"
  bottom: "res7b"
  top: "deconv2_deconv"
  param {
    name: "par_deconv2_deconv_w"
    lr_mult: 1.0
  }
  param {
    name: "par_deconv2_deconv_b"
    lr_mult: 0.0
  }
  convolution_param {
    num_output: 128
    pad: 1
    kernel_size: 4
    group: 128
    stride: 2
    weight_filler {
      type: "bilinear"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "deconv2_concat"
  type: "Concat"
  bottom: "res2b"
  bottom: "deconv2_deconv"
  top: "deconv2_concat"
  concat_param {
    axis: 1
  }
}

#
# res8a
#

layer {
	bottom: "deconv2_concat"
	top: "res8a_branch1"
	name: "res8a_branch1"
	type: "Convolution"
	convolution_param {
		num_output: 128
		kernel_size: 1
		pad: 0
		stride: 1
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res8a_branch1"
	top: "res8a_branch1"
	name: "bn8a_branch1"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res8a_branch1"
	top: "res8a_branch1"
	name: "scale8a_branch1"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "deconv2_concat"
	top: "res8a_branch2a"
	name: "res8a_branch2a"
	type: "Convolution"
	convolution_param {
		num_output: 128
		kernel_size: 3
		pad: 1
		stride: 1
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res8a_branch2a"
	top: "res8a_branch2a"
	name: "bn8a_branch2a"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res8a_branch2a"
	top: "res8a_branch2a"
	name: "scale8a_branch2a"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "res8a_branch2a"
	top: "res8a_branch2a"
	name: "res8a_branch2a_relu"
	type: "ReLU"
}

layer {
	bottom: "res8a_branch2a"
	top: "res8a_branch2b"
	name: "res8a_branch2b"
	type: "Convolution"
	convolution_param {
		num_output: 128
		kernel_size: 3
		pad: 1
		stride: 1
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res8a_branch2b"
	top: "res8a_branch2b"
	name: "bn8a_branch2b"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res8a_branch2b"
	top: "res8a_branch2b"
	name: "scale8a_branch2b"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "res8a_branch2b"
	top: "res8a_branch2b"
	name: "res8a_branch2b_relu"
	type: "ReLU"
}

layer {
	bottom: "res8a_branch1"
	bottom: "res8a_branch2b"
	top: "res8a"
	name: "res8a"
	type: "Eltwise"
}

layer {
	bottom: "res8a"
	top: "res8a"
	name: "res8a_relu"
	type: "ReLU"
}

#
# res8b
#

layer {
	bottom: "res8a"
	top: "res8b_branch2a"
	name: "res8b_branch2a"
	type: "Convolution"
	convolution_param {
		num_output: 128
		kernel_size: 3
		pad: 1
		stride: 1
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res8b_branch2a"
	top: "res8b_branch2a"
	name: "bn8b_branch2a"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res8b_branch2a"
	top: "res8b_branch2a"
	name: "scale8b_branch2a"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "res8b_branch2a"
	top: "res8b_branch2a"
	name: "res8b_branch2a_relu"
	type: "ReLU"
}

layer {
	bottom: "res8b_branch2a"
	top: "res8b_branch2b"
	name: "res8b_branch2b"
	type: "Convolution"
	convolution_param {
		num_output: 128
		kernel_size: 3
		pad: 1
		stride: 1
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res8b_branch2b"
	top: "res8b_branch2b"
	name: "bn8b_branch2b"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res8b_branch2b"
	top: "res8b_branch2b"
	name: "scale8b_branch2b"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "res8b_branch2b"
	top: "res8b_branch2b"
	name: "res8b_branch2b_relu"
	type: "ReLU"
}

layer {
	bottom: "res8a"
	bottom: "res8b_branch2b"
	top: "res8b"
	name: "res8b"
	type: "Eltwise"
}

layer {
	bottom: "res8b"
	top: "res8b"
	name: "res8b_relu"
	type: "ReLU"
}

#
# deconv3
#

layer {
  name: "deconv3_deconv"
  type: "Deconvolution"
  bottom: "res8b"
  top: "deconv3_deconv"
  param {
    name: "par_deconv3_deconv_w"
    lr_mult: 1.0
  }
  param {
    name: "par_deconv3_deconv_b"
    lr_mult: 0.0
  }
  convolution_param {
    num_output: 64
    pad: 1
    kernel_size: 4
    group: 64
    stride: 2
    weight_filler {
      type: "bilinear"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "deconv3_concat"
  type: "Concat"
  bottom: "res1b"
  bottom: "deconv3_deconv"
  top: "deconv3_concat"
  concat_param {
    axis: 1
  }
}

#
# res9a
#

layer {
	bottom: "deconv3_concat"
	top: "res9a_branch1"
	name: "res9a_branch1"
	type: "Convolution"
	convolution_param {
		num_output: 64
		kernel_size: 1
		pad: 0
		stride: 1
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res9a_branch1"
	top: "res9a_branch1"
	name: "bn9a_branch1"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res9a_branch1"
	top: "res9a_branch1"
	name: "scale9a_branch1"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "deconv3_concat"
	top: "res9a_branch2a"
	name: "res9a_branch2a"
	type: "Convolution"
	convolution_param {
		num_output: 64
		kernel_size: 5
		pad: 2
		stride: 1
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res9a_branch2a"
	top: "res9a_branch2a"
	name: "bn9a_branch2a"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res9a_branch2a"
	top: "res9a_branch2a"
	name: "scale9a_branch2a"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "res9a_branch2a"
	top: "res9a_branch2a"
	name: "res9a_branch2a_relu"
	type: "ReLU"
}

layer {
	bottom: "res9a_branch2a"
	top: "res9a_branch2b"
	name: "res9a_branch2b"
	type: "Convolution"
	convolution_param {
		num_output: 64
		kernel_size: 5
		pad: 2
		stride: 1
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res9a_branch2b"
	top: "res9a_branch2b"
	name: "bn9a_branch2b"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res9a_branch2b"
	top: "res9a_branch2b"
	name: "scale9a_branch2b"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "res9a_branch2b"
	top: "res9a_branch2b"
	name: "res9a_branch2b_relu"
	type: "ReLU"
}

layer {
	bottom: "res9a_branch1"
	bottom: "res9a_branch2b"
	top: "res9a"
	name: "res9a"
	type: "Eltwise"
}

layer {
	bottom: "res9a"
	top: "res9a"
	name: "res9a_relu"
	type: "ReLU"
}

#
# res9b
#

layer {
	bottom: "res9a"
	top: "res9b_branch2a"
	name: "res9b_branch2a"
	type: "Convolution"
	convolution_param {
		num_output: 64
		kernel_size: 5
		pad: 2
		stride: 1
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res9b_branch2a"
	top: "res9b_branch2a"
	name: "bn9b_branch2a"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res9b_branch2a"
	top: "res9b_branch2a"
	name: "scale9b_branch2a"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "res9b_branch2a"
	top: "res9b_branch2a"
	name: "res9b_branch2a_relu"
	type: "ReLU"
}

layer {
	bottom: "res9b_branch2a"
	top: "res9b_branch2b"
	name: "res9b_branch2b"
	type: "Convolution"
	convolution_param {
		num_output: 64
		kernel_size: 5
		pad: 2
		stride: 1
		bias_term: false
    		weight_filler {
		  type: "msra"
    		}

	}
}

layer {
	bottom: "res9b_branch2b"
	top: "res9b_branch2b"
	name: "bn9b_branch2b"
	type: "BatchNorm"
	batch_norm_param {
		use_global_stats: false
	}
}

layer {
	bottom: "res9b_branch2b"
	top: "res9b_branch2b"
	name: "scale9b_branch2b"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}

layer {
	bottom: "res9b_branch2b"
	top: "res9b_branch2b"
	name: "res9b_branch2b_relu"
	type: "ReLU"
}

layer {
	bottom: "res9a"
	bottom: "res9b_branch2b"
	top: "res9b"
	name: "res9b"
	type: "Eltwise"
}

layer {
	bottom: "res9b"
	top: "res9b"
	name: "res9b_relu"
	type: "ReLU"
}

#
# deconv4
#

layer {
  name: "deconv4_deconv"
  type: "Deconvolution"
  bottom: "res9b"
  top: "deconv4_deconv"
  param {
    name: "par_deconv4_deconv_w"
    lr_mult: 1.0
  }
  param {
    name: "par_deconv4_deconv_b"
    lr_mult: 0.0
  }
  convolution_param {
    num_output: 64
    pad: 1
    kernel_size: 4
    group: 64
    stride: 2
    weight_filler {
      type: "bilinear"
    }
    bias_filler {
      type: "constant"
      value: 0.0
    }
  }
}
layer {
  name: "deconv4_concat"
  type: "Concat"
  bottom: "conv0"
  bottom: "deconv4_deconv"
  top: "deconv4_concat"
  concat_param {
    axis: 1
  }
}

#
# conv10
#

layer {
        bottom: "deconv4_concat"
        top: "conv10"
        name: "conv10"
        type: "Convolution"
        convolution_param {
                num_output: 3
                kernel_size: 7
                pad: 3
                stride: 1
                weight_filler {
                  type: "msra"
                }
        }
}

layer {
        bottom: "conv10"
        top: "conv10"
        name: "bn_conv10"
        type: "BatchNorm"
	batch_norm_param {
			 use_global_stats: false
			 }
}

layer {
        bottom: "conv10"
        top: "conv10"
        name: "scale_conv10"
        type: "Scale"
        scale_param {
                bias_term: true
        }
}

layer {
        bottom: "conv10"
        top: "conv10"
        name: "conv10_relu"
        type: "ReLU"
}

layer {
  name: "crop_conv10"
  type: "Crop"
  bottom: "conv10"
  bottom: "seglabel"
  top: "crop_conv10"
  crop_param {
    axis: 2
    offset: 0
  }
}
layer {
  name: "softmax"
  type: "Softmax"
  bottom: "crop_conv10"
  top: "softmax"
}
layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "crop_conv10"
  bottom: "seglabel"
  top: "accuracy"
  accuracy_param {
    top_k: 1
    ignore_label: 0
  }
}
"""
