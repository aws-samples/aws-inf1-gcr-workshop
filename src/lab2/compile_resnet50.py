'''
  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 
  Licensed under the Apache License, Version 2.0 (the "License").
  You may not use this file except in compliance with the License.
  A copy of the License is located at
 
      http://www.apache.org/licenses/LICENSE-2.0
 
  or in the "license" file accompanying this file. This file is distributed
  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
  express or implied. See the License for the specific language governing
  permissions and limitations under the License.
'''

import mxnet as mx
import numpy as np

sym, args, aux = mx.model.load_checkpoint('resnet-50', 0)

# Compile for Inferentia using Neuron, fit to NeuronCore group size of 2
inputs = { "data" : mx.nd.ones([1,3,224,224], name='data', dtype='float32') }
compile_args = {} #{'--num-neuroncores' : 2}
sym, args, aux = mx.contrib.neuron.compile(sym, args, aux, inputs, **compile_args)

#save compiled model
mx.model.save_checkpoint("resnet-50_compiled", 0, sym, args, aux)
