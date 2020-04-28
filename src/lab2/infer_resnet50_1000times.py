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
import time


img = mx.image.imread('test.jpg')# convert into format (batch, RGB, width, height)
img = mx.image.imresize(img, 224, 224) # resize
img = img.transpose((2, 0, 1)) # Channel first
img = img.expand_dims(axis=0) # batchify
img = img.astype(dtype='float32')

sym, args, aux = mx.model.load_checkpoint('resnet-50_compiled', 0)
softmax = mx.nd.random_normal(shape=(1,))
args['softmax_label'] = softmax
args['data'] = img

# Inferentia context
ctx = mx.neuron()

exe = sym.bind(ctx=ctx, args=args, aux_states=aux, grad_req='null')

with open('synset.txt', 'r') as f:
     labels = [l.rstrip() for l in f]

for i in range(100):
  exe.forward(data=img)
  prob = exe.outputs[0].asnumpy()
  prob = np.squeeze(prob)

start = time.clock()
for i in range(1000):
  exe.forward(data=img)
  out = exe.outputs[0]
end = time.clock()
prob = out.asnumpy()
prob = np.squeeze(prob)
#end = time.clock()
print("average : ", (end - start)*1000/1000, " ms.")


a = np.argsort(prob)[::-1]
for i in a[0:5]:
     print('probability=%f, class=%s' %(prob[i], labels[i]))
