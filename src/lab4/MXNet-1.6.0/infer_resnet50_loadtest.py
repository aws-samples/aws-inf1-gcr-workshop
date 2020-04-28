import mxnet as mx
import os
import time
from concurrent import futures
import numpy as np

path='http://data.mxnet.io/models/imagenet/'
[mx.test_utils.download(path+'resnet/50-layers/resnet-50-0000.params'),
 mx.test_utils.download(path+'resnet/50-layers/resnet-50-symbol.json'),
 mx.test_utils.download(path+'synset.txt')]

ctx = mx.gpu(0)
ngpu = 1
group2ctx = {'embed': mx.gpu(0),\
             'decode': mx.gpu(ngpu - 1)}

with open('synset.txt', 'r') as f:
    labels = [l.rstrip() for l in f]

sym, args, aux = mx.model.load_checkpoint('resnet-50',0)

#fname = mx.test_utils.download('https://github.com/dmlc/web-data/blob/master/mxnet/doc/tutorials/python/predict_image/cat.jpg?raw=true')
fname = mx.test_utils.download('https://raw.githubusercontent.com/awslabs/mxnet-model-server/master/docs/images/kitten_small.jpg?raw=true')
img = mx.image.imread(fname)
# convert into format (batch, RGB, width, height)
img = mx.image.imresize(img, 224, 224) # resize
img = img.transpose((2, 0, 1)) # Channel first
img = img.expand_dims(axis=0) # batchify
img = img.astype(dtype='float32')
args['data'] = img

softmax = mx.nd.random_normal(shape=(1,))
args['softmax_label'] = softmax

#exe = sym.bind(ctx=ctx, args=args, aux_states=aux, grad_req='null',group2ctx=group2ctx)

#exe.forward()
#prob = exe.outputs[0].asnumpy()
# print the top-5
#prob = np.squeeze(prob)
#a = np.argsort(prob)[::-1]
#for i in a[0:5]:
#    print('probability=%f, class=%s' %(prob[i], labels[i]))

USER_BATCH_SIZE = 50
NUM_LOOPS_PER_THREAD = 100

pred_list = [sym.bind(ctx=ctx, args=args, aux_states=aux, grad_req='null',group2ctx=group2ctx) for _ in range(4)]
pred_list = [
    pred_list[0], pred_list[0], pred_list[0], pred_list[0],
    pred_list[1], pred_list[1], pred_list[1], pred_list[1],
    pred_list[2], pred_list[2], pred_list[2], pred_list[2],
    pred_list[3], pred_list[3], pred_list[3], pred_list[3],
]
num_infer_per_thread = []
for i in range(len(pred_list)):
    num_infer_per_thread.append(0)

def one_thread(pred, index):
    global num_infer_per_thread
    for _ in range(NUM_LOOPS_PER_THREAD):
#        print("_",_)
#        print("NUM_LOOPS_PER_THREAD",NUM_LOOPS_PER_THREAD)
        pred.forward()
        prob = pred.outputs[0].asnumpy()
        # print the top-5
        # print the top-5
#        prob = np.squeeze(prob)
#        a = np.argsort(prob)[::-1]
#        for i in a[0:5]:
#            print('probability=%f, class=%s' %(prob[i], labels[i]))
        num_infer_per_thread[index] += USER_BATCH_SIZE
#       print(num_infer_per_thread[index])

def current_throughput():
    global num_infer_per_thread
    num_infer = 0
    last_num_infer = num_infer
    print("NUM THREADS: ", len(pred_list))
    print("NUM_LOOPS_PER_THREAD: ", NUM_LOOPS_PER_THREAD)
    print("USER_BATCH_SIZE: ", USER_BATCH_SIZE)
    while num_infer < NUM_LOOPS_PER_THREAD * USER_BATCH_SIZE * len(pred_list):
        num_infer = 0
        for i in range(len(pred_list)):
            num_infer = num_infer + num_infer_per_thread[i]
        current_num_infer = num_infer
        throughput = current_num_infer - last_num_infer
        print('current throughput: {} images/sec'.format(throughput))
        last_num_infer = current_num_infer
        time.sleep(1.0)

# Run inference
#model_feed_dict={'input_1:0': img_arr3}

executor = futures.ThreadPoolExecutor(max_workers=16+1)
executor.submit(current_throughput)
for i,pred in enumerate(pred_list):
    executor.submit(one_thread, pred, i)
