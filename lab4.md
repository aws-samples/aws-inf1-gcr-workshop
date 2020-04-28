# 实验4 在G4实例上测试预训练模型Resnet-50(lab4)
## 4.1 设置G4实例
在AWS US East (N. Virginia)us-east-1 区域，从AMI Deep Learning AMI (Ubuntu 16.04) Version 27.0 (ami-0a79b70001264b442)，分别启动1台g4dn.xlarge， g4dn.2xlarge，g4dn.4xlarge，g4dn.8xlarge，g4dn.12xlarge，g4dn.16xlarge，指定合适的key pair。

## 4.2 性能测试(TensorFlow1,MXNet,PyTorch,TensorFlow2)
### 结果记录表格


| Family | Type | Tensorflow-1.15.2 | PyTorch-1.4.0 | MXNet-1.6.0 | Tensorflow-2.1 |
| -------- | -------- | -------- | -------- | -------- | -------- |
| inf1     | xlarge     |      |      |      |      |
| inf1     | 2xlarge     |      |      |      |      |
| inf1     | 6xlarge     |      |      |      |      |
| inf1     | 24xlarge     |      |      |      |      |
| -------- | -------- | -------- | -------- | -------- | -------- |
| g4dn     | xlarge    |      |      |      |      |
| g4dn     | 2xlarge     |      |      |      |      |
| g4dn     | 4xlarge     |      |      |      |      |
| g4dn     | 8xlarge     |      |      |      |      |
| g4dn     | 16xlarge     |      |      |      |      |
| g4dn     | 12xlarge     |      |      |      |      |



### 4.2.1 TensorFlow1 + keras
ssh 到 相应g4dn实例

下载代码

cd ./src/lab4/Tensorflow-1.15.2

`source deactivate`

`source activate tensorflow_p36`

打开savemodel_resnet50.py 文件检查

`more savemodel_resnet50.py`

```
l's
```

执行savemodel_resnet50.py 文件
`time python savemodel_resnet50.py `

下载推理用的图片
`curl -O https://raw.githubusercontent.com/awslabs/mxnet-model-server/master/docs/images/kitten_small.jpg`

打开 infer_resnet50.py 文件检查
`more infer_resnet50.py `

```
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import resnet50

tf.keras.backend.set_image_data_format('channels_last')

#Create input from image
img_sgl = image.load_img('kitten_small.jpg', target_size=(224, 224))
img_arr = image.img_to_array(img_sgl)
img_arr2 = np.expand_dims(img_arr, axis=0)
img_arr3 = resnet50.preprocess_input(img_arr2)

#Load model
MODEL_DIR = './ws_resnet50/resnet50/'
predictor_inferentia = tf.contrib.predictor.from_saved_model(MODEL_DIR)

#Run Inference and Display results
model_feed_dict={'input': img_arr3}
infa_rslts = predictor_inferentia(model_feed_dict)
print(resnet50.decode_predictions(infa_rslts["output"], top=5)[0])
```

执行 infer_resnet50.py 文件 确认推理无误
`python infer_resnet50.py`

打开 infer_resnet50_loadtest.py 文件检查
`more infer_resnet50_loadtest.py`


```
import os
import time
from concurrent import futures
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import resnet50

tf.keras.backend.set_image_data_format('channels_last')

USER_BATCH_SIZE = 50
NUM_LOOPS_PER_THREAD = 100

# Create input from image
img_sgl = image.load_img('kitten_small.jpg', target_size=(224, 224))
img_arr = image.img_to_array(img_sgl)
img_arr2 = np.expand_dims(img_arr, axis=0)
img_arr3 = resnet50.preprocess_input(np.repeat(img_arr2, USER_BATCH_SIZE, axis=0))

# Load model
COMPILED_MODEL_DIR = './ws_resnet50/resnet50'
pred_list = [tf.contrib.predictor.from_saved_model(COMPILED_MODEL_DIR) for _ in range(4)]
pred_list = [
    pred_list[0], pred_list[0], pred_list[0], pred_list[0],
    pred_list[1], pred_list[1], pred_list[1], pred_list[1],
    pred_list[2], pred_list[2], pred_list[2], pred_list[2],
    pred_list[3], pred_list[3], pred_list[3], pred_list[3],
]

num_infer_per_thread = []
for i in range(len(pred_list)):
    num_infer_per_thread.append(0)
def one_thread(pred, model_feed_dict, index):
    global num_infer_per_thread
    for i in range(NUM_LOOPS_PER_THREAD):
        result = pred(model_feed_dict)
        num_infer_per_thread[index] += USER_BATCH_SIZE

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
            #print("num_infer_:",num_infer)
        #print("num_infer:",num_infer)
        current_num_infer = num_infer
        throughput = current_num_infer - last_num_infer
        print('current throughput: {} images/sec'.format(throughput))
        last_num_infer = current_num_infer
        time.sleep(1.0)

# Run inference
model_feed_dict={'input': img_arr3}

executor = futures.ThreadPoolExecutor(max_workers=16+1)
executor.submit(current_throughput)
for i,pred in enumerate(pred_list):
     executor.submit(one_thread, pred, model_feed_dict, i)
     result = pred(model_feed_dict)
     #print(result)
```

执行infer_resnet50.py 文件查看测试结果
`time python infer_resnet50.py`

另开一个terminal，ssh 登录进去，监控GPU使用情况
`watch -n 1  nvidia-smi`

### 4.2.2 MXNet
ssh 到 相应g4dn实例。

下载代码

```
cd ./src/lab4/MXNet-1.6.0

source deactivate

source activate mxnet_p36
```

打开 infer_resnet50.py 文件检查
`more infer_resnet50.py`

```
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

exe = sym.bind(ctx=ctx, args=args, aux_states=aux, grad_req='null',group2ctx=group2ctx)

exe.forward()
prob = exe.outputs[0].asnumpy()
# print the top-5
prob = np.squeeze(prob)
a = np.argsort(prob)[::-1]
for i in a[0:5]:
    print('probability=%f, class=%s' %(prob[i], labels[i]))
```

执行infer_resnet50.py文件
`time python infer_resnet50.py`


打开 infer_resnet50_loadtest.py 文件检查
`more infer_resnet50_loadtest.py`

```
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

```

执行infer_resnet50_loadtest.py文件

`time python infer_resnet50_loadtest.py`

另开一个terminal，ssh 登录进去，监控GPU使用情况
`watch -n 1  nvidia-smi`


### 4.2.3 PyTorch
ssh 到 相应g4dn实例。


```
cd ./src/lab4/PyTorch-1.4.0

source deactivate


source activate pytorch_p36

```

打开 infer_resnet50.py 文件检查
`more infer_resnet50.py`

```
import torch
#model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet18', pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet34', pretrained=True)
model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet50', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet152', pretrained=True)
model.eval()
# Download an example image from the pytorch website
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)
# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
print(output[0].sort()[1][-5:])
#print(torch.nn.functional.softmax(output[0].sort()[1][-5:]))
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
#print(torch.nn.functional.softmax(output[0].sort()[1][-5:], dim=0))
```

执行 infer_resnet50.py 文件
`time python infer_resnet50.py`

打开 infer_resnet50_loadtest.py 文件检查
`more infer_resnet50_loadtest.py`

```
import torch
import os
import time
from concurrent import futures
import numpy as np
#model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet18', pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet34', pretrained=True)
model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet50', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet152', pretrained=True)
model.eval()
# Download an example image from the pytorch website
import urllib
#url, filename = ("https://github.com/pytorch/hub/raw/master/dog.jpg", "dog.jpg")
url, filename = ("https://raw.githubusercontent.com/awslabs/mxnet-model-server/master/docs/images/kitten_small.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)
# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')


USER_BATCH_SIZE = 50
NUM_LOOPS_PER_THREAD = 100
pred_list = [model for _ in range(4)]
pred_list = [
    pred_list[0], pred_list[0], pred_list[0], pred_list[0],
    pred_list[1], pred_list[1], pred_list[1], pred_list[1],
    pred_list[2], pred_list[2], pred_list[2], pred_list[2],
    pred_list[3], pred_list[3], pred_list[3], pred_list[3],
  ]

num_infer_per_thread = []
for i in range(len(pred_list)):
    num_infer_per_thread.append(0)

def one_thread(pred, input_batch, index):
    global num_infer_per_thread
    for _ in range(NUM_LOOPS_PER_THREAD):
        with torch.no_grad():
            result = pred(input_batch)
            num_infer_per_thread[index] += USER_BATCH_SIZE
#            print("result",result)

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
    executor.submit(one_thread, pred, input_batch, i)
```

执行infer_resnet50_loadtest.py文件
`time python infer_resnet50_loadtest.py`

另开一个terminal，ssh 登录进去，监控GPU使用情况
`watch -n 1  nvidia-smi`

### 4.2.4 TensorFlow2 + keras(code is still been debugging)
ssh 到 相应g4dn实例。
```
cd ./src/lab4/Tensorflow-2.1

source deactivate

source activate tensorflow2_p36
```

打开 infer_resnet50_loadtest.py 文件检查
`more infer_resnet50_loadtest.py`

```

```

执行infer_resnet50_loadtest.py文件
`time python infer_resnet50_loadtest.py`

另开一个terminal，ssh 登录进去，监控GPU使用情况
`watch -n 1  nvidia-smi`
