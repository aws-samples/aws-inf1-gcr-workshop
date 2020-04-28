

# 实验3 在inf1实例上测试基于pytorch框架的预训练模型(lab3)

## 3.1 编译Resnet-50模型

通过SSH客户端连接到 lab0 中创建好的c5.xlarge实例。

切换环境
`source test_env_p36/bin/activate`

运行下列命令安装torch-neuron等包

```
pip install torch-neuron
pip install neuron-cc[tensorflow]
pip install pillow==6.2.2
pip install torchvision==0.4.2 --no-deps
```


使用nano或者vi创建内容如下的compile_resnet50.py脚本，或者您也可以从src目录获得该源码。该脚本用于将resnet50的模型编译为Neuron优化的版本。

```
import torch
import numpy as np
import os
import torch_neuron
from torchvision import models
 
image = torch.zeros([1, 3, 224, 224], dtype=torch.float32)
 
## Load a pretrained ResNet50 model
model = models.resnet50(pretrained=True)

## Tell the model we are using it for evaluation (not training)
model.eval()
model_neuron = torch.neuron.trace(model, example_inputs=[image])

## Export to saved model
model_neuron.save("resnet50_neuron.pt")
```
  

运行上述脚本，等待几分钟即可在当前目录看到模型文件resnet50_neuron.pt

`time python compile_resnet50.py` 


确认看到下列成功信息
`INFO:Neuron:compiling module ResNet with neuron-cc`


运行如下命令将编译好的模型上传到S3桶

`aws s3 cp resnet50_neuron.pt s3://resnet50neuron-xxxx/resnet50_neuron.pt`

## 3.2 部署及推理

通过SSH客户端连接到 lab0 中创建好的inf1.2xlarge实例。

切换环境
`source test_env_p36/bin/activate`

运行下列命令安装torch-neuron等包
```
pip install torch-neuron
pip install pillow==6.2.2
pip install torchvision==0.4.2 --no-deps
```

运行如下命令将编译好的模型从S3桶下载到本地

`aws s3 cp s3://resnet50neuron-xxxx/resnet50_neuron.pt resnet50_neuron.pt`


使用nano或者vi创建内容如下的 infer_resnet50.py 脚本，或者您也可以从src目录获得该源码。该脚本用于使用Neuron优化后的模型进行推理。

```
import os
import time
import torch
import torch_neuron
import json
import numpy as np

from urllib import request

from torchvision import models, transforms, datasets

## Create an image directory containing a small kitten
os.makedirs("./torch_neuron_test/images", exist_ok=True)
request.urlretrieve("https://raw.githubusercontent.com/awslabs/mxnet-model-server/master/docs/images/kitten_small.jpg",
                    "./torch_neuron_test/images/kitten_small.jpg")


## Fetch labels to output the top classifications
request.urlretrieve("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json","imagenet_class_index.json")
idx2label = []

with open("imagenet_class_index.json", "r") as read_file:
    class_idx = json.load(read_file)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

## Import a sample image and normalize it into a tensor
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])

eval_dataset = datasets.ImageFolder(
    os.path.dirname("./torch_neuron_test/"),
    transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    normalize,
    ])
)

image, _ = eval_dataset[0]
image = torch.tensor(image.numpy()[np.newaxis, ...])

## Load model
model_neuron = torch.jit.load( 'resnet50_neuron.pt' )

## Predict
results = model_neuron( image )

# Get the top 5 results
top5_idx = results[0].sort()[1][-5:]

# Lookup and print the top 5 labels
top5_labels = [idx2label[idx] for idx in top5_idx]

print("Top 5 labels:\n {}".format(top5_labels) )
```


运行infer_resnet50.py文件检查推理结果

`time python infer_resnet50.py`

多次执行，计算平均推理时间

`python infer_resnet50_1000times.py`

运行下列命令清理model
```
neuron-cli list-model
neuron-cli reset
```

## 3.3 负载测试(code is still been tunning)

使用nano或者vi创建内容如下的 infer_resnet50_loadtest.py 脚本，或者您也可以从src目录获得该源码。该脚本用于对Neuron优化后的模型进行推理压力测试。


```
import os
import time
import torch
import torch_neuron
import json
import numpy as np
from concurrent import futures
from urllib import request

from torchvision import models, transforms, datasets

## Create an image directory containing a small kitten
os.makedirs("./torch_neuron_test/images", exist_ok=True)
request.urlretrieve("https://raw.githubusercontent.com/awslabs/mxnet-model-server/master/docs/images/kitten_small.jpg",
                    "./torch_neuron_test/images/kitten_small.jpg")


## Fetch labels to output the top classifications
request.urlretrieve("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json","imagenet_class_index.json")
idx2label = []

with open("imagenet_class_index.json", "r") as read_file:
    class_idx = json.load(read_file)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

## Import a sample image and normalize it into a tensor
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])

eval_dataset = datasets.ImageFolder(
    os.path.dirname("./torch_neuron_test/"),
    transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    normalize,
    ])
)

image, _ = eval_dataset[0]
image = torch.tensor(image.numpy()[np.newaxis, ...])

# begin of infer once
## Load model
#model_neuron = torch.jit.load( 'resnet50_neuron.pt' )

## Predict
#results = model_neuron( image )

# Get the top 5 results
#top5_idx = results[0].sort()[1][-5:]

# Lookup and print the top 5 labels
#top5_labels = [idx2label[idx] for idx in top5_idx]

#print("Top 5 labels:\n {}".format(top5_labels) )
# end of infer once

USER_BATCH_SIZE = 50
NUM_LOOPS_PER_THREAD = 100
pred_list = [torch.jit.load( 'resnet50_neuron.pt' ) for _ in range(4)]
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
    executor.submit(one_thread, pred, image, i)
```


通过SSH客户端连接到 lab0 中创建好的inf1.2xlarge实例。使用neuron工具命令查看资源消耗。

```
neuron-top

watch -n1 neuron-cli list-model

neuron-ls
```

[go to lab4](https://code.awsrun.com/zhazhn/inf1/src/branch/master/lab4.md)