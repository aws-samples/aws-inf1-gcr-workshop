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

