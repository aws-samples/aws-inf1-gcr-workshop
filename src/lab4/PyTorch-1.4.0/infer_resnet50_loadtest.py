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
    