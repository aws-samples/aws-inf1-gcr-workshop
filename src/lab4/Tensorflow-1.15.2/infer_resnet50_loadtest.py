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