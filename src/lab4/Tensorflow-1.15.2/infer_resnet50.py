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