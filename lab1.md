
####  1.1 编译Resnet-50模型


* 通过SSH客户端连接到上一步骤创建好的c5.xlarge实例。


* 进入到~/src/lab2目录

  ```shell
  cd ~/src/lab1
  ```
  
* 使用nano或者vi创建内容如下的`compile_resnet50.py`脚本，或者您也可以从src目录获得该源码。该脚本用于将resnet50的模型编译为Neuron优化的版本。

  ```python
  import os
  import time
  import shutil
  import tensorflow as tf
  import tensorflow.neuron as tfn
  import tensorflow.compat.v1.keras as keras
  from tensorflow.keras.applications.resnet50 import ResNet50
  from tensorflow.keras.applications.resnet50 import preprocess_input
  
  # Create a workspace
  WORKSPACE = './ws_resnet50'
  os.makedirs(WORKSPACE, exist_ok=True)
  
  # Prepare export directory (old one removed)
  model_dir = os.path.join(WORKSPACE, 'resnet50')
  compiled_model_dir = os.path.join(WORKSPACE, 'resnet50_neuron')
  shutil.rmtree(model_dir, ignore_errors=True)
  shutil.rmtree(compiled_model_dir, ignore_errors=True)
  
  # Instantiate Keras ResNet50 model
  keras.backend.set_learning_phase(0)
  tf.keras.backend.set_image_data_format('channels_last')
  model = ResNet50(weights='imagenet')
  
  # Export SavedModel
  tf.saved_model.simple_save(
      session            = keras.backend.get_session(),
      export_dir         = model_dir,
      inputs             = {'input': model.inputs[0]},
      outputs            = {'output': model.outputs[0]})
  
  # Compile using Neuron
  #tfn.saved_model.compile(model_dir, compiled_model_dir) #default compiles to 1 neuron core.
  tfn.saved_model.compile(model_dir, compiled_model_dir, compiler_args =['--num-neuroncores', '4']) # compile to 4 neuron cores.
  
  # Prepare SavedModel for uploading to Inf1 instance
  shutil.make_archive('./resnet50_neuron', 'zip', WORKSPACE, 'resnet50_neuron')
  ```


* 运行上述脚本，等待几分钟即可压缩后的模型包resnet50_neuron.zip

  ```shell
  time python compile_resnet50.py  
  ```
* 运行如下代码将编译好的模型上传到S3桶

  ```shell
  aws s3 cp resnet50_neuron.zip s3://resnet50neuron-xxxx/resnet50_neuron.zip
  ```
  



####  1.2 部署及推理


* 通过SSH客户端连接到准备章节中第5步创建好的Inf1.2xlarge实例。


* 查看Inferentia设备信息

  ```shell
  neuron-ls
  ```

* 查看Neuron-RTD进程的状态

  ```shell
  sudo systemctl status neuron-rtd
  ```

* 使用nano或者vi创建内容如下的`infer_resnet50.py`脚本，或者您也可以从src目录获得该源码。该脚本用于执行resnet50模型的推理。

  ```python
  import os
  import time
  import numpy as np
  import tensorflow as tf
  from tensorflow.keras.preprocessing import image
  from tensorflow.keras.applications import resnet50
  
  tf.keras.backend.set_image_data_format('channels_last')
  
  # Create input from image
  img_sgl = image.load_img('kitten_small.jpg', target_size=(224, 224))
  img_arr = image.img_to_array(img_sgl)
  img_arr2 = np.expand_dims(img_arr, axis=0)
  img_arr3 = resnet50.preprocess_input(img_arr2)
  
  # Load model
  COMPILED_MODEL_DIR = './resnet50_neuron/'
  predictor_inferentia = tf.contrib.predictor.from_saved_model(COMPILED_MODEL_DIR)
  
  # Run Inference and Display results
  model_feed_dict={'input': img_arr3}
  infa_rslts = predictor_inferentia(model_feed_dict)
  print(resnet50.decode_predictions(infa_rslts["output"], top=5)[0])
  ```

  

* 运行如下代码从S3桶下载编译好的模型

```
aws s3 cp s3://resnet50neuron-xxxx/resnet50_neuron.zip resnet50_neuron.zip 
```

- 解压缩下载后的模型


```shell
  unzip resnet50_neuron.zip
```

- 下载推理使用的示例图片


```
curl -O https://raw.githubusercontent.com/awslabs/mxnet-model-server/master/docs/images/kitten_small.jpg
```

- 执行推理代码


```shell
python infer_resnet50.py
```

应该会获得如下推理结果

```shell
[('n02123045', 'tabby', 0.684492), ('n02127052', 'lynx', 0.1263369), ('n02123159', 'tiger_cat', 0.086898394), ('n02124075', 'Egyptian_cat', 0.067847595), ('n02128757', 'snow_leopard', 0.00977607)]
```

- 多次执行，计算平均推理时间

```shell
python infer_resnet50_1000times.py
```

#### 1.3 运行推理服务

- 准备编译好的Saved Model

```shell
mkdir -p resnet50_inf1_serve
cp -rf resnet50_neuron resnet50_inf1_serve/1
```

* 运行推理服务

  ```shell
  tensorflow_model_server_neuron --model_name=resnet50_inf1_serve --model_base_path=$(pwd)/resnet50_inf1_serve/ --port=8500
  ```

* 通过另外一个SSH客户端连接到准备章节中第5步创建好的Inf1.2xlarge实例。

* 使用nano或者vi创建内容如下的`tfs_client.py`脚本，或者您也可以从src目录获得该源码。该脚本用于调用推理服务执行resnet50模型的推理。

  ```python
  import numpy as np
  import grpc
  import tensorflow as tf
  from tensorflow.keras.preprocessing import image
  from tensorflow.keras.applications.resnet50 import preprocess_input
  from tensorflow.keras.applications.resnet50 import decode_predictions
  from tensorflow_serving.apis import predict_pb2
  from tensorflow_serving.apis import prediction_service_pb2_grpc
  
  tf.keras.backend.set_image_data_format('channels_last')
  
  if __name__ == '__main__':
      channel = grpc.insecure_channel('localhost:8500')
      stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
      img_file = tf.keras.utils.get_file(
          "./kitten_small.jpg",
          "https://raw.githubusercontent.com/awslabs/mxnet-model-server/master/docs/images/kitten_small.jpg")
      img = image.load_img(img_file, target_size=(224, 224))
      img_array = preprocess_input(image.img_to_array(img)[None, ...])
      request = predict_pb2.PredictRequest()
      request.model_spec.name = 'resnet50_inf1_serve'
      request.inputs['input'].CopyFrom(
          tf.contrib.util.make_tensor_proto(img_array, shape=img_array.shape))
      result = stub.Predict(request)
      prediction = tf.make_ndarray(result.outputs['output'])
      print(decode_predictions(prediction))
  ```


* 执行客户端程序

  ```shell
  python tfs_client.py
  ```

  应该会获得如下推理结果

  ```shell
  [('n02123045', 'tabby', 0.684492), ('n02127052', 'lynx', 0.1263369), ('n02123159', 'tiger_cat', 0.086898394), ('n02124075', 'Egyptian_cat', 0.067847595), ('n02128757', 'snow_leopard', 0.00977607)]
  ```

* 清理

  ```shell
  neuron-cli list-model
  neuron-cli reset
```
  
  

#### 1.4 负载测试

- 下载性能测试代码包


```shell
wget https://reinventinf1.s3.amazonaws.com/keras_fp16_benchmarking_db.tgz
tar -xzf keras_fp16_benchmarking_db.tgz
cd keras_fp16_benchmarking_db
```

* 将Keras ResNet50 FP32进行推理优化，转化为FP16格式。

````shell
python gen_resnet50_keras.py

python optimize_for_inference.py --graph resnet50_fp32_keras.pb --out_graph resnet50_fp32_keras_opt.pb

python fp32tofp16.py  --graph resnet50_fp32_keras_opt.pb --out_graph resnet50_fp16_keras_opt.pb
````


* 运行`pb2sm_compile.py`脚本编译ResNet50固化图，获得`rn50_fp16_compiled_batch5.zip`

  ````shell
  time python pb2sm_compile.py
  ````
  
* 解压缩上一步得到的模型

  ````shell
  unzip ~/rn50_fp16_compiled_batch5.zip
  ````

* 运行`infer_resnet50_keras_loadtest.py`进行负载测试

  ````shell
  time python infer_resnet50_keras_loadtest.py
  ````

  将获得如下运行结果：

  ```
  NUM THREADS:  16
  NUM_LOOPS_PER_THREAD:  100
  USER_BATCH_SIZE:  50
  current throughput: 0 images/sec
  current throughput: 0 images/sec
  current throughput: 1050 images/sec
  current throughput: 2000 images/sec
  current throughput: 2350 images/sec
  current throughput: 2350 images/sec
  current throughput: 2450 images/sec
  current throughput: 2150 images/sec
  current throughput: 2100 images/sec
  current throughput: 2550 images/sec
  current throughput: 2400 images/sec
  current throughput: 2350 images/sec
  current throughput: 2350 images/sec
  current throughput: 2250 images/sec
  current throughput: 2200 images/sec
  current throughput: 2200 images/sec
  current throughput: 2400 images/sec
  current throughput: 2250 images/sec
  current throughput: 2400 images/sec
  current throughput: 2350 images/sec
  current throughput: 2300 images/sec
  current throughput: 2350 images/sec
  current throughput: 2350 images/sec
  current throughput: 2300 images/sec
  current throughput: 2300 images/sec
  current throughput: 2300 images/sec
  current throughput: 2300 images/sec
  current throughput: 2250 images/sec
  current throughput: 2300 images/sec
  current throughput: 2350 images/sec
  current throughput: 2200 images/sec
  current throughput: 2400 images/sec
  current throughput: 2400 images/sec
  current throughput: 2150 images/sec
  current throughput: 2250 images/sec
  current throughput: 2300 images/sec
  current throughput: 1900 images/sec
  current throughput: 1050 images/sec
  current throughput: 100 images/sec
  
  real	0m46.553s
  user	1m23.796s
  sys	0m4.248s
  ```

* 在运行上一步的负载测试的同时，可以通过另外一个SSH客户端连接到该Inf1.2xlarge实例，通过如下命令查看资源使用率。

  ````shell
  neuron-top
  ````
  

[go to lab2](https://code.awsrun.com/zhazhn/inf1/src/branch/master/lab2.md)
