####  2.1 编译Resnet-50模型

* 使用SSH工具连接到环境准备章节中创建的C5.xlarge实例。

* 安装MXNet-Neuron 和 Neuron Compiler

  ```shell
  pip install mxnet-neuron
  pip install neuron-cc
  ```


* 进入到~/src/lab2目录

  ```shell
  cd ~/src/lab2
  ```
* 下载预训练的resnet-50模型

  ```shell
  wget http://data.mxnet.io/models/imagenet/resnet/50-layers/resnet-50-0000.params
  wget http://data.mxnet.io/models/imagenet/resnet/50-layers/resnet-50-symbol.json
  ```
  
* 运行compile_resnet50.py编译模型

  ```shell
  python compile_resnet50.py
  ```

  

* 将编译后到模型复制到Inf1实例

  ```shell
  aws s3 cp resnet-50_compiled-0000.params s3://resnet50neuron-xxxx/resnet-50_compiled-0000.params
  aws s3 cp resnet-50_compiled-symbol.json s3://resnet50neuron-xxxx/resnet-50_compiled-symbol.json
  ```


####  2.2 执行推理

* 使用SSH工具连接到环境准备章节中创建的Inf12xlarge实例。

* 安装配置Neuron-RTD

* 修改apt存储库配置以指向Neuron存储库。

  ```shell
  sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
  deb https://apt.repos.neuron.amazonaws.com xenial main
  EOF
  
  wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -
   
  sudo apt-get update
  sudo apt-get install aws-neuron-runtime
  sudo apt-get install aws-neuron-tools
  ```

  注意：如果在apt-get安装过程中看到以下错误，请等待一分钟左右，以完成后台更新，然后重试apt-get安装：

  ```shell
  E: Could not get lock /var/lib/dpkg/lock-frontend - open (11: Resource temporarily unavailable)
  E: Unable to acquire the dpkg frontend lock (/var/lib/dpkg/lock-frontend), is another process using it?
  ```

* 配置nr_hugepages


​       Neuron运行时将2MB的hugepages用于所有已加载模型的输入要素映射缓冲区和输出要素映射缓冲区。默认情况下，Nuron运行时每个Inferentia使用128个2MB的hugepages。hugepages是系统范围的资源。大型页面的分配应在引导时或引导后尽快完成。要在启动时进行分配，请向内核传递hugepages选项。

您可在Inf1实例重新引导后修改2MB hugepages

```
sudo sysctl -w vm.nr_hugepages=128
```

如果您要永久修改2MB hugepages，请将以下内容添加到/etc/sysctl.conf中，然后重新启动实例

```shell
vm.nr_hugepages=128
```

通过以下命令可以查看2MB hugepages的数量

```shell
grep HugePages_Total /proc/meminfo | awk {'print $2'}
```

* 查看Inferentia设备信息

  ```shell
  neuron-ls
  ```

* 查看Neuron-RTD进程的状态

  ```shell
  sudo systemctl status neuron-rtd
  ```

* 安装MXNet-Neuron 

  ```shell
  pip install mxnet-neuron
  ```

* 进入lab2目录

  ```shell
  cd ~/src/lab2
  ```

* 运行如下代码从S3桶下载编译好的模型

  ```shell
  aws s3 cp s3://resnet50neuron-xxxx/resnet-50_compiled-0000.params resnet-50_compiled-0000.params
  aws s3 cp s3://resnet50neuron-xxxx/resnet-50_compiled-symbol.json resnet-50_compiled-symbol.json
  ```
  
* 运行infer_resnet50.py执行推理

  ```shell
  python infer_resnet50.py
  ```


* 观察运行结果。

* 多次执行，计算平均推理时间

  ```shell
  python infer_resnet50_1000times.py
  ```

  

#### 2.3 搭建web服务

在前面的实验中，您已经编译Resnet-50模型。下面准备签名文件signature.json以配置输入名称（name）和形状（shape）：

```json
{
  "inputs": [
    {
      "data_name": "data",
      "data_shape": [
        1,
        3,
        224,
        224
      ]
    }
  ]
}
```

* 使用mxnet框架中，data_shape的格式是[1,3,224,224]，(N, C, H, W)分别表示batch大小,channel数目,高度,宽度。


* 首先安装model-archiver

  ````shell
  pip install model-archiver
  ````
  

  
* 用模型归档工具(model-archiver)打包模型

  ````shell
  cd ~/src
  model-archiver --force --model-name resnet-50_compiled --model-path lab2 --handler mxnet_vision_service:handle
  ````

* 安装

  ````shell
  sudo apt-get install -y -q default-jre 
  ````

  ````shell
  pip install mxnet-model-server
  ````

  

* 启动MXNet模型服务器（MMS）并使用RESTful API加载模型。

  ````shell
  cd ~/src/
  mxnet-model-server --start --model-store ~/src/
  # Pipe to log file if you want to keep a log of MMS
  curl -v -X POST "http://localhost:8081/models?initial_workers=1&max_workers=1&synchronous=true&url=resnet-50_compiled.mar"
  sleep 10 # allow sufficient time to load model
  ````

  每个worker都需要NeuronCore组，该组可以容纳已编译的模型。只要有足够的NeuronCore，就可以通过增加max_workers配置来添加其他工作程序。使用neuron-cli list-ncg查看已创建的NeuronCore组。

* 使用示例图像测试推理服务

  ````shell
  cd ~/src/lab2/
  curl -X POST http://127.0.0.1:8080/predictions/resnet-50_compiled -T test.jpg
  ````

  您将能看到以下的输出

  ````json
  [
    {
      "probability": 0.6375716328620911,
      "class": "n02123045 tabby, tabby cat"
    },
    {
      "probability": 0.1692783385515213,
      "class": "n02123159 tiger cat"
    },
    {
      "probability": 0.12187337130308151,
      "class": "n02124075 Egyptian cat"
    },
    {
      "probability": 0.028840631246566772,
      "class": "n02127052 lynx, catamount"
    },
    {
      "probability": 0.019691042602062225,
      "class": "n02129604 tiger, Panthera tigris"
    }
  ]
  ````

* 测试后进行环境清理，请通过RESTful API发出删除命令并停止模型服务器

  ````shell
  curl -X DELETE http://127.0.0.1:8081/models/resnet-50_compiled
  
  mxnet-model-server --stop
  
  /opt/aws/neuron/bin/neuron-cli reset
  ````

  

[go to lab3](https://code.awsrun.com/zhazhn/inf1/src/branch/master/lab3.md)