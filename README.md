# AWS机器学习推理芯片Inferentia动手实验

##### 版本 - Version 1.2 （2020-03-21）

## 实验介绍 

在本研讨会中，您将获得由定制AWS Inferentia芯片提供支持的Amazon EC2 Inf1实例的动手经验。 Amazon EC2 Inf1实例在云中提供了低延迟，高吞吐量和经济高效的机器学习推理。本讲习班将引导您通过使用经过训练的深度学习模型，通过使用AWS Neuron（一种用于使用AWS Inferentia处理器优化推理的SDK）在Amazon EC2 Inf1实例上进行部署。

## 技能要求

要成功完成本实验，您应该熟悉AWS管理控制台的基本导航以及Amazon Sagemaker上笔记本的导航和运行。

## 环境准备

本实验的所有步骤均在US East (N. Virginia) us-east-1区域进行。在实验前，请参照[lab0](lab0.md)准备以下实例。

1. C5.xlarge实例
2. Inf1.2xlarge实例
2. G4实例 (可选)

## 实验内容

#### 实验1 在inf1实例上测试基于tensorflow框架的预训练模型([lab1](lab1.md))

##### 1.1 编译Resnet-50模型
##### 1.2 部署及推理
##### 1.3 运行推理服务
##### 1.4 负载测试

####  实验2 在inf1实例上测试基于mxnet框架的预训练模型([lab2](lab2.md))
##### 2.1 编译Resnet-50模型
##### 2.2 部署及推理
##### 2.3 搭建web服务

####  实验3 在inf1实例上测试基于pytorch框架的预训练模型([lab3](lab3.md))
##### 3.1 编译Resnet-50模型
##### 3.2 部署及推理
##### 3.3 负载测试

####  实验4 在G4实例上测试预训练模型Resnet-50([lab4](lab4.md)) 
##### 4.1 设置G4实例
##### 4.2 性能测试(TensorFlow,MXNet,PyTorch)

## 参考资料

 https://docs.aws.amazon.com/zh_cn/dlami/latest/devguide/tutorial-inferentia.html

 https://github.com/aws/aws-neuron-sdk

 https://github.com/awshlabs/reinvent19Inf1Lab









