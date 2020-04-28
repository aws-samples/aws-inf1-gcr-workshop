使用您的账号登陆到Amazon Management Console，选择**弗吉尼亚北部**区域；

#### 0.1. 创建S3存储桶

访问AWS管理控制台，导航到S3存储桶。点击**创建存储桶**，在存储桶名称中输入**resnet50neuron-xxxx**，后缀为自定义随机数字。区域选择**美国东部（弗吉尼亚北部）us-east-1**，点击**创建存储桶**。

#### 0.2. 创建角色

访问AWS管理控制台，导航到IAM页面，在左侧导航栏中选择**角色**，点击**创建角色**，选择**EC2**，点击下一步按钮，在attach权限策略中选择**AmazonS3FullAccess**，连续点击两次下一步按钮，在角色名称中输入**inf1s3access**，然后点击**创建角色**按钮，角色创建成功。

#### 0.3. 创建C5.xlarge实例

#### 0.3.1 创建实例

访问EC2门户，点击**启动实例**，AMI选择**Deep Learning AMI (Ubuntu 16.04) Version 27.0 (ami-0a79b70001264b442)**， 实例类型选择**c5.xlarge**，实例详细信息页面中**IAM角色**选择刚刚创建的**inf1s3access**角色，在添加存储页面，添加**50G**EBS类型的SSD存储，在添加标签页面，添加一组标签，**键**输入**Name**，**值**输入**lab_comp**。安全组选择默认安全组，在审核和启动页面，选择一个已经存在的SSH密钥对。

点击**启动实例**按钮以启动一个新的实例。

#### 0.3.2 安装Neuron SDK环境

连接到您刚创建的名称为**lab_comp**的实例。SSH的配置方法可以参考附录。

* 创建虚拟环境

```shell
sudo apt-get update
sudo apt-get -y install virtualenv
```

注意：如果在apt-get安装过程中看到以下错误，请等待一分钟左右，以完成后台更新，然后重试apt-get安装：

```shell
E: Could not get lock /var/lib/dpkg/lock-frontend - open (11: Resource temporarily unavailable)
E: Unable to acquire the dpkg frontend lock (/var/lib/dpkg/lock-frontend), is another process using it?
```

* 配置一个新的python3.6的虚拟环境。

```shell
virtualenv --python=python3.6 test_env_p36
source test_env_p36/bin/activate
```

* 自动激活test_env_p36作为当前的实验环境。编辑.bashrc文件，在文件最后加入以下代码

```shell
source test_env_p36/bin/activate
```

* 修改pip的配置，指向Neuron存储库

```shell
tee $VIRTUAL_ENV/pip.conf > /dev/null <<EOF
[global]
extra-index-url = https://pip.repos.neuron.amazonaws.com
EOF
```
#### 0.3.3 下载试验代码
```shell
cd ~
git clone https://github.com/aws-samples/aws-inf1-gcr-workshop
mv ./aws-inf1-gcr-workshop/src .
```



#### 0.4. 创建Inf1.2xlarge实例

#### 0.4.1 创建实例

访问EC2门户，点击**启动实例**，AMI选择**Deep Learning AMI (Ubuntu 16.04) Version 27.0 (ami-0a79b70001264b442)**， 在Machine learning ASIC instances系列中选择**inf1.2xlarge**实例类型，实例详细信息页面中**IAM角色**选择刚刚创建的**inf1s3access**角色，在添加存储页面，添加**50G**EBS类型的SSD存储，在添加标签页面，添加一组标签，**键**输入**Name**，**值**输入**lab_inf**。安全组选择默认安全组，在审核和启动页面，选择一个已经存在的SSH密钥对。

点击**启动实例**按钮以启动一个新的实例

#### 0.4.2 安装Neuron SDK环境

连接到您刚创建的名称为**lab_inf**的实例。SSH的配置方法可以参考附录。

* 创建虚拟环境

```shell
sudo apt-get update
sudo apt-get -y install virtualenv
```

注意：如果在apt-get安装过程中看到以下错误，请等待一分钟左右，以完成后台更新，然后重试apt-get安装：

```shell
E: Could not get lock /var/lib/dpkg/lock-frontend - open (11: Resource temporarily unavailable)
E: Unable to acquire the dpkg frontend lock (/var/lib/dpkg/lock-frontend), is another process using it?
```

* 配置一个新的python3.6的虚拟环境。

```shell
virtualenv --python=python3.6 test_env_p36
source test_env_p36/bin/activate
```

* 自动激活test_env_p36作为当前的实验环境。编辑.bashrc文件，在文件最后加入以下代码

```shell
source test_env_p36/bin/activate
```

* 修改pip的配置，指向Neuron存储库

```shell
tee $VIRTUAL_ENV/pip.conf > /dev/null <<EOF
[global]
extra-index-url = https://pip.repos.neuron.amazonaws.com
EOF
```

#### 0.4.3 下载试验代码
```shell
cd ~
git clone https://github.com/aws-samples/aws-inf1-gcr-workshop
mv ./aws-inf1-gcr-workshop/src .
```

[go to lab1](https://code.awsrun.com/zhazhn/inf1/src/branch/master/lab1.md)


#### 附录：配置SSH环境

具体可以参考：[使用 SSH 连接到 Linux 实例](https://docs.aws.amazon.com/zh_cn/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html)

