# 基于深度强化学习的移动机器人避障导航

### 要求

python 3.5

Tensorflow 1.14.0

ROS Melodic

### 使用步骤

因为有未知问题，需要把小车在gazebo中的启动，与tesorflow强化学习分开成两个文件夹，合在一起会报错

##### 1.创建虚拟环境 NDDDQN

##### 2.安装tensorflow

```
pip install tensorflow-gpu==1.14.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

##### 3.在两个工作空间进行编译

在catkin_ws和catkin_ws1分别编译：

```
catkin_make --cmake-args \
            -DCMAKE_BUILD_TYPE=Release \
            -DPYTHON_EXECUTABLE=/usr/bin/python3 \
            -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
            -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
```

##### 4.运行

首先在运行小车的catkin_ws1文件夹中：

```
cd catkin_ws1
source devel/setup.sh
roslaunch pioneer_utils xxx
```

xxx对应运行环境：

```
                                     bizhang.launch   静态避障

​                                daohang.launch   静态导航

​                                 dongtai.launch    动态导航

​              keyboard_teleop.launch    键盘控制
```

然后在运行强化学习的文件夹catkin_ws中：

```
conda activate NDDDQN
cd catkin_ws
source devel/setup.sh
cd src/Tensorflow/xxx
python main.py
```

xxx对应运行算法：

```
                                     DQN-bizhang     静态避障-DQN

​                              DDQN-bizhang     静态避障-DDQN

​                DQN-Dueling-bizhang      静态避障-Dueling-DQN

​             DDQN-Dueling-bizhang      静态避障-Dueling-DDQN

​          NDDQN-Dueling-bizhang      静态避障-Dueling-NDDQN

​  Beta-DDQN-Dueling-bizhang      静态避障-Beta-Dueling-DDQN

​                        Empty-Navigation      静态导航-Dueling-NDDQN

​     separate-Empty-Navigation      静态导航-separate-Dueling-NDDQN

​                         Navigation-DDQN       静态导航-DDQN

​                       people-Navigation       动态导航-Dueling-NDDQN
```

##### 5.可能出现的问题

###### （1）安装的库不足

解决方法：

```
sudo apt update

sudo apt install python3-catkin-pkg-modules python3-rospkg-modules python3-empy

sudo apt-get install ros-kinetic-cmake-modules
```

###### （2）dynamic module does not define module export function (PyInit tf2)

解决方法：见https://blog.csdn.net/weixin_42044401/article/details/111246979

###### （3） Could not dlopen library 'libcudnn.so.x（x为数字）

解决方法：cuda版本和cudnn版本要和显卡驱动版本以及TensorFlow版本对应