## **1.Readme：文件介绍**

总的来说吧，复现没有完成，算是跑通吧，可以运行 `train` 部分，`test` 等部分都没有尝试

数据集也仅仅尝试了 `celeba_hq` 数据集

下面开始介绍文件:

```
├── core
│   ├── __init__.py
│   ├── checkpoint_.py
│   ├── data_loader_.py
│   ├── model_.py
│   ├── solver_.py
│   ├── transforms_.py
│   ├── utils_.py
│   └── wing_.py
├── data
│   └── data48884
│       └── data.zip
├── expr
│   └── checkpoints
└── main_.py
```

`main_.py` 和 `Pytorch` 版本的 `main.py` 文件基本相同

`core/__init__.py` 和 `Pytorch` 版本的 `core/__init__.py` 均为空

`core/checkpoint_.py` 和 `Pytorch` 版本的 `core/checkpoint.py` 相同, 将 Pytorch 版本翻译为 `paddlepaddle` 版本

`core/transforms_.py` 和 `Pytorch` 的 `torchvision.transforms` 相对应，为了不被人说我抄袭 `Pytorch` 的 `transforms` 函数，我的底层是基于 `numpy` 和 `cv2`(`torchvision`基于`PIL`)

`core/data_loader_.py` 和 `Pytorch` 版本的 `core/data_loader.py` 对应，笔者基于 `paddle.fluid.io.xmap_readers` 封装了好几个类

`core/utils_.py` 和 `Pytorch` 版本的 `core/utils.py` 对应，函数基本都是对应的，但是部分还没改完，`fluid.dygraph.no_grad` 的bug 好像在2.0版本依旧没改完(一孔之见)

`core/model_.py` 和 `Pytorch` 版本的 `core/model.py` 对应，完全实现了对应的过程

`core/solver_.py` 和 `Pytorch` 版本的 `core/solver.py` 对应，实现了部分过程

`core/wing_.py` 为空, 没有使用

`data`目录和`expr`目录不再赘述

## **2.Readme：运行过程**

```shell
pip install munch -i https://pypi.tuna.tsinghua.edu.cn/simple
vim /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/container.py
# 键盘输出 `273G`, 将那个 `assert` 注释掉
# 按 `Esc` 然后 `:wq`

# 然后需要解压数据文件(AI Studio的数据集)
unzip data/data48884/data.zip -d data/celeba_hq
# 复制到对应的位置
cp -rf data/celeba_hq/data/celeba_hq data/
```

这个 `LayerList` 有bug, 提交了两次pr[#26793](https://github.com/PaddlePaddle/Paddle/pull/26793)和[#26790](https://github.com/PaddlePaddle/Paddle/pull/26790), 一次[issue](https://github.com/PaddlePaddle/Paddle/issues/26795)

## **3.Notice**

我本地环境是 `paddlepaddle1.8.4` AI Studio目前不支持，需要手动安装
```shell
python -m pip install paddlepaddle-gpu==1.8.4.post107 -i https://mirror.baidu.com/pypi/simple

# 同时需要pip安装munch
pip install munch -i https://mirror.baidu.com/pypi/simple
```

## **4.亿点心得**

距离论文复现结束还有50分钟，我想留下一下话与诸君共勉

