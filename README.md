# segmentation_shengteng2021
2021“昇腾杯”遥感影像智能处理算法大赛——语义分割赛道，冠军方案
## 赛题
[2021“昇腾杯”遥感影像智能处理算法大赛](http://rsipac.whu.edu.cn/index)


遥感图像语义分割竞赛即利用遥感图像中各类地物光谱信息和空间信息进行分析，将图像中具有语义信息的各个像元分别赋予语义类别标签。
本项竞赛提供包含R、G、B、NIR四个波段的遥感影像数据，分辨在0.8米-2米，考察提出算法在47类地物要素中的分类能力。

## 算法方案 ##
- **Baseline选择**
  - 将Swin Transformer+Upernet作为基础网络。

- **波段选取**
  - 为了有效利用rgb和近红外四波段的数据，尝试了rgb、rgbn（无预训练模型）、ngb数据，最终选用效果最好的ngb数据。

- **数据增强**
  - 为了增强模型泛化性，团队采用水平、垂直、水平垂直翻转（p=0.5）作为几何上的数据增强，
  采用光度失真（p=0.5）作为色彩上的数据增强。光度失真指调整图像的亮度，色度，对比度，饱和度，以及加入噪点。
  
- **损失函数设计**
  - **OHEM难例挖掘：** 为了更加关注分类困难的样本，采用难例挖掘。具体做法是对于每个像素点，只计算分类概率<0.7像素点的损失。
  - **Soft-label交叉熵损失：** 为了应对细粒度分类中，二级类目相似度过高的问题，团队采用Soft-label交叉熵损失，将原本的hard label转换成soft label进行学习。
 
- **多尺度策略**
  - 多尺度训练上，图片大小（512，512），在区间[0.75，1.25]随机选择尺度。
  - 多尺度测试上，图片大小（512，512），尺度为{0.75、1.0、1.25}。
  - 决赛时，为了加快测试速度，采用了1.25单尺度无增强测试。

- **整图测试与滑窗测试**
  - 为了应对测试集的大图，尝试了整图和滑窗测试，不过在复赛中，发现两者精度差距并不大，故最终采用了整图测试。
  
- **速度提升策略**
  - **目的：**
    1. 复赛24小时内完成训练和测试，必须提升两者速度；
    2. swin属于transformer结构，其优化器Adamw并不像SGD一样，swin不满足lr和bs呈线性的规律，
    所以为了在V100 32G的显存下，最大程度维持default setting，保持bs=16的默认设置，进行了以下的修改。
  - **做法：**
    1. fp16混合精度训练；
    2. 将LN关闭；
    3. 将swin的前两层冻结，训练过程中不再更新；
    4. 将解码部分的层数从512减小384；
    
- **单模型多结果融合**
  - 因为比赛要求“高精度和高速度”两者兼顾，常见的“多模型融合”能显著提升精度，但也会极大降低速度。所以，采用了EMA指数移动平均和SWA随机权值平均两种模型融合方法。
  
## 环境 ##
```
python 3.7
pytorch 1.6.0
mmseg 0.18.0
```
按[mmsegmentation要求](https://github.com/open-mmlab/mmsegmentation)安装mmseg 0.18.0

## 模型训练与测试 ##
 - **数据集位置** 
 ```
|--train
|  |--images
|  |--labels
|--workspace
    |--mmsegmentation-master
        |--trainval.txt
        |--val.txt
```
 - **下载预训练模型**
 
 
      [这里](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth)下载预训练模型，
  并用tools/model_converters/swin2mmseg.py 转换格式，最终存入pretrain_new/中。
  
 - **训练测试**
 
 
    保存代码至/workspace/mmsegmentation-master/下。
 
 
    进行训练，并对/input_path下的图片进行测试，将结果存入/output_path。
```
cd /workspace/mmsegmentation-master/
python tools/train.py myconfigs/swin_base_ngb.py --gpus 1 --work-dir /save_dir/swin_base_ngb/ --no-validate
python run.py /input_path /output_path --config /workspace/mmsegmentation-master/myconfigs/swin_base_ngb.py --checkpoint /save_dir/swin_base_ngb/iter_80000.pth
```
 
