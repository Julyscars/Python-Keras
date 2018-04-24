from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression


import tflearn.datasets.oxflower17 as oxflower17

X, Y = oxflower17.load_data(one_hot=True, resize_pics=(227, 227)) 

input_data()
# Building 'AlexNet'

network = input_data(shape=[None, 227, 227, 3])#图像的 高度，宽度，通道rgb,None是批次
network = conv_2d(network, 96, 11, strides=4, activation='relu')
#使用的96个大小规格为11*11的过滤器filter实际大小规格为11*11*3，或者称为卷积核，进行特征提取，
#我们会依据这个公式来提取特征图： 【img_size - filter_size】/stride +1 = new_feture_size,
# 所以这里我们得到的特征图大小为：#[227-11] / 4 + 1 ）= 55 注意【】表示向下取整. 我们得到的新的特征图规格为55*55，
# 注意这里提取到的特征图是彩色的.这样得到了96个55*55大小的特征图了，并且是RGB通道的. 使用RELU激励函数，
# 来确保特征图的值范围在合理范围之内，比如{0,1}，{0,255}输入图片：
# 图片大小 W×H
# Filter大小 F×F
# 步长 S
# padding的像素数 P
#
# 输出特征图：
# 特征图大小 W'×H'
# W' = (W − F + 2P )/S+1
# H' = (H − F + 2P )/S+1
network = max_pool_2d(network, 3, strides=2)
#官方给的是内核是3*3大小，该过程就是3*3区域的数据进行处理（求均值，最大/小值，就是区域求均值，区域求最大值，区域求最小值）
# ，通过降采样处理，我们可以得到#（ [55-3] / 2 + 1 ) = 27 ，也就是得到96个27*27的特征图,然后再以这些特征图，
# 为输入数据，进行第二次卷积.
network = local_response_normalization(network)
#默认的是ACROSS_CHANNELS ，跨通道归一化（这里我称之为弱化），local_size：5（默认值），
#表示局部弱化在相邻五个特征图间中求和并且每一个值除去这个和.
network = conv_2d(network, 256, 5, activation='relu')
#(27-5)/1+1=
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 17, activation='softmax')
network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Training
model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                    max_checkpoints=1, tensorboard_verbose=2)
model.fit(X, Y, n_epoch=1000, validation_set=0.1, shuffle=True,
          show_metric=True, batch_size=64, snapshot_step=200,
          snapshot_epoch=False, run_id='alexnet_oxflowers17')
