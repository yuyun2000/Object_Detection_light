import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, optimizers

# （1）标准卷积模块
def conv_block(input_tensor, filters, alpha, kernel_size=(3, 3), strides=(1, 1)):
    # 超参数alpha控制卷积核个数
    filters = int(filters * alpha)

    # 卷积+批标准化+激活函数
    x = layers.Conv2D(filters, kernel_size,
                      strides=strides,  # 步长
                      padding='same',  # 0填充，卷积后特征图size不变
                      use_bias=False)(input_tensor)  # 有BN层就不需要计算偏置

    x = layers.BatchNormalization()(x)  # 批标准化

    x = layers.ReLU(6.0)(x)  # relu6激活函数

    return x  # 返回一次标准卷积后的结果

# （2）深度可分离卷积块
def depthwise_conv_block(input_tensor, point_filters, alpha, depth_multiplier, strides=(1, 1)):
    # 超参数alpha控制逐点卷积的卷积核个数
    point_filters = int(point_filters * alpha)

    # ① 深度卷积--输出特征图个数和输入特征图的通道数相同
    x = layers.DepthwiseConv2D(kernel_size=(3, 3),  # 卷积核size默认3*3
                               strides=strides,  # 步长
                               padding='same',  # strides=1时，卷积过程中特征图size不变
                               depth_multiplier=depth_multiplier,  # 超参数，控制卷积层中间输出特征图的长宽
                               use_bias=False)(input_tensor)  # 有BN层就不需要偏置

    x = layers.BatchNormalization()(x)  # 批标准化

    x = layers.ReLU(6.0)(x)  # relu6激活函数

    # ② 逐点卷积--1*1标准卷积
    x = layers.Conv2D(point_filters, kernel_size=(1, 1),  # 卷积核默认1*1
                      padding='same',  # 卷积过程中特征图size不变
                      strides=(1, 1),  # 步长为1，对特征图上每个像素点卷积
                      use_bias=False)(x)  # 有BN层，不需要偏置

    x = layers.BatchNormalization()(x)  # 批标准化

    x = layers.ReLU(6.0)(x)  # 激活函数

    return x  # 返回深度可分离卷积结果

import numpy as np

def process_layer(image):

    image = tf.keras.layers.RandomHeight(0.2,interpolation='bilinear', seed=2022)(image)
    image = tf.keras.layers.RandomWidth(0.2, interpolation='bilinear', seed=2022)(image)
    image = tf.keras.layers.RandomCrop(128,128, seed=2022)(image)

    np.random.seed(2022)#不同的顺序会影响最终的结果，设定随机的顺序减少这些影响
    mode = np.random.randint(3)
    if mode ==0:
        image = tf.image.random_brightness(image,max_delta=0.125,seed=2022)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif mode == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    elif mode == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    elif mode == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    image = tf.image.random_flip_left_right(image,2022)
    return tf.clip_by_value(image,0.0,1.0)#把最终的结果限制在0-1的区间

premodel = tf.keras.models.load_model("./mbnet2.h5")
# premodel.summary()
# （3）主干网络
def mobilenet_v1( input_shape, alpha, depth_multiplier, dropout_rate):
    # 创建输入层
    # inputs = layers.Input(shape=input_shape)
    # inputs = process_layer(inputs)

    cut_point = premodel.get_layer('re_lu_22').output
    x = layers.Conv2D(64,(1,1),1,activation='relu',name='head1')(cut_point)

    x = layers.Conv2D(32, (1, 1), 1, activation='relu', name='head3')(x)
    x = layers.Conv2D(8, (1, 1), 1, activation='relu', name='head5')(x)
    x = layers.Conv2D(1, (1, 1), 1, activation='relu', name='logits')(x)

    # 构建模型
    model = Model(premodel.inputs, x)

    # 返回模型结构
    return model

if __name__ == '__main__':
    # 获得模型结构
    model = mobilenet_v1(input_shape=[128, 128, 3],  # 模型输入图像shape
                      alpha=0.25,  # 超参数，控制卷积核个数
                      depth_multiplier=1,  # 超参数，控制图像分辨率
                      dropout_rate=1e-3)  # 随即杀死神经元的概率
    # 查看网络模型结构
    model.summary()
    # model.save("./mbtest.h5", save_format="h5")
    # print(model.layers[-3])