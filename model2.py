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
    image = tf.keras.layers.RandomCrop(96,96, seed=2022)(image)

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

# （3）主干网络
def mobilenet_v1(classes, input_shape, alpha, depth_multiplier, dropout_rate):
    # 创建输入层
    inputs = layers.Input(shape=input_shape)
    inputs = process_layer(inputs)
    x = conv_block(inputs, 32, alpha, strides=(2, 2))  # 步长为2，压缩宽高，提升通道数

    x = depthwise_conv_block(x, 64, alpha, depth_multiplier)  # 深度可分离卷积。逐点卷积时卷积核个数为64

    x = depthwise_conv_block(x, 128, alpha, depth_multiplier, strides=(2, 2))  # 步长为2，压缩特征图size
    x = depthwise_conv_block(x, 128, alpha, depth_multiplier)

    x = depthwise_conv_block(x, 256, alpha, depth_multiplier, strides=(2, 2))
    x = depthwise_conv_block(x, 256, alpha, depth_multiplier)

    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, strides=(2, 2))
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier)
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier)
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier)
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier)
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier)

    x = depthwise_conv_block(x, 1024, alpha, depth_multiplier, strides=(2, 2))
    x = depthwise_conv_block(x, 1024, alpha, depth_multiplier)

    x = layers.GlobalAveragePooling2D()(x)  # 通道维度上对size维度求平均
    # 超参数调整卷积核（特征图）个数
    shape = (1, 1, int(1024 * alpha))
    # 调整输出特征图x的特征图个数
    x = layers.Reshape(target_shape=shape)(x)
    # Dropout层随机杀死神经元，防止过拟合
    x = layers.Dropout(rate=dropout_rate)(x)
    # 卷积层，将特征图x的个数转换成分类数
    x = layers.Conv2D(classes, kernel_size=(1, 1), padding='same')(x)
    # 经过softmax函数，变成分类概率
    x = layers.Activation('softmax')(x)
    # 重塑概率数排列形式
    x = layers.Reshape(target_shape=(classes,))(x)
    # 构建模型
    model = Model(inputs, x)

    # 返回模型结构
    return model


if __name__ == '__main__':
    # 获得模型结构
    model = mobilenet_v1(classes=1000,  # 分类种类数
                      input_shape=[128, 128, 3],  # 模型输入图像shape
                      alpha=0.25,  # 超参数，控制卷积核个数
                      depth_multiplier=1,  # 超参数，控制图像分辨率
                      dropout_rate=1e-3)  # 随即杀死神经元的概率

    # 查看网络模型结构
    model.summary()
    model.save("./mbtest.h5", save_format="h5")