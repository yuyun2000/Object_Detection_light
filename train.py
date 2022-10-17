from model import mobilenet_v1
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import optimizers
from data import train_iterator
import config as c


def train_step(model, images, labels, optimizer):
    with tf.GradientTape() as tape:
        prediction = model(images, training=True)

        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,logits=prediction)*labels
        # print(loss.shape)
        loss = tf.reduce_mean(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, prediction

def train(model, data_iterator, optimizer):

    for i in tqdm(range(int(c.imgnum/ c.batchsize))):
        images, labels = data_iterator.next()
        ce, prediction = train_step(model, images, labels, optimizer)

        print('ce: {:.4f}'.format(ce))

class CosineDecayWithWarmUP(tf.keras.experimental.CosineDecay):
    def __init__(self, initial_learning_rate, decay_steps, alpha=0.0, warm_up_step=0, name=None):
        self.warm_up_step = warm_up_step
        super(CosineDecayWithWarmUP, self).__init__(initial_learning_rate=initial_learning_rate,
                                                    decay_steps=decay_steps,
                                                    alpha=alpha,
                                                    name=name)
    @tf.function
    def __call__(self, step):
        if step <= self.warm_up_step:
            return step / self.warm_up_step * self.initial_learning_rate
        else:
            return super(CosineDecayWithWarmUP, self).__call__(step - self.warm_up_step)

if __name__ == '__main__':
    train_data_iterator = train_iterator()
    model = mobilenet_v1(input_shape=[128, 128, 3],  # 模型输入图像shape
                      alpha=0.25,  # 超参数，控制卷积核个数
                      depth_multiplier=1,  # 超参数，控制图像分辨率
                      dropout_rate=1e-3)  # 随即杀死神经元的概率
    model.build(input_shape=(None,) + (128,128,3))
    # model.summary()

    # optimizer = optimizers.SGD(learning_rate=0.0001, momentum=0.9)
    # optimizer = optimizers.Adam()

    import tensorflow_addons as tfa
    optimizers = [
        tf.keras.optimizers.Adam(learning_rate=1e-6),
        tf.keras.optimizers.Adam(learning_rate=1e-2)
    ]
    optimizers_and_layers = [(optimizers[0], model.layers[:-4]), (optimizers[1], model.layers[-4:])]
    optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
    for epoch_num in range(c.epoch):
        train(model, train_data_iterator, optimizer)
        model.save('./beizi16m.h5', save_format='h5')

