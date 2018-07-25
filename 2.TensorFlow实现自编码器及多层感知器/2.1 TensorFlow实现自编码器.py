import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
    说明：
    本段代码运行所展示出的自编码器效果，主要是将提取高维特征（输入->隐藏层）和重构（隐藏层->输出层）分开处理
    前提条件是将输入的标准数据用高斯噪声加以影响，使得自编码器展现出提取高维特征，屏蔽噪声信号的效果
    总体网络结构是三层：输入层，隐藏层，输出层
    输入层：由所读取的输入数据原型所决定，额外加上高斯噪声
    隐藏层：由操作者自定义神经元数目，一般少于输入层
    输出层：模型与输入数据原型完全一致，降噪
    最终通过计算loss值，来反应重构后的预测值与数据原值之间的差距大小，进而反应降噪效果
'''


# Xavier初始化器，让初始化的参数处于一个不低不高的水平
def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


class AdditiveGaussianNoiseAutoencoder(object):
    # 构造函数
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(),
                 scale=0.1):
        # 输入的特征数
        self.n_input = n_input
        # 隐藏层节点数
        self.n_hidden = n_hidden
        # 隐藏层激活函数，默认softplus
        self.transfer = transfer_function
        # 高斯噪声系数，默认0.1
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        # 神经网络初始化参数
        network_weights = self._initialize_weights()
        # 获取初始化的神经网络
        self.weights = network_weights
        # 初始化输入的模型，列数为n_input个
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        # 计算隐藏层，输入数据首先融入噪声，然后与权值w1相乘，最后加上偏值b1
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)), self.weights['w1']),
                                           self.weights['b1']))
        # 计算预测输出值，将隐藏层的结果与权值矩阵w2相乘，在于偏值b2相加作为得出输出的结果
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])
        # 计算损失函数，计算预测值与数据本身之间的误差
        self.cost = 0.5 * tf.reduce_mean(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        # 选择默认Adam优化器对损失函数进行优化
        self.optimizer = optimizer.minimize(self.cost)
        # 初始化全局变量
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    # 权值初始化函数
    def _initialize_weights(self):
        # 声明权重值字典，一共四个key，w1，b1，w2，b2
        all_weights = dict()
        # w1权重矩阵介于输入层和隐藏层之间，所以行数为n_input，列数为n_hidden
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        # b1偏值作用于输入层和隐藏层之间，所以尺寸为n_hidden大小
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        # w2权重矩阵介于隐藏层和预测结果层之间，所以行数为n_hidden，列数为n_input，因为输出和输入的尺寸一样
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        # b2偏值作用于隐藏层和预测结果层之间，所以尺寸为n_input大小，因为输出和输入的尺寸一样
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    # 计算损失函数和运行优化器
    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X, self.scale: self.training_scale})
        return cost

    # 单独计算损失函数
    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x: X, self.scale: self.training_scale})

    # 单独计算隐含层输出结果，用来提取数据的高阶特征，是整体拆分的前半部分
    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X, self.scale: self.training_scale})

    # 将隐含层的高阶特征复原为原始数据，是整体拆分的后半部分
    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    # 包括transform和generate两部分的重构
    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X, self.scale: self.training_scale})

    # 获取权重w1
    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    # 获取偏值b1
    def getBiases(self):
        return self.sess.run(self.weights['b1'])


# 读取数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# 数据标准化处理函数
def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


# 最大限度不重复地获取数据
def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]


# 标准化训练集和测试集
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
# 总训练样本数
n_samples = int(mnist.train.num_examples)
# 训练轮数20
training_epochs = 20
# 每一次训练束大小128
batch_size = 128
# 设置每一轮显示一次loss值
display_step = 1

autoencoder = AdditiveGaussianNoiseAutoencoder(n_input=784, n_hidden=200, transfer_function=tf.nn.softplus,
                                               optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                               scale=0.01)

for epoch in range(training_epochs):
    avg_cost = 0
    # 计算一共有多少束数据集
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        # 获取预设大小的数据集
        batch_xs = get_random_block_from_data(X_train, batch_size)
        cost = autoencoder.partial_fit(batch_xs)
        avg_cost += cost / n_samples * batch_size
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

print("Total cost: " + str(autoencoder.calc_total_cost(X_test)))
