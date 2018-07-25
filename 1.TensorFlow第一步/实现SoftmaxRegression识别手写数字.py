# TensorFlow实现Softmax Regression识别手写数字

# -------------------------------------------------------------------#

# 加载数据集
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 55000个训练集合
print(mnist.train.images.shape, mnist.train.labels.shape)

# 10000个测试集
print(mnist.test.images.shape, mnist.test.labels.shape)

# 5000个验证集
print(mnist.validation.images.shape, mnist.validation.labels.shape)

# -------------------------------------------------------------------#

# 算法实现
import tensorflow as tf

# 注册session
sess = tf.InteractiveSession()

# 创建输入数据地方，(1)=>输入的参数类型，(2)=>tensor的shape
x = tf.placeholder(tf.float32, [None, 784])

# 初始化Weight权重矩阵和bias偏值矩阵,初值为0
# 此处使用Variable初始化，它在模型训练迭代中是持久化的
# 搭建简单神经网络，没有中间层，784的输入与10的输出全连接
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# y = softmax(Wx + b)
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 建立cross-entropy（信息熵）作为loss function来描述模型精确度
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# 训练步骤采用随机梯度下降SGD，每一轮迭代减小loss值
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 全局参数初始化器
tf.global_variables_initializer().run()

# 从训练数据中每次抽取100个进行随机梯度下降
for i in range(5000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})

# 完成训练，测试模型准确率（计算概率最大的数字类别 = label值对应的数字类别 ？）
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# 统计全部样本预测的accuracy
# 转换预测值类型
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 打印准确率
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
