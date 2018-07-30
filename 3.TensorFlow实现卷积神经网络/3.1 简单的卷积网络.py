import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 读取数字识别数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 开启TensorFlow
sess = tf.InteractiveSession()


# 卷积核权值初始化函数
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 生成一个截断的正态分布，标准差为0.1
    return tf.Variable(initial)


# 每个卷积核对应的偏置初始化函数
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积函数
def conv2d(x, W):
    '''
    x是一个四维的tensor [batch, in_height, in_width, in_channels] 1.批次 2.图片高 3.图片宽 4.通道数：黑白为1，彩色为3
    W是一个滤波器/卷积核 [filter_height, filter_width, in_channels, out_channels] 1.滤波器高 2.滤波器宽 3.输入通道数 4.输出通道数
    固定 strides[0] = strides[3] = 1， strides[1]代表x方向的步长，strides[2]代表y方向的步长
    padding= 'SAME' / 'VALID' ; SAME在外围适当补0 , VALID不填补0
    '''
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 池化函数
def max_pool_2x2(x):
    '''
    x是一个四维的tensor [batch, in_height, in_width, in_channels] 1.批次 2.图片高 3.图片宽 4.通道数：黑白为1，彩色为3
    ksize是窗口大小 [1,x,y,1] , 固定ksize[0] = ksize[3] = 1 ,  ksize[1]代表x方向的大小 , ksize[2]代表y方向的大小
    固定 strides[0] = strides[3] = 1， strides[1]代表x方向的步长，strides[2]代表y方向的步长
    padding= 'SAME' / 'VALID' ; SAME在外围适当补0 , VALID不填补0
    '''
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# 改变x的格式转为2维的向量 [batch, in_height, in_width, in_channels] 1.批次 2.二维高 3.二维宽 4.通道数：黑白为1，彩色为3
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 第一层卷积
# 5*5的采样窗口，32个卷积核从1个平面抽取特征
W_conv1 = weight_variable([5, 5, 1, 32])
# 每一个卷积核一个偏置值
b_conv1 = bias_variable([32])
# 把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# 最后进行池化
h_pool1 = max_pool_2x2(h_conv1)  # 进行max-pooling

# 第二层卷积
# 5*5的采样窗口，64个卷积核从32个平面抽取特征
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# 进行max-pooling
h_pool2 = max_pool_2x2(h_conv2)

# 对两层卷积后的结果与预测值之间进行神经网络搭建

# 第一层
# 输入层有7*7*64个列的属性，全连接层有1024个隐藏神经元
W_fc1 = weight_variable([7 * 7 * 64, 1024])
# 1024个隐藏层节点偏值
b_fc1 = bias_variable([1024])

# 把池化层2的输出扁平化为1维，-1代表任意值
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
# 求第一个全连接层的输出，并激活
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# Dropout处理，keep_prob用来表示处于激活状态的神经元比例
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 第二层
# 输入为1024个隐藏层神经元，输出层为10个数字可能结果
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
# 将中间层与输出层10个结果全连接
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 计算代价函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
# 使用AdamOptimizer进行优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 计算准确率
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 初始化全局变量
tf.global_variables_initializer().run()

for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
