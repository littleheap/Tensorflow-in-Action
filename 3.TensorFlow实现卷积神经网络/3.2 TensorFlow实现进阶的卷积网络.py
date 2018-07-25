from cifar10 import cifar10, cifar10_input
import tensorflow as tf
import numpy as np
import time

'''
    数据集：
        进阶卷积网络使用的是CIFAR-10
        60000张32*32彩色图像
        50000张训练集，10000张测试集
        10个类别，每个类别6000张
        [飞机，汽车，鸟类，猫，小鹿，狗，青蛙，马，轮船，卡车]
    进阶新技巧：
        对weights进行L2正则化
        对图片数据集进行翻转剪切制造更多样本
        每层卷积池化后，添加了LRN层，增强了泛化能力，提高容错率
'''

# 训练次数
max_steps = 3000
# 批大小
batch_size = 128
# 数据下载默认路径
data_dir = './cifar10/cifar-10-batches-bin'

'''
    防止过拟合，通过L2正则化weights使得有效特征权重越大，无效特征权重越小
    即：要想使用某个特征，就要付出loss的代价，特征越有效，loss越小，反之越无效，loss越大
    wl用来控制L2的loss大小，两者相乘，最后统一扔到总losses收集中
'''


def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var


# 下载数据集到指定位置
cifar10.maybe_download_and_extract()

# 提取训练集和测试集
# distort_inputs函数包含了一些数据增强操作
images_train, labels_train = cifar10_input.distorted_inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)
# 测试集数据正常读取，不做增强处理
images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)

# 设置输入数据承接
image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.float32, [batch_size])

# 第一层卷积操作
# 初始化卷积核，第一层卷积核不进行L2正则化，大小为5*5，3通道，64个卷积核，标准差为0.05
weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, wl=0.0)
# 第一次卷积两个步长都为1，padding值为SAME
kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding='SAME')
# 偏值为64个，初始化为0
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
# 激活函数进行非线性化
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
# 池化层，大小为3*3，步长两个方向都为2
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
# LRN操作，会使得响应大的神经元变得相对更大，并抑制反馈小的神经元，增强泛化能力
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

# 第二层卷积操作
weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, wl=0.0)
kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# 搭建卷积结果与输出之间的神经网络

# 扁平化处理卷积数据，行数为批大小，列数为卷积输出的总属性数
reshape = tf.reshape(pool2, [batch_size, -1])
# 获取属性个数，以便建立中间层神经元
dim = reshape.get_shape()[1].value
# 输入层和第一隐藏层连接，第一隐藏层有384个神经元
weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, wl=0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

# 第二隐藏层有192个神经元
weight4 = variable_with_weight_loss(shape=[384.192], stddev=0.04, wl=0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

# 结果层有10个神经元
weight5 = variable_with_weight_loss(shape=[192, 10], stddev=1 / 192.0, wl=0.0)
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.add(tf.matmul(local4, weight5), bias5)


# 定义loss函数
def loss(logits, labels):
    # 将labels转型
    labels = tf.cast(labels, tf.int64)
    # 交叉熵代价函数，计算拟合度
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                   name='cross_entropy_per_example')
    # 计算代价函数平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy, 'cross_entropy')
    # 将代价函数加入到总losses中，与各层wl不为0的weights相加，作为总losses
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


# 计算loss代价函数
loss = loss(logits, label_holder)
# 优化器训练操作
train_op = tf.train.AdagradOptimizer(1e-3).minimize(loss)
# 求输出结果中概率最高的那个分类与标签对应的准确率，类似softmax
top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

# 创建session
sess = tf.InteractiveSession()
# 全局初始化变量
tf.global_variables_initializer().run()

# 启动图片数据增强的线程队列，针对distort_inputs函数
tf.train.start_queue_runners()

# 训练
for step in range(max_steps):
    # 记录开始时间
    start_time = time.time()
    # 获取训练批次
    image_batch, label_batch = sess.run([images_train, labels_train])
    # 进行一次训练，并获取当前loss值
    _, loss_value = sess.run([train_op, loss], feed_dict={image_holder: image_batch, label_holder: label_batch})
    # 计算结束时间
    duration = time.time() - start_time
    if step % 10 == 0:
        # 记录每秒训练样本数量
        examples_per_sec = batch_size / duration
        # 记录每批训练需要的时间
        sec_per_batch = float(duration)

        format_str = ('step %d,loss=%.2f (%.1f examples/sec; %.3f sec/batch)')
        print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))

# 评测模型
num_examples = 10000
import math

num_iter = int(math.ceil(num_examples / batch_size))
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    image_batch, label_batch = sess.run([images_test, labels_test])
    predictions = sess.run([top_k_op], feed_dict={image_holder: image_batch, label_holder: label_batch})
    true_count += np.sum(predictions)
    step += 1

# 计算精准度
precision = true_count / total_sample_count
print('precision @ 1 = %.3f' % precision)
