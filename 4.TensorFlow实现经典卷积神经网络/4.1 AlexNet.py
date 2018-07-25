from datetime import datetime
import math
import time
import tensorflow as tf

'''
    AlexNet卷积神经网络进步：
        （1）用Relu激活函数解决了原先Softmax的梯度弥散问题
        （2）使用Dropout一定程度上避免了过拟合
        （3）池化采用重叠最大池化
        （4）使用LRN层对局部神经元建立竞争机制
        （5）使用CUDA加速深度卷积网络的训练
        （6）数据增强，更多元的数据输入防止CNN陷入过拟合
'''

batch_size = 32
num_batches = 100


# 打印数据状态函数
def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())


def inference(images):
    # 用于存储卷积核数据和偏值数据
    parameters = []

    # 第一层卷积层
    with tf.name_scope('conv1') as scope:
        # 卷积核大小为11*11，通道数为3，数量为64
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64],
                                                 stddev=1e-1, dtype=tf.float32, name='weights'))
        # 卷积步长为4，padding值为SAME
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        # 偏值64个
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32,
                                         name='biaes'))
        # 卷积操作后，加上偏值
        bias = tf.nn.bias_add(conv, biases)

        # 用relu激活函数
        conv1 = tf.nn.relu(bias, name=scope)
        # 打印卷积数据状态
        print_activations(conv1)
        # 将卷积核和偏值数据存入参量集合
        parameters += [kernel, biases]
        # 使用LRN层对卷积数据进行处理
        lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn1')
        # 采用3*3最大重叠池化，步长为2，小于池化窗口本身大小，所以会产生重叠
        pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='VALID', name='pool1')
        # 打印池化数据状态
        print_activations(pool1)

    # 第二层卷积
    with tf.name_scope('conv2') as scope:
        # 卷积核大小为5*5，数量为192，上一层通道数为64
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192],
                                                 stddev=1e-1, dtype=tf.float32, name='weights'))
        # 卷积操作步长为1，padding值为SAME
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        # 偏值192个
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[192], name='biaes'))
        # 将卷积数据与偏值相加
        bias = tf.nn.bias_add(conv, biases)
        # 采用relu激活函数
        conv2 = tf.nn.relu(bias, name=scope)
        # 将本层卷积核和偏值数据存入参量集合
        parameters += [kernel, biases]
        # 打印当前卷积数据状态
        print_activations(conv2)
        # 使用LRN处理卷积数据
        lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn2')
        # 采用3*3最大重叠池化，步长为2，小于池化窗口本身大小，所以会产生重叠
        pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='VALID', name='pool2')
        # 打印池化数据状态
        print_activations(pool2)

    # 第三层卷积
    with tf.name_scope('conv3') as scope:
        # 采用3*3卷积核，共384个，输入通道为192
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                                 stddev=1e-1, dtype=tf.float32, name='weights'))
        # 卷积操作步长为1，padding值为SAME
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        # 偏值384个
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[384], name='biaes'))
        # 偏值与卷积数据相加
        bias = tf.nn.bias_add(conv, biases)
        # 使用relu函数激活
        conv3 = tf.nn.relu(bias, name=scope)
        # 将卷积核和偏值数据存入参量集合
        parameters += [kernel, biases]
        # 打印当前数据状态
        print_activations(conv3)

    # 第四层卷积
    with tf.name_scope('conv4') as scope:
        # 卷积核大小为3*3，256个，输入通道数为384
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                                 stddev=1e-1, dtype=tf.float32, name='weights'))
        # 卷积步长为1，padding值为SAME
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        # 偏值256个
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[256],
                                         name='biaes'))
        # 偏值与卷积数据相加
        bias = tf.nn.bias_add(conv, biases)
        # 激活函数relu
        conv4 = tf.nn.relu(bias, name=scope)
        # 将卷积核数据和偏值数据存入参量集合
        parameters += [kernel, biases]
        # 打印当前卷积数据状态
        print_activations(conv4)

    # 第五层卷积
    with tf.name_scope('conv5') as scope:
        # 卷积核大小为3*3，256个，通道数为256个
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                                 stddev=1e-1, dtype=tf.float32, name='weights'))
        # 卷积步长为1，padding值为SAME
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        # 偏值256个
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[256],
                                         name='biaes'))
        # 偏值加上卷积数据
        bias = tf.nn.bias_add(conv, biases)
        # 用relu函数激活
        conv5 = tf.nn.relu(bias, name=scope)
        # 将卷积核数据和偏值数据存入参量集合
        parameters += [kernel, biases]
        # 打印当前卷积数据状态
        print_activations(conv5)
        # 采用3*3最大重叠池化，步长为2，小于池化窗口本身大小，所以会产生重叠
        pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='VALID', name='pool5')
        # 打印池化数据状态
        print_activations(pool5)

    # 返回最后池化结果和参量记录
    return pool5, parameters


# 每轮计算时间的评估函数[Session对象，需要评测的运算算子，测试名称]
def time_tensorflow_run(session, target, info_string):
    # 预热轮数
    num_steps_burn_in = 10
    # 记录总时间
    total_duration = 0.0
    # 记录总时间平方和用来计算方差
    total_duration_squared = 0.0

    for i in range(num_batches + num_steps_burn_in):
        # 记录开始时间
        start_time = time.time()
        # 运行图
        _ = session.run(target)
        # 计算运算时间
        duration = time.time() - start_time

        # 如果完成10次预热
        if i >= num_steps_burn_in:
            # 每10次迭代打印一次当前时间，迭代次数，计算时间
            if not i % 10:
                print('%s:step %d,duration = %.3f' % (datetime.now(), i - num_steps_burn_in, duration))
            # 记录总计算时间
            total_duration += duration
            # 累加子时间平方
            total_duration_squared += duration * duration
    # 计算每个批所用时间
    mn = total_duration / num_batches
    # 总用时平方和均值减总平均用时
    vr = total_duration_squared / num_batches - mn * mn
    # 开平方
    sd = math.sqrt(vr)
    # 打印耗时评估数据
    print('%s:%s across %d steps,%.3f +/- %.3f sec / batch' %
          (datetime.now(), info_string, num_batches, mn, sd))


# 运行函数
def run_benchmark():
    # 初始化图
    with tf.Graph().as_default():
        # 图片大小为224*224*3
        image_size = 224
        # 随机初始化图片
        images = tf.Variable(tf.random_normal([batch_size,
                                               image_size,
                                               image_size, 3],
                                              stddev=1e-1,
                                              dtype=tf.float32))
        # 获取卷积结果和参量记录
        pool5, parameters = inference(images)
        # 初始化全局变量
        init = tf.global_variables_initializer()
        # 初始化Session对象
        sess = tf.Session()
        # 运行图
        sess.run(init)

        # 评估pool5运算时间
        time_tensorflow_run(sess, pool5, "Forword")
        # L2正则化pool5数据
        objective = tf.nn.l2_loss(pool5)
        # 梯度下降数据
        grad = tf.gradients(objective, parameters)
        # 评估梯度下降运算时间
        time_tensorflow_run(sess, grad, "Forword-backward")


run_benchmark()
