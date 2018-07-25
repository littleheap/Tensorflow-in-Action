from datetime import datetime
import math
import time
import tensorflow as tf

batch_size = 32
num_batches = 100


def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())


# 卷积层创建函数，并把本层参数存入参数列表
# [输入张量，当前层名称，卷积核的高，卷积核的宽，卷积核数量(输出通道数)，步长的高，步长的宽，参数列表]
def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    # 获取输入张量的通道数
    n_in = input_op.get_shape()[-1].value

    # 设置当前层命名空间
    with tf.name_scope(name) as scope:
        # 初始化卷积核
        kernel = tf.get_variable(scope + "w",
                                 shape=[kh, kw, n_in, n_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        # 进行卷积操作
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1),
                            padding='SAME')
        # 初始化偏值，为0
        bias_init_val = tf.constant(0.0, dtype=tf.float32, shape=[n_out])
        # 偏值转型为变量参数
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        # 卷积结果与偏值相加
        z = tf.nn.bias_add(conv, biases)
        # 激活函数
        activation = tf.nn.relu(z, name=scope)
        # 记录当前层卷积核，偏值数据
        p += [kernel, biases]
        # 返回卷积层输出
        return activation


# 全连接层创建函数
# [输入张量，当前层的命名，输出通道数，参数列表]
def fc_op(input_op, name, n_out, p):
    # 获取输入张量的通道数
    n_in = input_op.get_shape()[-1].value

    # 定义命名空间
    with tf.name_scope(name) as scope:
        # 初始化权值矩阵
        kernel = tf.get_variable(scope + 'w',
                                 shape=[n_in, n_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        # 初始化偏值，为0.1
        biases = tf.Variable(tf.constant(0.1, dtype=tf.float32,
                                         shape=[n_out], name='b'))
        # 激活函数
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
        # 记录参数
        p += [kernel, biases]
        # 返回输出结果
        return activation


# 最大池化层创建函数
# [输入张量，命名空间，池化框高，池化框宽，步长高，步长宽]
def mpool_op(input_op, name, kh, kw, dh, dw):
    # 返回最大池化结果
    return tf.nn.max_pool(input_op,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='SAME',
                          name=name)


# VGG-16网络创建
# [输入张量，Dropout激活比例]
def inference_op(input_op, keep_prob):
    # 初始化参数记录列表
    p = []
    # 第一层第一次卷积
    conv1_1 = conv_op(input_op, name="conv1_1",
                      kh=3, kw=3,
                      n_out=64, dh=1, dw=1, p=p)
    # 第一层第二次卷积
    conv1_2 = conv_op(conv1_1, name="conv1_2",
                      kh=3, kw=3,
                      n_out=64, dh=1, dw=1, p=p)
    # 第一层池化
    pool1 = mpool_op(conv1_2, name="pool1", kh=2, kw=2, dh=2, dw=2)

    # 第二层第一次卷积
    conv2_1 = conv_op(pool1, name="conv2_1",
                      kh=3, kw=3,
                      n_out=128, dh=1, dw=1, p=p)
    # 第二层第二次卷积
    conv2_2 = conv_op(conv2_1, name="conv2_1",
                      kh=3, kw=3,
                      n_out=128, dh=1, dw=1, p=p)
    # 第二层池化
    pool2 = mpool_op(conv2_2, name="pool2", kh=2, kw=2, dh=2, dw=2)

    # 第三层第一次卷积
    conv3_1 = conv_op(pool2, name="conv3_1",
                      kh=3, kw=3,
                      n_out=256, dh=1, dw=1, p=p)
    # 第三层第二次卷积
    conv3_2 = conv_op(conv3_1, name="conv3_2",
                      kh=3, kw=3,
                      n_out=256, dh=1, dw=1, p=p)
    # 第三层第三次卷积
    conv3_3 = conv_op(conv3_2, name="conv3_3",
                      kh=3, kw=3,
                      n_out=256, dh=1, dw=1, p=p)
    # 第三层池化
    pool3 = mpool_op(conv3_3, name="pool3", kh=2, kw=2, dh=2, dw=2)

    # 第四层第一次卷积
    conv4_1 = conv_op(pool3, name="conv4_1",
                      kh=3, kw=3,
                      n_out=512, dh=1, dw=1, p=p)
    # 第四层第二次卷积
    conv4_2 = conv_op(conv4_1, name="conv4_2",
                      kh=3, kw=3,
                      n_out=512, dh=1, dw=1, p=p)
    # 第四层第三次卷积
    conv4_3 = conv_op(conv4_2, name="conv4_3",
                      kh=3, kw=3,
                      n_out=512, dh=1, dw=1, p=p)
    # 第四层池化
    pool4 = mpool_op(conv4_3, name="pool4", kh=2, kw=2, dh=2, dw=2)

    # 第五层第一次卷积
    conv5_1 = conv_op(pool4, name="conv5_1",
                      kh=3, kw=3,
                      n_out=512, dh=1, dw=1, p=p)
    # 第五层第二次卷积
    conv5_2 = conv_op(conv5_1, name="conv5_2",
                      kh=3, kw=3,
                      n_out=512, dh=1, dw=1, p=p)
    # 第五层第三次卷积
    conv5_3 = conv_op(conv5_2, name="con5_3",
                      kh=3, kw=3,
                      n_out=512, dh=1, dw=1, p=p)
    # 第五层池化
    pool5 = mpool_op(conv5_3, name="pool4", kh=2, kw=2, dh=2, dw=2)

    # 获取VGG-16网络处理后的数据尺寸
    shp = pool5.get_shape()
    # 计算VGG-16网络输出结果一维拉伸后的尺寸
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    # 将VGG-16网络输出张量结果一维拉伸
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name="resh1")

    # 第一隐含层，4096个神经元，与输入全连接
    fc6 = fc_op(resh1, name="fc6", n_out=4096, p=p)
    # Dropout处理
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop")

    # 第二隐含层，4096个神经元，与上一层全连接
    fc7 = fc_op(fc6_drop, name="fc7", n_out=4096, p=p)
    # Dropout处理
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")

    # 输出层，1000个神经元，与上一层全连接
    fc8 = fc_op(fc7_drop, name="fc8", n_out=1000, p=p)

    # softmax处理输出层
    softmax = tf.nn.softmax(fc8)

    # 取最大概率预测
    predictions = tf.argmax(softmax, 1)
    # 返回预测，softmax处理，输出层数据，数据记录列表
    return predictions, softmax, fc8, p


# 网络性能评测函数
def time_tensorflow_run(session, target, feed, info_string):
    # 预热步数
    num_steps_burn_in = 10
    # 总持续时间
    total_duration = 0.0
    # 总持续时间平方和
    total_duration_squared = 0.0

    # 总迭代循环
    for i in range(num_batches + num_steps_burn_in):
        # 记录开始时间
        start_time = time.time()
        # 运行图
        _ = session.run(target, feed_dict=feed)
        # 记录持续时间
        duration = time.time() - start_time

        # 如果完成预热
        if i >= num_steps_burn_in:
            # 每十次迭代，打印一次目前情况
            if not i % 10:
                print('%s:step %d,duration = %.3f' % (datetime.now(), i - num_steps_burn_in, duration))

            # 累加计算总持续时间
            total_duration += duration
            # 累加计算总持续时间平方和
            total_duration_squared += duration * duration
    # 计算每批平均时间
    mn = total_duration / num_batches
    # 计算方差
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    # 打印数据
    print('%s:%s across %d steps,%.3f +/- %.3f sec / batch' %
          (datetime.now(), info_string, num_batches, mn, sd))


# 运行测试函数
def run_benchmark():
    # 定义图
    with tf.Graph().as_default():
        # 定义图尺寸
        image_size = 224
        # 随机生成图
        images = tf.Variable(tf.random_normal([batch_size,
                                               image_size,
                                               image_size, 3],
                                              stddev=1e-1,
                                              dtype=tf.float32))
        # 设置Dropout激活比例
        keep_prob = tf.placeholder(dtype=tf.float32)
        # 获取预测，softma数据，输出层数据，数据记录列表
        predictions, softmax, fc8, p = inference_op(images, keep_prob)

        # 全局变量初始化
        init = tf.global_variables_initializer()
        # 初始化session
        sess = tf.Session()
        # 初始化运行图
        sess.run(init)

        # 前馈性能测试
        time_tensorflow_run(sess, predictions, {keep_prob: 1.0}, "Forward")
        # 计算输出层的l2 loss值
        objective = tf.nn.l2_loss(fc8)
        # 求相对于该loss所有模型的参数梯度
        grad = tf.gradients(objective, p)
        # 反馈性能测试
        time_tensorflow_run(sess, grad, {keep_prob: 0.1}, "Forward-backward")


run_benchmark()
