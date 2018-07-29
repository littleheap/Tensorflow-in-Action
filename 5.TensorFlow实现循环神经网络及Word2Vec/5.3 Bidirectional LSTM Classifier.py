import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# 导入MINST数据
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# 超参数初始化
learning_rate = 0.01  # 学习率
max_samples = 400000  # 最大训练样本数40万
batch_size = 128  # 批次尺寸
display_step = 10  # 显示进度效果间隔次数

# 网络参数初始化
n_input = 28  # MNIST图像输入尺寸为28*28，一次读取一行28个像素
n_steps = 28  # LSTM展开步数，与图像高度一致
n_hidden = 256  # 隐层节点数
n_classes = 10  # 数据集类别个数

# 创建X和Y的预存器
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# 设置Softmax层的W和b，直接随机初始化
weights = {
    # 双向RNN网络W参数量乘以2
    'out': tf.Variable(tf.random_normal([2 * n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Bidirectional LSTM网络的生成函数
def BiRNN(x, weights, biases):
    # 将(batch_size, n_steps, n_input)标准化为元素是(batch_size, n_input)长度为为n_steps的列表
    x = tf.transpose(x, [1, 0, 2])
    # 重新(n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # 切割成n_steps份(batch_size, n_input)
    x = tf.split(x, n_steps)

    # 创建前馈单元
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # 创建反馈单元
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # 将正向和反向的单元网络传入static_bidirectional_rnn接口，生成双向RNN网络
    outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                            dtype=tf.float32)
    # 最后对双向RNN网络的输出结果做一个矩阵乘法加上偏置
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


# 用上面函数生成双向RNN网络
pred = BiRNN(x, weights, biases)

# 定义损失函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# 定义优化器
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 评估模型
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# 计算准确率
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 全局初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < max_samples:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # 规范化输入数据
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # 计算准确率
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # 计算损失函数值
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # 测试集
    test_len = 10000
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
