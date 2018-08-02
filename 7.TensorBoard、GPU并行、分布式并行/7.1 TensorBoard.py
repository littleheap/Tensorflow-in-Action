import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 最大步数
max_steps = 1000
# 学习率
learning_rate = 0.001
# dropout比例
dropout = 0.9
# 数据路径
data_dir = '/tmp/tensorflow/mnist/input_data'
# log日志路径
log_dir = './logs/mnist_with_summaries'

# 导入数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

# 输入的命名空间
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

# 输入向量reshape标准化成28*28，类别有10类
with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)


def weight_variable(shape):
    """权重W初始化函数"""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """偏置b初始化函数"""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def variable_summaries(var):
    """tensor数据衡量标准"""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """创建一层神经网络"""
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            # 初始化权重W
            weights = weight_variable([input_dim, output_dim])
            # 汇总W变化
            variable_summaries(weights)
        with tf.name_scope('biases'):
            # 初始化偏置b
            biases = bias_variable([output_dim])
            # 汇总b变化
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            # 获取网络结构
            preactivate = tf.matmul(input_tensor, weights) + biases
            # 直方图记录当前结果情况
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        return activations


# 隐层500个节点
hidden1 = nn_layer(x, 784, 500, 'layer1')

# dropout层
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    # 记录dropout变化
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)

# 定义网络输出层，隐层节点500，输出节点10，激活函数使用identity
y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

# 交叉熵对输出层结果进行softmax处理
with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(diff)
# 汇总交叉熵结果
tf.summary.scalar('cross_entropy', cross_entropy)

# 定义训练步伐
with tf.name_scope('train'):
    # 使用adam优化器
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# 计算准确率
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 汇总准确率
tf.summary.scalar('accuracy', accuracy)

# 汇总所有变量
merged = tf.summary.merge_all()
# 定义文件记录器存放训练和测试的日志数据
train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
test_writer = tf.summary.FileWriter(log_dir + '/test')

# 初始化全局变量
tf.global_variables_initializer().run()


def feed_dict(train):
    """定义训练和测试操作"""
    if train:
        xs, ys = mnist.train.next_batch(100)
        k = dropout
    else:
        xs, ys = mnist.test.images, mnist.test.labels
        k = 1.0
    return {x: xs, y_: ys, keep_prob: k}


# 初始化存储器
saver = tf.train.Saver()

for i in range(max_steps):
    # 每10次记录一下数据汇总和准确率
    if i % 10 == 0:
        summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
        test_writer.add_summary(summary, i)
        print('Accuracy at step %s: %s' % (i, acc))
    else:
        # 每100次记录训练运算时间和内存占用等信息
        if i % 100 == 99:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, _ = sess.run([merged, train_step],
                                  feed_dict=feed_dict(True),
                                  options=run_options,
                                  run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
            train_writer.add_summary(summary, i)
            saver.save(sess, log_dir + "/model.ckpt", i)
            print('Adding run metadata for', i)
        # 正常情况只进行汇总和训练步伐操作
        else:
            summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
            train_writer.add_summary(summary, i)

# 关闭记录器
train_writer.close()
test_writer.close()
