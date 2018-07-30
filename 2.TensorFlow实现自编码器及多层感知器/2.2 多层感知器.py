import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
    说明：
    多层感知器主要解决三个问题：
    1.过拟合 ---- 解决：Dropout
    2.SDG函数的调参困难问题 ---- 解决：Adagrad，Adam，Adadelta
    3.Sigmoid激活函数梯度弥散 ---- 解决：换用ReLU更好的激活函数
'''

# 读取数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

# 输入数据的属性个数，即输入矩阵的列数
in_units = 784
# 隐藏层神经元个数
hi_units = 300

# 权重W1初始化为正态分布，标准差为0.1，用于输入层->隐藏层
W1 = tf.Variable(tf.truncated_normal([in_units, hi_units], stddev=0.1))
# 输入层->隐藏层的偏值b1
b1 = tf.Variable(tf.zeros([hi_units]))
# 隐藏层->结果层之间的W2权值矩阵
W2 = tf.Variable(tf.zeros([hi_units, 10]))
# 隐藏层->结果层之间的b2偏值
b2 = tf.Variable(tf.zeros([10]))

# 定义输入层矩阵
x = tf.placeholder(tf.float32, [None, in_units])
# 定义Dropout参数
keep_prob = tf.placeholder(tf.float32)

# 计算隐藏层输出
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
# 对隐藏层进行Dropout处理并得到输出
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)

# 将隐藏层的Dropout输出作为结果层的输入，计算预测值
y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)

# 定义正确结果labels矩阵
y_ = tf.placeholder(tf.float32, [None, 10])

# 计算预测值和实际值之间的误差
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# 使用Adagrad优化器对误差进行优化
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

# 声明全局变量
tf.global_variables_initializer().run()

for i in range(3000):
    # 每次取100个作为一束训练集
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # 训练时将Dropout设置为75%节点运作
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})

# 计算预测值和实际值之间的差异
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# 通过差异计算准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 输出测试集在本模型的准确率，此时将Dropout关闭，即100%节点运作
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
