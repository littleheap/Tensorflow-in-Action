import numpy as np
import random
import tensorflow as tf
import os
import GridWorld
import matplotlib.pyplot as plt

env = GridWorld.gameEnv(size=5)


# Deep Q-Network
class Qnetwork():
    def __init__(self, h_size):
        # 输入时被扁平化的长度为84*84*3=21168的向量
        self.scalarInput = tf.placeholder(shape=[None, 21168], dtype=tf.float32)
        # 将输入恢复成多个84*84*3尺寸的图片
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 84, 84, 3])
        # 第1个卷积层：卷积核尺寸8*8，步长为4*4，输出通道数为32，padding模式VALID，输出维度20*20*32
        self.conv1 = tf.contrib.layers.convolution2d(inputs=self.imageIn, num_outputs=32, kernel_size=[8, 8],
                                                     stride=[4, 4], padding='VALID',
                                                     biases_initializer=None)
        # 第2个卷积层：卷积核尺寸4*4，步长为2*2，输出通道数为64，padding模式VALID，输出维度9*9*64
        self.conv2 = tf.contrib.layers.convolution2d(inputs=self.conv1, num_outputs=64, kernel_size=[4, 4],
                                                     stride=[2, 2], padding='VALID',
                                                     biases_initializer=None)
        # 第3个卷积层：卷积核尺寸3*3，步长为1*1，输出通道数为64，padding模式VALID，输出维度7*7*64
        self.conv3 = tf.contrib.layers.convolution2d(inputs=self.conv2, num_outputs=64, kernel_size=[3, 3],
                                                     stride=[1, 1], padding='VALID',
                                                     biases_initializer=None)
        # 第4个卷积层：卷积核尺寸7*7，步长为1*1，输出通道数为512，padding模式VALID，输出维度1*1*512
        self.conv4 = tf.contrib.layers.convolution2d(inputs=self.conv3, num_outputs=512, kernel_size=[7, 7],
                                                     stride=[1, 1], padding='VALID',
                                                     biases_initializer=None)
        # 将卷积层输出平均分拆成两段，AC和VC，分别对应Action价值和环境本身价值
        self.streamAC, self.streamVC = tf.split(self.conv4, 2, 3)  # 2段，第3维度
        # 扁平化处理AC
        self.streamA = tf.contrib.layers.flatten(self.streamAC)
        # 扁平化处理VC
        self.streamV = tf.contrib.layers.flatten(self.streamVC)
        # 创建线性全连接层初始化权重W
        self.AW = tf.Variable(tf.random_normal([h_size // 2, env.actions]))
        self.VW = tf.Variable(tf.random_normal([h_size // 2, 1]))
        # 获得Advantage结果
        self.Advantage = tf.matmul(self.streamA, self.AW)
        # 获得Value结果
        self.Value = tf.matmul(self.streamV, self.VW)
        # Q值由Advantage和Value复合而成
        self.Qout = self.Value + tf.subtract(self.Advantage,
                                             tf.reduce_mean(self.Advantage, reduction_indices=1, keep_dims=True))
        # 计算Q最大的Action
        self.predict = tf.argmax(self.Qout, 1)
        # 定义目标Q值输入
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        # 定义动作action输入
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        # actions转化为onehot编码模式
        self.actions_onehot = tf.one_hot(self.actions, env.actions, dtype=tf.float32)
        # 将Qout和actions_onehot相乘得到Q值
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), reduction_indices=1)
        # 计算targetQ和Q的均方误差
        self.td_error = tf.square(self.targetQ - self.Q)
        # 定义loss损失函数
        self.loss = tf.reduce_mean(self.td_error)
        # 使用Adam优化器
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        # 最小化loss函数
        self.updateModel = self.trainer.minimize(self.loss)


# Experience策略
class experience_buffer():
    # 存储样本最大容量buffer size
    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size

    # 存储样本越界的话，清空早期一些样本
    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    # 随机抽样一些样本
    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])


# 扁平化数据84*84*3函数
def processState(states):
    return np.reshape(states, [21168])


# 更新DQN模型参数方法：全部参数、主DQN学习率
def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    # 取前一半参数即主DQN模型参数
    for idx, var in enumerate(tfVars[0:total_vars // 2]):
        # 缓慢更新参数
        op_holder.append(tfVars[idx + total_vars // 2].assign(
            (var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars // 2].value())))
    return op_holder


# 执行更新模型参数操作
def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)


batch_size = 32  # 从experience buffer中获取样本批次尺寸
update_freq = 4  # 每隔4步更新一次模型参数
y = .99  # Q值衰减系数
startE = 1  # 起始随机Action概率，训练需要随机搜索，预测时不需要
endE = 0.1  # 最终执行Action概率
anneling_steps = 10000.  # 初始随机概率到最终随机概率下降的步数
num_episodes = 10000  # 实验次数
pre_train_steps = 10000  # 使用DQN选择Action前进行多少步随机测试
max_epLength = 50  # 每个episode执行多少次Action
load_model = False  # 是否读取之前训练的模型
path = "./dqn"  # 模型存储路径
h_size = 512  # DQN全连接层隐含节点数
tau = 0.001  # target DQN向主DQN学习的学习率

tf.reset_default_graph()
# 初始化主DQN
mainQN = Qnetwork(h_size)
# 初始化辅助targetDQN
targetQN = Qnetwork(h_size)

# 初始化全局变量
init = tf.global_variables_initializer()

# 获得所有可训练参数
trainables = tf.trainable_variables()

# 创建更新targetDQN参数的操作
targetOps = updateTargetGraph(trainables, tau)

# 初始化experience buffer
myBuffer = experience_buffer()

# 设置当前学习率
e = startE
# 计算每一步衰减值
stepDrop = (startE - endE) / anneling_steps

# 初始化存储episode的reward列表
rList = []
# 初始化总步数
total_steps = 0

# 创建模型存储器并检验保存路径
saver = tf.train.Saver()
if not os.path.exists(path):
    os.makedirs(path)


with tf.Session() as sess:
    # 如果已有模型
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    # 初始化全部参数
    sess.run(init)
    # 执行更新参数操作
    updateTarget(targetOps, sess)
    # 创建实验循环
    for i in range(num_episodes + 1):
        episodeBuffer = experience_buffer()
        # 重置环境
        s = env.reset()
        # 获取环境信息并将其扁平化
        s = processState(s)
        # done标记
        d = False
        # episode内部的总reward
        rAll = 0
        # episode内部的步数
        j = 0
        # 创建内循环，迭代每一次执行的Action
        while j < max_epLength:
            j += 1
            # 步数小于pre_train_steps时，强制使用随机Action
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                a = np.random.randint(0, 4)
            # 达到pre_train_steps后，保留一个较小概率随机选择Action，若不选择随机Action，将当前状态s传入DQN，预测Action
            else:
                a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [s]})[0]
            # 执行一步Action，和获取接下来状态s1，reward和done标记
            s1, r, d = env.step(a)
            # 扁平化处理s1
            s1 = processState(s1)
            # 总步数+1
            total_steps += 1
            # 将数据和结果传入episodeBuffer存储
            episodeBuffer.add(
                np.reshape(np.array([s, a, r, s1, d]), [1, 5]))  # Save the experience to our episode buffer.
            # 如果总步数超过pre_train_steps
            if total_steps > pre_train_steps:
                # 持续降低随机选择Action概率直到endE
                if e > endE:
                    e -= stepDrop
                # 步数达到update_freq整倍数时，进行一次训练，更新一次参数
                if total_steps % (update_freq) == 0:
                    # 从myBuffer获取样本
                    trainBatch = myBuffer.sample(batch_size)
                    # 训练样本中第3列信息即下一个状态s1传入主DQN，得到主模型选择的Action
                    A = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 3])})
                    # 再将s1传入辅助的targetDQN，得到s1状态下所有Action的Q值
                    Q = sess.run(targetQN.Qout, feed_dict={targetQN.scalarInput: np.vstack(trainBatch[:, 3])})
                    # 将主DQN输出的Action和targetDQN输出的Q值，得到doubleQ
                    doubleQ = Q[range(batch_size), A]
                    # 使用训练样本第二列reward值+doubleQ*衰减系数y，获得targetQ
                    targetQ = trainBatch[:, 2] + y * doubleQ
                    # 传入当前状态s，学习targetQ和这一步的Action
                    _ = sess.run(mainQN.updateModel,
                                 feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 0]), mainQN.targetQ: targetQ,
                                            mainQN.actions: trainBatch[:, 1]})
                    # 更新一次主DQN参数
                    updateTarget(targetOps, sess)
            # 累计当前获取的reward
            rAll += r
            # 更新当前状态为下一步实验做准备
            s = s1
            # 如果done为True，则终止实验
            if d == True:
                break

        # episode内部的episodeBuffer添加到myBuffer中
        myBuffer.add(episodeBuffer.buffer)
        # episode中reward添加到rList中
        rList.append(rAll)
        # 每25次展示一次平均reward
        if i > 0 and i % 25 == 0:
            print('episode', i, ', average reward of last 25 episode', np.mean(rList[-25:]))
        # 每1000次保存当前模型
        if i > 0 and i % 1000 == 0:
            saver.save(sess, path + '/model-' + str(i) + '.cptk')
            print("Saved Model")
    saver.save(sess, path + '/model-' + str(i) + '.cptk')

# 计算每100个episode平均reward
rMat = np.resize(np.array(rList), [len(rList) // 100, 100])
# 使用plot绘制reward变化趋势
rMean = np.average(rMat, 1)
plt.plot(rMean)
