import numpy as np
import tensorflow as tf

import gym

# 创建CartPole-v0环境
env = gym.make('CartPole-v0')
# 初始化环境
env.reset()

# 随机试验次数
random_episodes = 0
# 奖励总和
reward_sum = 0
# 进行10次随机试验
while random_episodes < 10:
    # 渲染图像
    env.render()
    # 执行step返回三个数据，done为True则本次实验结束
    observation, reward, done, _ = env.step(np.random.randint(0, 2))
    # 累加奖励
    reward_sum += reward
    # 当前实验结束，打印当前奖励
    if done:
        random_episodes += 1
        print("Reward for this episode was:", reward_sum)
        reward_sum = 0
        env.reset()

'''
    Reward for this episode was: 13.0
    Reward for this episode was: 22.0
    Reward for this episode was: 30.0
    Reward for this episode was: 11.0
    Reward for this episode was: 15.0
    Reward for this episode was: 10.0
    Reward for this episode was: 19.0
    Reward for this episode was: 17.0
    Reward for this episode was: 25.0
    Reward for this episode was: 21.0
'''

# 隐含层超参数设定
H = 50  # 隐含层节点数
batch_size = 25  # 批尺寸
learning_rate = 1e-1  # 学习率
gamma = 0.99  # reward/discount
D = 4  # 环境信息维度

tf.reset_default_graph()

# 策略网络基本内结构：游戏有两个Action，左力和右力，通过输出的概率值决定
# observations输入信息，维度为4
observations = tf.placeholder(tf.float32, [None, D], name="input_x")
# 初始化隐含层W1，维度为[D,H]
W1 = tf.get_variable("W1", shape=[D, H],
                     initializer=tf.contrib.layers.xavier_initializer())
# relu激活函数处理得到隐含层输出layer1
layer1 = tf.nn.relu(tf.matmul(observations, W1))
# 初始化Sigmoid输出层W2
W2 = tf.get_variable("W2", shape=[H, 1],
                     initializer=tf.contrib.layers.xavier_initializer())
# 计算输出值
score = tf.matmul(layer1, W2)
# 得到最后输出概率
probability = tf.nn.sigmoid(score)

# 获取策略网络中全部可训练的参数
tvars = tf.trainable_variables()
# 设置虚拟label的placeholder
input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
# 每个Action的潜在价值的placeholder
advantages = tf.placeholder(tf.float32, name="reward_signal")
# 当前Action对应的概率的对数，用来与advantages相乘，优化辅助advantages对应Action的概率
loglik = tf.log(input_y * (input_y - probability) + (1 - input_y) * (input_y + probability))
# 计算损失函数loss
loss = -tf.reduce_mean(loglik * advantages)
# 计算损失函数梯度
newGrads = tf.gradients(loss, tvars)

# 优化器选择Adam
adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
# 两层网络参数的梯度placeholder
W1Grad = tf.placeholder(tf.float32, name="batch_grad1")
W2Grad = tf.placeholder(tf.float32, name="batch_grad2")
# 整合两层梯度结果
batchGrad = [W1Grad, W2Grad]
# 累计到一定样本梯度，执行updateGrads更新参数
updateGrads = adam.apply_gradients(zip(batchGrad, tvars))


def discount_rewards(r):
    """估算每一个Action潜在的价值discount_r"""
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


# 定义参数：xs-环境信息observation列表、ys为自定义label列表、drs记录每一个Action的reward
xs, ys, drs = [], [], []
# 累计reward
reward_sum = 0
# 当前训练次数
episode_number = 1
# 总实验次数
total_episodes = 10000

with tf.Session() as sess:
    # rendering标志关闭
    rendering = False
    # 初始化全局变量
    init = tf.global_variables_initializer()
    sess.run(init)
    # 初始化环境并获取初始化状态
    observation = env.reset()
    # 创建存储参数梯度的缓冲器
    gradBuffer = sess.run(tvars)
    # 初始化为0
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0
    while episode_number <= total_episodes:
        # 当前batch平均reward达到100说明良好，对实验环境进行展示
        if reward_sum / batch_size > 100 or rendering == True:
            env.render()
            rendering = True
        # 规范化网络输入格式
        x = np.reshape(observation, [1, D])
        # 执行sess.run输出概率，即Action取值为1的概率
        tfprob = sess.run(probability, feed_dict={observations: x})
        # 0-1之间随机取样，小于tfprob就对应action 1否则选择action 0
        action = 1 if np.random.uniform() < tfprob else 0

        # 输入的环境信息添加到xs列表中
        xs.append(x)
        # 自定义虚拟label与action取反
        y = 1 if action == 0 else 0
        # 将其添加到ys列表中
        ys.append(y)

        # 执行一次action，获取返回的四个数据
        observation, reward, done, info = env.step(action)
        # 累加当前总reward
        reward_sum += reward
        # 将reward添加到drs列表中
        drs.append(reward)

        # 一次实验结束后
        if done:
            # 当前实验次数加1
            episode_number += 1
            # 将几个列表元素纵向堆叠
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            # 清空三个列表
            xs, ys, drs = [], [], []

            # 计算每一步潜在价值
            discounted_epr = discount_rewards(epr)
            # 价值进行标准化
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            # 得到的三个数据输入神经网络，返回求解梯度
            tGrad = sess.run(newGrads, feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})
            # 将获得的梯度累加到gradBuffer中
            for ix, grad in enumerate(tGrad):
                gradBuffer[ix] += grad

            # 当实验次数达到batch size整倍数，累积的梯度更新一次参数
            if episode_number % batch_size == 0:
                sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0], W2Grad: gradBuffer[1]})
                # 清空gradBuffer
                for ix, grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0

                # 显示当前实验次数和batch平均reward
                print('Average reward for episode %d : %f.' % (episode_number, reward_sum / batch_size))
                # 如果reward大于200，策略网络就完成任务
                if reward_sum / batch_size > 200:
                    print("Task solved in", episode_number, 'episodes!')
                    break
                reward_sum = 0

            observation = env.reset()
