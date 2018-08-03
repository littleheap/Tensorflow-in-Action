import os.path
import re
import time
import numpy as np
import tensorflow as tf
import cifar10

batch_size = 128
max_steps = 1000000
num_gpus = 4


# 损失函数
def tower_loss(scope):
    # 获取数据增强后的images和labels
    images, labels = cifar10.distorted_inputs()

    # 生成卷积网络，GPU共享参数模型
    logits = cifar10.inference(images)

    # 计算损失函数
    _ = cifar10.loss(logits, labels)

    # 获取当前GPU上的loss
    losses = tf.get_collection('losses', scope)

    # 损失叠加计算总损失
    total_loss = tf.add_n(losses, name='total_loss')

    return total_loss


# 将不同的GPU计算出的梯度进行合成
def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            # 添加一个冗余的维度0
            expanded_g = tf.expand_dims(g, 0)
            # 把这些梯度放到列表grad中
            grads.append(expanded_g)
        # 将他们在维度0上合并
        grad = tf.concat(grads, 0)
        # 求均值
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


# 训练函数
def train():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # 记录全局训练步数
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)
        # 计算一个epoch对应的batch数
        num_batches_per_epoch = (cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size)
        # 计算学习率衰减需要的步数
        decay_steps = int(num_batches_per_epoch * cifar10.NUM_EPOCHS_PER_DECAY)
        # 创建随训练步数衰减的学习率
        lr = tf.train.exponential_decay(cifar10.INITIAL_LEARNING_RATE,
                                        global_step,
                                        decay_steps,
                                        cifar10.LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)
        # 使用梯度下降优化器
        opt = tf.train.GradientDescentOptimizer(lr)
        # 定义存储各GPU计算结果的列表
        tower_grads = []
        # 循环GPU个数
        for i in range(num_gpus):
            # 限定使用第i个GPU
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % (cifar10.TOWER_NAME, i)) as scope:
                    # 对当前GPU计算损失
                    loss = tower_loss(scope)
                    # 共享一致参数
                    tf.get_variable_scope().reuse_variables()
                    # 计算单个GPU梯度
                    grads = opt.compute_gradients(loss)
                    # 将计算的梯度添加到列表中
                    tower_grads.append(grads)
        # 计算平均梯度
        grads = average_gradients(tower_grads)
        # 更新模型参数
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        # 创建模型保存器
        saver = tf.train.Saver(tf.all_variables())
        # 初始化全局变量
        init = tf.global_variables_initializer()
        # 设置session参数True
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(init)
        # 准备大量数据增强训练样本
        tf.train.start_queue_runners(sess=sess)
        # 最大迭代次数
        for step in range(max_steps):
            start_time = time.time()
            # 每一步执行更新参数并计算loss损失
            _, loss_value = sess.run([apply_gradient_op, loss])
            duration = time.time() - start_time
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            # 每10步展示一次当前batch的loss、每秒驯良的样本数和每个batch训练所需花费的时间
            if step % 10 == 0:
                num_examples_per_step = batch_size * num_gpus
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / num_gpus
                format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))
            # 每1000步保存整个模型
            if step % 1000 == 0 or (step + 1) == max_steps:
                saver.save(sess, '/tmp/cifar10_train/model.ckpt', global_step=step)


cifar10.maybe_download_and_extract()
train()
