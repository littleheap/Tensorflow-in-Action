import time
import numpy as np
import tensorflow as tf
import reader


class PTBInput(object):
    """语音模型输入类"""

    def __init__(self, config, data, name=None):
        # 定义batch size
        self.batch_size = batch_size = config.batch_size
        # 定义LSTM展开步数
        self.num_steps = num_steps = config.num_steps
        # 计算epoch size
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        # 使用reader获取输入数据和label数据targets
        self.input_data, self.targets = reader.ptb_producer(
            data, batch_size, num_steps, name=name)


class PTBModel(object):
    """语言模型类"""

    # is_training:训练标记、config:配置参数、input_:输入实例
    def __init__(self, is_training, config, input_):
        # 读取输入input
        self._input = input_
        # 读取batch size
        batch_size = input_.batch_size
        # 读取num steps
        num_steps = input_.num_steps
        # config两个参数读取到本地变量
        size = config.hidden_size
        vocab_size = config.vocab_size

        # 使用BasicLSTMCell设置默认LSTM单元
        def lstm_cell():
            # 隐藏节点为之前的hidden size，forget bias初始化为0，state is truple选择True接受和返回2-tuple
            return tf.contrib.rnn.BasicLSTMCell(
                size, forget_bias=0.0, state_is_tuple=True)

        attn_cell = lstm_cell
        # 训练状态下，keep prod小于1，则在lstm cell前接一个dropout层
        if is_training and config.keep_prob < 1:
            # 调用DropoutWrapper
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    lstm_cell(), output_keep_prob=config.keep_prob)
        # 使用RNN堆叠函数MultiRNNCell将前面多层的lstm cell堆叠到cell，堆叠次数num layers
        cell = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)
        # 设置LSTM单元初始化状态为0
        self._initial_state = cell.zero_state(batch_size, tf.float32)

        # 创建网络词嵌入embedding部分，将onthot单词转化为向量形式
        # 限定使用cpu运行
        with tf.device("/cpu:0"):
            # 初始化embedding矩阵，词汇表数vocab_size，列数hidden size和之前的LSTM隐层节点数一致
            embedding = tf.get_variable(
                "embedding", [vocab_size, size], dtype=tf.float32)
            # embedding_lookup函数查询单词对应的向量表达获得input
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

        # 训练状态下，再添加一层dropout
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        # 定义输出outputs
        outputs = []
        state = self._initial_state
        # variable_scope接下来的操作名称设为RNN
        with tf.variable_scope("RNN"):
            # num steps限制梯度在方向传播时可以展开的步数
            for time_step in range(num_steps):
                # 从第二次循环开始，get_variable_scope().reuse_variables()设置复用变量
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                # 每次循环，传入inputs和state到堆叠的LSTM
                # input三个维度：batch中第几个样本，样本中第几个单词，单词的向量表达的维度
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                # 将结果添加到输出列表
                outputs.append(cell_output)

        # 将output内容串到一起，使用reshape转化为一维向量
        output = tf.reshape(tf.concat(outputs, 1), [-1, size])
        # 构建softmax层
        # 权重W
        softmax_w = tf.get_variable(
            "softmax_w", [size, vocab_size], dtype=tf.float32)
        # 偏置b
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
        # 结果得到logits获得网络输出
        logits = tf.matmul(output, softmax_w) + softmax_b
        # 定义损失函数loss，计算logits和targets偏差，并汇总误差
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(input_.targets, [-1])],
            [tf.ones([batch_size * num_steps], dtype=tf.float32)])
        # 计算每个样本平均误差
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        # 保留最终的状态
        self._final_state = state

        # 如果不是训练，就直接返回
        if not is_training:
            return

        # 定义学习率lr
        self._lr = tf.Variable(0.0, trainable=False)
        # 获取全部参数
        tvars = tf.trainable_variables()
        # 针对前面的cost，计算tvars梯度
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        # 梯度下降算法优化器
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        # 再创建训练操作
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        # 设置new lr用来控制学习速率
        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        # 同时定义学习率更新操作lr update
        self._lr_update = tf.assign(self._lr, self._new_lr)

    # 设置lr函数
    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    # 设置一些装饰器
    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


# 定义几种不同大小模型的参数

class SmallConfig(object):
    """小模型"""
    init_scale = 0.1  # 网络中权重值的初始
    learning_rate = 1.0  # 学习率初始值
    max_grad_norm = 5  # 梯度的最大范数
    num_layers = 2  # LSTM堆叠的层数
    num_steps = 20  # 梯度反向传播的展开步数
    hidden_size = 200  # LSTM隐含节点数
    max_epoch = 4  # 初始学习率可训练的epoch数
    max_max_epoch = 13  # 总训练epoch数
    keep_prob = 1.0  # dropout层保留节点比例
    lr_decay = 0.5  # 学习率衰减系数
    batch_size = 20  # 每个batch中样本的数量
    vocab_size = 10000


class MediumConfig(object):
    """中模型"""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000


class LargeConfig(object):
    """大模型"""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


def run_epoch(session, model, eval_op=None, verbose=False):
    """定义训练一个epoch数据的函数"""
    start_time = time.time()
    # 初始化costs
    costs = 0.0
    # 初始化iters
    iters = 0
    # initial_state初始化状态并获得初始状态
    state = session.run(model.initial_state)

    # 创建输出结果字典表
    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
    }
    # 如果有评测操作，则加入字典表
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    # 训练次数为epoch size
    for step in range(model.input.epoch_size):
        # 传入输入字典
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        # 获取cost和state
        cost = vals["cost"]
        state = vals["final_state"]

        # 累加cost
        costs += cost
        # 累加iters
        iters += model.input.num_steps

        # 每完成10%，打印输出一次结果
        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                   iters * model.input.batch_size / (time.time() - start_time)))

    # 最后返回perplexity
    return np.exp(costs / iters)


# reader读取解压后的数据
raw_data = reader.ptb_raw_data('simple-examples/data/')
# 获取训练数据，验证数据，和测试数据
train_data, valid_data, test_data, _ = raw_data

# 定义训练模型的配置为SmallConfig
config = SmallConfig()
eval_config = SmallConfig()
# 测试配置和训练配置一致
eval_config.batch_size = 1
eval_config.num_steps = 1

# 创建Graph
with tf.Graph().as_default():
    # 设置参数初始化器
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    # 训练
    with tf.name_scope("Train"):
        train_input = PTBInput(config=config, data=train_data, name="TrainInput")
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            m = PTBModel(is_training=True, config=config, input_=train_input)
            # tf.scalar_summary("Training Loss", m.cost)
            # tf.scalar_summary("Learning Rate", m.lr)

    # 验证
    with tf.name_scope("Valid"):
        valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
            # tf.scalar_summary("Validation Loss", mvalid.cost)

    # 测试
    with tf.name_scope("Test"):
        test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mtest = PTBModel(is_training=False, config=eval_config,
                             input_=test_input)

    # 创建训练的管理器
    sv = tf.train.Supervisor()
    # 创建默认session
    with sv.managed_session() as session:
        # 执行多个epoch数据的循环
        for i in range(config.max_max_epoch):
            # 计算累计的学习率衰减值
            lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
            # 设定新学习率
            m.assign_lr(session, config.learning_rate * lr_decay)

            # 在循环内执行一个epoch的训练和验证
            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
            train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                         verbose=True)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            valid_perplexity = run_epoch(session, mvalid)
            print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

        # 并输出当前的学习率，训练和验证集上的perplexity
        test_perplexity = run_epoch(session, mtest)
        print("Test Perplexity: %.3f" % test_perplexity)

        # if FLAGS.save_path:
        #   print("Saving model to %s." % FLAGS.save_path)
        #   sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)

if __name__ == "__main__":
    tf.app.run()
