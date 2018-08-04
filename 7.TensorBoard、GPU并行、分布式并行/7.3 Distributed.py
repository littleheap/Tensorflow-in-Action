import math
import tempfile
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 定义标记
flags = tf.app.flags
# 设定存储路径
flags.DEFINE_string("data_dir", "/tmp/mnist-data",
                    "Directory for storing mnist data")
# 设置task index
flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")
# 累计多少个梯度来更新模型
flags.DEFINE_integer("replicas_to_aggregate", None,
                     "Number of replicas to aggregate before parameter update"
                     "is applied (For sync_replicas mode only; default: "
                     "num_workers)")
# 默认隐层节点100
flags.DEFINE_integer("hidden_units", 100,
                     "Number of units in the hidden layer of the NN")
# 最大训练步数1000000
flags.DEFINE_integer("train_steps", 1000000,
                     "Number of (global) training steps to perform")
# batch size大小为100
flags.DEFINE_integer("batch_size", 100, "Training batch size")
# 学习率0.01
flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
# 设定是否使用同步并行标记
flags.DEFINE_boolean("sync_replicas", False,
                     "Use the sync_replicas (synchronized replicas) mode, "
                     "wherein the parameter updates from workers are aggregated "
                     "before applied to avoid stale gradients")
# 设置ps地址
flags.DEFINE_string("ps_hosts", "192.168.233.201:2222",
                    "Comma-separated list of hostname:port pairs")
# 设置worker地址
flags.DEFINE_string("worker_hosts", "192.168.233.202:2223,192.168.233.203:2224",
                    "Comma-separated list of hostname:port pairs")
# 设置job name
flags.DEFINE_string("job_name", None, "job name: worker or ps")
# 设置flag
FLAGS = flags.FLAGS
# 图片尺寸
IMAGE_PIXELS = 28


# 主函数
def main(unused_argv):
    # 获取数据集
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    # 确保有job name和task index两个参数
    if FLAGS.job_name is None or FLAGS.job_name == "":
        raise ValueError("Must specify an explicit `job_name`")
    if FLAGS.task_index is None or FLAGS.task_index == "":
        raise ValueError("Must specify an explicit `task_index`")
    # 打印两个参数
    print("job name = %s" % FLAGS.job_name)
    print("task index = %d" % FLAGS.task_index)

    # 将ps和worker地址解析成列表
    ps_spec = FLAGS.ps_hosts.split(",")
    worker_spec = FLAGS.worker_hosts.split(",")

    # 获取worker数量
    num_workers = len(worker_spec)

    # 生成cluster对象，传入ps和worker
    cluster = tf.train.ClusterSpec({
        "ps": ps_spec,
        "worker": worker_spec})

    # 创建当前机器server连接到cluster
    server = tf.train.Server(
        cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    # 如果当前是ps就join等待worker
    if FLAGS.job_name == "ps":
        server.join()

    # 查看当前机器是否为主节点
    is_chief = (FLAGS.task_index == 0)

    # 定义当前机器work device
    worker_device = "/job:worker/task:%d/gpu:0" % FLAGS.task_index
    # 设置一台机器中资源
    with tf.device(
            tf.train.replica_device_setter(worker_device=worker_device, ps_device="/job:ps/cpu:0", cluster=cluster)):
        # 记录全局训练步数变量
        global_step = tf.Variable(0, name="global_step", trainable=False)

        # 定义神经网络模型
        # 初始化权重
        hid_w = tf.Variable(
            tf.truncated_normal(
                [IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
                stddev=1.0 / IMAGE_PIXELS),
            name="hid_w")
        # 初始化偏置
        hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b")

        # 创建隐层节点到输出层节点
        sm_w = tf.Variable(
            tf.truncated_normal(
                [FLAGS.hidden_units, 10],
                stddev=1.0 / math.sqrt(FLAGS.hidden_units)),
            name="sm_w")
        # softmax输出结果
        sm_b = tf.Variable(tf.zeros([10]), name="sm_b")

        # 初始化输入和对应标记
        x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
        y_ = tf.placeholder(tf.float32, [None, 10])

        # 计算第一个隐层输出
        hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
        # 激活函数激活输出
        hid = tf.nn.relu(hid_lin)

        # 计算最终输出结果
        y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
        # 计算交叉熵
        cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

        # adam优化器
        opt = tf.train.AdamOptimizer(FLAGS.learning_rate)

        # 如果设置了同步模式
        if FLAGS.sync_replicas:
            # 先获取同步更新模型所需要的副本数
            if FLAGS.replicas_to_aggregate is None:
                replicas_to_aggregate = num_workers
            # 没有副本数设置，使用worker数
            else:
                replicas_to_aggregate = FLAGS.replicas_to_aggregate

            # 创建同步训练优化器
            opt = tf.train.SyncReplicasOptimizer(
                opt,
                replicas_to_aggregate=replicas_to_aggregate,
                total_num_replicas=num_workers,
                replica_id=FLAGS.task_index,
                name="mnist_sync_replicas")
        # 设定训练步伐
        train_step = opt.minimize(cross_entropy, global_step=global_step)

        # 如果是同步训练模式主节点
        if FLAGS.sync_replicas and is_chief:
            # 初始化队列执行器
            chief_queue_runner = opt.get_chief_queue_runner()
            # 创建全局参数初始化器
            init_tokens_op = opt.get_init_tokens_op()

        # 本地初始化全局变量
        init_op = tf.global_variables_initializer()
        # 创建临时训练目录
        train_dir = tempfile.mkdtemp()
        # 创建分布式训练监督器
        sv = tf.train.Supervisor(
            is_chief=is_chief,
            logdir=train_dir,
            init_op=init_op,
            recovery_wait_secs=1,
            global_step=global_step)
        # 设置session参数
        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index])

        # 如果为主节点
        if is_chief:
            # 显示初始化session
            print("Worker %d: Initializing session..." % FLAGS.task_index)
        else:
            # 其他节点显示等待主节点初始化操作
            print("Worker %d: Waiting for session to be initialized..." %
                  FLAGS.task_index)

        # 主节点会创建session，分支节点会等待
        sess = sv.prepare_or_wait_for_session(server.target,
                                              config=sess_config)
        # 打印task index
        print("Worker %d: Session initialization complete." % FLAGS.task_index)

        # 如果同步模式且为主节点
        if FLAGS.sync_replicas and is_chief:
            # 执行队列化执行器
            print("Starting chief queue runner and running init_tokens_op")
            sv.start_queue_runners(sess, [chief_queue_runner])
            sess.run(init_tokens_op)

        # 记录启动时间
        time_begin = time.time()
        print("Training begins @ %f" % time_begin)

        local_step = 0
        while True:
            # 获取训练数据
            batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
            # 喂数据
            train_feed = {x: batch_xs, y_: batch_ys}

            _, step = sess.run([train_step, global_step], feed_dict=train_feed)
            local_step += 1

            now = time.time()
            print("%f: Worker %d: training step %d done (global step: %d)" %
                  (now, FLAGS.task_index, local_step, step))

            # 达到预设最大值停止训练
            if step >= FLAGS.train_steps:
                break

        time_end = time.time()
        print("Training ends @ %f" % time_end)
        training_time = time_end - time_begin
        print("Training elapsed time: %f s" % training_time)

        # 计算验证集交叉熵损失
        val_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        val_xent = sess.run(cross_entropy, feed_dict=val_feed)
        print("After %d training step(s), validation cross entropy = %g" %
              (FLAGS.train_steps, val_xent))


if __name__ == "__main__":
    tf.app.run()
