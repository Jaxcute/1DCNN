import tensorflow as tf
import proprecess as pp

batch_size = 128
epochs = 20
num_classes = 4
length = 2048
BatchNorm = True  # 是否批量归一化
number = 1000  # 每类样本的数量
normal = True  # 是否标准化
rate = [0.6, 0.1, 0.3]  # 测试集验证集划分比例

path = r'data\0HP'
x_train, y_train, x_valid, y_valid, x_test, y_test = pp.prepro(d_path=path, length=length,
                                                               number=number,
                                                               normal=False,
                                                               rate=rate,
                                                               enc=False, enc_step=28)

print("ddlolfdkop", x_train.shape)
print('训练样本维度:', x_train.shape)
print(x_train.shape[0], '训练样本个数')
print('验证样本的维度', x_valid.shape)
print(x_valid.shape[0], '验证样本个数')
print('测试样本的维度', x_test.shape)
print(x_test.shape[0], '测试样本个数')


# 计算准确率：
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={x: v_xs, keep_prob: 0.5})
    # print("y_pre.shape: ", y_pre.shape)
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))  # TP + tn
    # print("correct_prediction: ", correct_prediction)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # result = sess.run(accuracy, feed_dict={x: v_xs, y: v_ys, keep_prob: 1})
    equal = tf.equal(tf.argmax(y_pre, axis=1), tf.argmax(v_ys, axis=1))
    # 取均值计算正确率
    correct = tf.reduce_mean(tf.cast(equal, tf.float32))
    result = sess.run(correct)

    return result

# 输入层：
with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, [None, 2048], name="x_input")
    y = tf.placeholder(tf.float32, [None, 4], name="y_output")
    keep_prob = tf.placeholder(tf.float32)
    input = tf.reshape(x, [-1, 2048, 1])

# 卷积层1：
with tf.name_scope("conv1"):
    w1 = tf.Variable(tf.truncated_normal([4, 1, 32], stddev=0.1, ), name='w1')
    b1 = tf.Variable(tf.truncated_normal([32], stddev=0.1), name='b1')
    conv1_1 = tf.nn.conv1d(value=input, filters=w1, stride=2, padding="SAME", name='conv1_1')
    print("conv1_1.shape: ", conv1_1.shape)
    conv1_2 = tf.nn.bias_add(value=conv1_1, bias=b1, name='conv1_2')
    print("conv1_2.shape: ", conv1_2.shape)
    activation1 = tf.nn.relu(conv1_2, name='activation1')
    print("activation1.shape: ", activation1.shape)
    pool1 = tf.layers.max_pooling1d(inputs=activation1, pool_size=2, strides=2, padding="SAME", name='pool1')
    print(pool1.shape)

# 卷积层2：
with tf.name_scope("conv2"):
    w2 = tf.Variable(tf.truncated_normal([4, 32, 64], stddev=0.1, ), name='w2')
    b2 = tf.Variable(tf.truncated_normal([64], stddev=0.1), name='b2')
    conv2_1 = tf.nn.conv1d(value=pool1, filters=w2, stride=2, padding="SAME", name='conv2_1')
    print("conv2_1.shape: ", conv2_1.shape)
    conv2_2 = tf.nn.bias_add(value=conv2_1, bias=b2, name='conv2_2')
    print("conv2_2.shape: ", conv2_2.shape)
    activation2 = tf.nn.relu(conv2_2, name='activation2')
    print("activation2.shape: ", activation2.shape)
    pool2 = tf.layers.max_pooling1d(inputs=activation2, pool_size=2, strides=2, padding="SAME", name='pool2')
    print(pool2.shape)

# 卷积层3：
with tf.name_scope("conv3"):
    w3 = tf.Variable(tf.truncated_normal([4, 64, 128], stddev=0.1, ), name='w3')
    b3 = tf.Variable(tf.truncated_normal([128], stddev=0.1), name='b3')
    conv3_1 = tf.nn.conv1d(value=pool2, filters=w3, stride=2, padding="SAME", name='conv3_1')
    print("conv3_1.shape: ", conv3_1.shape)
    conv3_2 = tf.nn.bias_add(value=conv3_1, bias=b3, name='conv3_2')
    print("conv3_2.shape: ", conv3_2.shape)
    activation3 = tf.nn.relu(conv3_2, name='activation3')
    print("activation3.shape: ", activation3.shape)
    pool3 = tf.layers.max_pooling1d(inputs=activation3, pool_size=2, strides=2, padding="SAME", name='pool3')
    print(pool3.shape)

# 全连接层1:
with tf.name_scope("full_layer1"):
    w_q1 = tf.Variable(tf.truncated_normal([32 * 128, 1024], stddev=0.1, ), name='w_q1')
    b_q1 = tf.Variable(tf.constant(0.0, shape=[1024]), name='b_q1')
    pool_q1 = tf.reshape(pool3, [-1, 32 * 128])
    activation_q1 = tf.nn.relu(tf.matmul(pool_q1, w_q1) + b_q1, name='activation_q1')
    print("activation_q1: ", activation_q1.shape)
    drop = tf.nn.dropout(activation_q1, keep_prob, name='drop')
    # print("drop: ", drop.shape)

# 全连接层2：
with tf.name_scope("full_layer2"):
    w_q2 = tf.Variable(tf.truncated_normal([1024, 4], stddev=0.1, ), name='w_q2')
    b_q2 = tf.Variable(tf.constant(0.0, shape=[4]), name='b_q2')
    logits = tf.matmul(drop, w_q2) + b_q2
    print("logits: ", logits.shape)
    prediction = tf.nn.softmax(logits, name='prediction')
    # print("prediction: ", prediction)

# 计算正确率：
with tf.name_scope("compute"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
    print("loss: ", loss)
    train_step = tf.train.GradientDescentOptimizer(0.00008).minimize(loss)
    correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# 保存模型
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('logs', sess.graph)

    print(x_train.shape)
    print(y_train.shape)
    for i in range(1, 2501):
        for idx in range(len(x_train) // batch_size):
            batch_x_train = x_train[idx * batch_size:(idx + 1) * batch_size]
            batch_y_train = y_train[idx * batch_size:(idx + 1) * batch_size]
            train_step.run(feed_dict={x: batch_x_train,
                                      y: batch_y_train,
                                      keep_prob: 0.5})
        if (i % 100 == 0):
            print("第", i, "轮训练集正确率：", str('%.4f' % (compute_accuracy(x_train, y_train) * 100)), "%")
            print("第", i, "轮测试集正确率：", str('%.4f' % (compute_accuracy(x_test, y_test) * 100)), "%")
            print("")

    saver.save(sess, 'my_test_model', global_step=2500)
