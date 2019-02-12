import tensorflow as tf
from tensorflow.data.experimental import CsvDataset
import numpy as np
from data import preprocess
import os
import datetime
import time
from data import cut


class CNNConfig(object):
    """
    TODO: Create a class to store parameters for CNN.
    """
    embedding_dim = 64      # 词向量维度
    seq_length = 600        # 序列长度
    num_classes = 10        # 类别数
    num_filters = 128        # 卷积核数目
    kernel_size = 5         # 卷积核尺寸
    vocab_size = 5000       # 词汇表达小

    hidden_dim = 128        # 全连接层神经元

    dropout_keep_prob = 0.5 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 64         # 每批训练大小
    num_epochs = 10         # 总迭代轮次

    print_per_batch = 100    # 每多少轮输出一次结果
    save_per_batch = 10      # 每多少轮存入tensorboard


class TextCNN(object):
    def __init__(self, class_num, filter_sizes, filter_num,
                 vocab_size = 4000,
                 text_length=preprocess.MAX_TEXT_LENGTH, embedding_dim=128):
        """
        :param class_num: Number of classes in the output layer.
        :param filter_sizes: A list, specifying each filter size we want.
        :param filter_num: The number of filter of each filter size.
        :param text_length: The length of each text, padding if needed.
        :param embedding_dim: The dimension of our embedding. If pre-trained word2vec is used, 300.
        """
        self.class_num = class_num
        self.filter_sizes = filter_sizes
        self.filter_num = filter_num
        self.vocab_size = vocab_size
        self.text_length = text_length
        self.embedding_size = embedding_dim
        self.input_x = ''
        self.input_y = ''
        self.dropout_keep_prob = ''
        self.embedding_inputs = ''
        self.embedding_inputs_expanded = ''
        self.loss = ''
        self.accuracy = ''
        self.prediction = ''

    def convert_input(self, titles, labels):
        """
        将数据集转换为id表示
        """
        batch_x = []
        # 读取词汇表
        # 1.字符级
        # vocab = preprocess.read_vocab(os.path.join('./data',preprocess.CHAR_VOCAB_PATH))
        # for title in titles:
        #     batch_x.append(preprocess.to_id(title.decode('gbk'), vocab))
        # 2.词级
        vocab = preprocess.read_vocab(os.path.join('./data', preprocess.WORD_VOCAB_PATH))

        for title in titles:
            t = cut.cut_and_filter(title.decode('gbk'))
            batch_x.append(preprocess.to_id(t, vocab))

        batch_x = np.stack(batch_x)
        batch_y = labels
        return batch_x, batch_y

    def setCNN(self):
        # TODO:在此修改CNN参数
        # Input layer
        # Placeholders for input, output and dropout
        # Input for conv2d should have shape[batch, height, width, channel]
        self.input_x = tf.placeholder(tf.int32, [None, self.text_length], name="input_x")
        #self.input_y = tf.placeholder(tf.float32, [None, self.class_num], name="input_y")
        self.labels = tf.placeholder(tf.int32, [None], name="input_y")
        self.input_y = tf.one_hot(self.labels, self.class_num)
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.device('/cpu:0'):
            W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0))
            self.embedding_inputs = tf.nn.embedding_lookup(W, self.input_x)
            self.embedding_inputs_expanded = tf.expand_dims(self.embedding_inputs, -1)


        # The final pooling output, containing outputs from each filter
        pool_outputs = []
        # Iterate to create convolution layer for each filter
        for filter_size in self.filter_sizes:

            # Convolution layer 1
            # ==================================================================
            # To perform conv2d, filter param should be [height, width, in_channel, out_channel]
            filter_shape = [filter_size, self.embedding_size]
            conv_1 = tf.layers.conv2d(
                inputs=self.embedding_inputs_expanded,
                filters=self.filter_num,
                kernel_size=filter_shape,
                strides=[1, 1],
                padding='VALID',
                use_bias=True,
                kernel_initializer=tf.initializers.truncated_normal(stddev=0.1),
                bias_initializer=tf.initializers.constant(0.1)
            )
            # ===================================================================
            # Do batch normalization
            # =================================================================
            # For image, do norm on dimension[0, 1, 2] for [batch, height, width]
            # conv_1_shape = conv_1.shape.as_list()
            # axes = list(range(len(conv_1_shape)-1))
            # mean, varience = tf.nn.moments(conv_1, axes)

            # dim = conv_1_shape[3]
            # scale = tf.Variable(tf.ones([dim]))
            # offset = tf.Variable(tf.zeros([dim]))
            # epsilon = 0.001
            # conv_1_output = tf.nn.batch_normalization(conv_1, mean, varience,
            #                                           offset, scale, epsilon)
            # Use more convenient tf.layers.batch_normalization
            conv_1_output = tf.layers.batch_normalization(conv_1, training=True)
            conv_1_output = tf.nn.relu(conv_1_output)
            # ======================================================================
            # Pooling layer 1
            # ====================================================================
            conv_1_output_shape = conv_1_output.shape.as_list()
            pool_1 = tf.layers.max_pooling2d(
                inputs=conv_1_output,
                pool_size=[conv_1_output_shape[1] - 1 + 1, 1],
                strides=[1, 1],
                padding='VALID'
            )
            # =====================================================================

            pool_outputs.append(pool_1)

        # Combine all the pooling output
        # The total number of filters.
        total_filter_num = self.filter_num * len(self.filter_sizes)
        h_pool = tf.concat(pool_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, total_filter_num])
        # Output shape[batch, total_filter_num]


        # Full-connected layer
        # ========================================================================
        # TODO:
        h_full = tf.layers.dense(
            h_pool_flat,
            units=512,
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
            bias_initializer=tf.constant_initializer(0.1)
        )
        # Add dropout
        h_drop = tf.nn.dropout(h_full, self.dropout_keep_prob)
        # =========================================================================

        # Output layer
        score = tf.layers.dense(
            h_full,
            units=self.class_num,
            activation=None,
            use_bias=True,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
            bias_initializer=tf.constant_initializer(0.1)
        )

        self.prediction = tf.argmax(score, 1)

        # Loss function
        losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=score, labels=self.input_y)
        self.loss = tf.reduce_mean(losses)

        # Calculate accuracy
        correct_predictions = tf.equal(self.prediction, tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    def prepare_data(self):
        # Data preparation.
        # =======================================================

        # Load data
        dataset = CsvDataset(os.path.join('./data',preprocess.TRAIN_WITH_ID_PATH),
                             [tf.string, tf.int32]).shuffle(500000)
        # Load trained word2vec file
        # productdataset.load_vecs(productdataset.SGNS_WORD_NGRAM_PATH)

        # Splite dataset
        # TODO: Should use k-fold cross validation
        train_dataset = dataset.take(preprocess.TRAIN_SIZE).batch(128).repeat()
        valid_dataset = dataset.skip(preprocess.VALID_SIZE).batch(1000).repeat()

        # Create a reinitializable iterator
        train_iterator = train_dataset.make_initializable_iterator()
        valid_iterator= valid_dataset.make_initializable_iterator()

        train_init_op = train_iterator.initializer
        valid_init_op = valid_iterator.initializer

        next_train_element = train_iterator.get_next()
        next_valid_element = valid_iterator.get_next()

        return train_dataset, valid_dataset, train_init_op, valid_init_op, next_train_element, next_valid_element
        # =============================================================
        # Date preparation ends.


def train():
    # Training procedure
    # ======================================================
    with tf.Session() as sess:

        text_length = preprocess.MAX_TEXT_LENGTH
        cnn = TextCNN(
            class_num=1258,
            filter_sizes=[2, 3, 4],
            filter_num=300,
            text_length=text_length,
        )
        train_dataset, valid_dataset, train_init_op, valid_init_op, next_train_element, next_valid_element = cnn.prepare_data()
        cnn.setCNN()

        global_step = tf.Variable(0, trainable=False)

        # 保证Batch normalization的执行
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):  # 保证train_op在update_ops执行之后再执行。
            train_op = tf.train.AdamOptimizer(1e-4).minimize(cnn.loss, global_step)

        def train_step(batch_x, batch_y, keep_prob=0.5):
            feed_dict = {
                cnn.input_x: batch_x,
                cnn.labels: batch_y,
                cnn.dropout_keep_prob: keep_prob
            }
            sess.run(train_op, feed_dict=feed_dict)
            step, loss, accuracy = sess.run([global_step, cnn.loss, cnn.accuracy], feed_dict={cnn.input_x: batch_x,
                cnn.labels: batch_y,
                cnn.dropout_keep_prob: 1.0})
            time = datetime.datetime.now().isoformat()
            print('%s: step: %d, loss: %f, accuracy: %f' % (time, step, loss, accuracy))

        def valid_step(batch_x, batch_y):
            feed_dict = {
                cnn.input_x: batch_x,
                cnn.labels: batch_y,
                cnn.dropout_keep_prob: 1.0,
            }
            step, loss, accuracy = sess.run([global_step, cnn.loss, cnn.accuracy],
                                            feed_dict)
            time = datetime.datetime.now().isoformat()
            print('Validation loss: %f, accuracy: %f' % (loss, accuracy))

        # Set checkpoint to save model
        checkpoint_dir = os.path.abspath("checkpoints")
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        sess.run(tf.global_variables_initializer())

        sess.run(train_init_op)
        sess.run(valid_init_op)

        # Training loop
        for epoch in range(30001):
            titles, labels = sess.run(next_train_element)
            batch_x, batch_y = cnn.convert_input(titles, labels)
            train_step(batch_x, batch_y, 0.5)
            if epoch % 500 == 0:
                for _ in range(10):
                    titles, labels = sess.run(next_valid_element)
                    batch_x, batch_y = cnn.convert_input(titles, labels)
                    valid_step(batch_x, batch_y)
                time.sleep(1)


if __name__ == '__main__':
    train()







