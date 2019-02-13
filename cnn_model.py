import tensorflow as tf
from tensorflow.data.experimental import CsvDataset
import numpy as np
from data import preprocess
from data import cut
import os


class CNNConfig(object):
    """
    # TODO: 在此修改TextCNN以及训练的参数
    """
    train_mode = 'CHAR'     # 训练模式，'CHAR'为字符级，'WORD'为词级
                            # 'CHAR'务必使用MAX_CHAR_TEXT_LENGTH的序列长度
                            # 'WORD'务必使用MAX_WORD_TEXT_LENGTH的序列长度
    text_length = preprocess.MAX_CHAR_TEXT_LENGTH   # 序列长度
    class_num = 1258        # 输出类别的数目
    embedding_dim = 128      # 词向量维度（64）
    filter_num = 300        # 卷积核数目
    filter_sizes = [2, 3, 4]         # 卷积核尺寸
    vocab_size = preprocess.VOCAB_SIZE      # 词汇表大小

    dense_unit_num = 512        # 全连接层神经元

    dropout_keep_prob = 0.5     # dropout保留比例
    learning_rate = 1e-3    # 学习率

    train_batch_size = 128         # 每批训练大小
    valid_batch_size = 3000       # 每批验证大小
    test_batch_size = 5000        # 每批测试大小
    epoch_num = 30001        # 总迭代轮次
    # TODO: 改成迭代n轮，每轮遍历一次训练集和验证集


class TextCNN(object):
    def __init__(self, config):
        self.class_num = config.class_num
        self.filter_sizes = config.filter_sizes
        self.filter_num = config.filter_num
        self.vocab_size = config.vocab_size

        self.dense_unit_num = config.dense_unit_num
        self.train_batch_size = config.train_batch_size
        self.valid_batch_size = config.valid_batch_size
        self.test_batch_size = config.test_batch_size

        self.text_length = config.text_length
        self.embedding_dim = config.embedding_dim
        self.train_mode = config.train_mode

        self.input_x = ''
        self.input_y = ''
        self.dropout_keep_prob = ''
        self.training = ''
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
        if self.train_mode == 'CHAR':
            # 1.字符级
            vocab = preprocess.read_vocab(os.path.join('./data',preprocess.CHAR_VOCAB_PATH))
            for title in titles:
                batch_x.append(preprocess.to_id(title.decode('gbk'), vocab, self.train_mode))

        elif self.train_mode == 'WORD':
            # 2.词级
            vocab = preprocess.read_vocab(os.path.join('./data', preprocess.WORD_VOCAB_PATH))
            for title in titles:
                t = cut.cut_and_filter(title.decode('gbk'))
                batch_x.append(preprocess.to_id(t, vocab, self.train_mode))

        batch_x = np.stack(batch_x)
        batch_y = labels
        return batch_x, batch_y

    def setCNN(self):
        # Input layer
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, self.text_length], name="input_x")
        self.labels = tf.placeholder(tf.int32, [None], name="input_y")
        self.input_y = tf.one_hot(self.labels, self.class_num)
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # 训练时batch_normalization的Training参数应为True,
        # 验证或测试时应为False
        self.training = tf.placeholder(tf.bool, name='training')

        with tf.device('/cpu:0'):
            W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_dim], -1.0, 1.0))
            self.embedding_inputs = tf.nn.embedding_lookup(W, self.input_x)
            self.embedding_inputs_expanded = tf.expand_dims(self.embedding_inputs, -1)

        # The final pooling output, containing outputs from each filter
        pool_outputs = []
        # Iterate to create convolution layer for each filter
        for filter_size in self.filter_sizes:

            # Convolution layer 1
            # ==================================================================
            # To perform conv2d, filter param should be [height, width, in_channel, out_channel]
            filter_shape = [filter_size, self.embedding_dim]
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
            conv_1_output = tf.layers.batch_normalization(conv_1, training=self.training)
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
        h_full = tf.layers.dense(
            h_pool_flat,
            units=self.dense_unit_num,
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

        self.prediction = tf.argmax(score, 1, name='prediction')

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
                             [tf.string, tf.int32]).shuffle(preprocess.TOTAL_TRAIN_SIZE)
        # Load trained word2vec file
        # productdataset.load_vecs(productdataset.SGNS_WORD_NGRAM_PATH)

        # Splite dataset
        # TODO: 使用k折交叉验证
        train_dataset = dataset.take(preprocess.TRAIN_SIZE).batch(self.train_batch_size).repeat()
        valid_dataset = dataset.skip(preprocess.VALID_SIZE).batch(self.valid_batch_size).repeat()

        # Create a reinitializable iterator
        train_iterator = train_dataset.make_initializable_iterator()
        valid_iterator = valid_dataset.make_initializable_iterator()

        train_init_op = train_iterator.initializer
        valid_init_op = valid_iterator.initializer

        next_train_element = train_iterator.get_next()
        next_valid_element = valid_iterator.get_next()

        return train_dataset, valid_dataset, train_init_op, valid_init_op, next_train_element, next_valid_element
        # =============================================================
        # Date preparation ends.

    def prepare_test_data(self):
        dataset = CsvDataset(os.path.join('./data',preprocess.TEST_PATH),
                             [tf.string])
        dataset = dataset.shuffle(preprocess.TOTAL_TEST_SIZE).batch(self.test_batch_size)

        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        return dataset, next_element




