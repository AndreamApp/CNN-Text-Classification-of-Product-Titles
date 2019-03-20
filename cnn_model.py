# coding=utf-8
from sklearn.metrics import confusion_matrix
import sklearn as sk
import tensorflow as tf
from tensorflow.data import TextLineDataset
import numpy as np
from data import preprocess
from data import cut
import os


class CNNConfig(object):
    """
    # TODO: 在此修改TextCNN以及训练的参数
    """
    def __init__(self, train_mode='CHAR-RANDOM'):
        self.train_mode = train_mode  # 训练模式，'CHAR-RANDOM'为字符级，随机初始化词向量并训练优化
        # 'WORD-NON-STATIC'为词级, 使用word2vec预训练词向量并能够继续在训练中优化
        # 'MULTI'
        self.class_num = 1258  # 输出类别的数目
        self.embedding_dim = 128  # 词向量维度，仅'CHAR-RANDOM'模式适用，
        # 'WORD-NON-STATIC'模式默认为preprocess.py中定义的vec_dim

        self.filter_num = 200  # 卷积核数目
        self.filter_sizes = [2, 3, 4, 5, 6]  # 卷积核尺寸
        self.vocab_size = preprocess.VOCAB_SIZE  # 词汇表大小

        self.dense_unit_num = 512  # 全连接层神经元

        self.dropout_keep_prob = 0.5  # dropout保留比例
        self.learning_rate = 1e-3  # 学习率

        self.train_batch_size = 128  # 每批训练大小
        self.valid_batch_size = 3000  # 每批验证大小
        self.test_batch_size = 5000  # 每批测试大小
        self.valid_per_batch = 1000  # 每多少批进行一次验证
        self.epoch_num = 26  # 总迭代轮次


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

        if config.train_mode == 'CHAR-RANDOM':
            # 文本长度
            self.text_length = preprocess.MAX_CHAR_TEXT_LENGTH
            # 词嵌入维度
            self.embedding_dim = config.embedding_dim

        elif config.train_mode == 'WORD-NON-STATIC' or config.train_mode == 'MULTI':
            self.text_length = preprocess.MAX_WORD_TEXT_LENGTH
            self.embedding_dim = preprocess.vec_dim

        self.train_mode = config.train_mode

        self.input_x = None
        self.input_y = None
        self.labels = None
        self.dropout_keep_prob = None
        self.training = None
        self.embedding_inputs_expanded = None
        self.loss = None
        self.accuracy = None
        self.prediction = None
        self.vocab = None
        self.vecs_dict = {}
        self.embedding_W = None
        self.dataset = None

    def setCNN(self):
        # 输入层
        self.input_x = tf.placeholder(tf.int32, [None, self.text_length], name="input_x")
        self.labels = tf.placeholder(tf.int32, [None], name="input_y")
        # 把数字标签转为one hot形式
        self.input_y = tf.one_hot(self.labels, self.class_num)
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # 训练时batch_normalization的Training参数应为True,
        # 验证或测试时应为False
        self.training = tf.placeholder(tf.bool, name='training')

        # 词嵌入层
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            if self.train_mode == 'CHAR-RANDOM':
                # 随机初始化的词向量
                W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_dim], -1.0, 1.0))
                embedding_inputs = tf.nn.embedding_lookup(W, self.input_x)
                self.embedding_inputs_expanded = tf.expand_dims(embedding_inputs, -1)
            elif self.train_mode == 'WORD-NON-STATIC':
                # 用之前读入的预训练词向量
                W = tf.Variable(self.embedding_W)
                embedding_inputs = tf.nn.embedding_lookup(W, self.input_x)
                self.embedding_inputs_expanded = tf.expand_dims(embedding_inputs, -1)
            elif self.train_mode == 'MULTI':
                W1 = tf.Variable(self.embedding_W)
                W2 = tf.Variable(self.embedding_W, trainable=False)
                embedding_inputs1 = tf.nn.embedding_lookup(W1, self.input_x)
                embedding_inputs2 = tf.nn.embedding_lookup(W2, self.input_x)
                self.embedding_inputs_expanded = tf.stack([embedding_inputs1, embedding_inputs2], axis=-1)

        # The final pooling output, containing outputs from each filter
        pool_outputs = []
        # Iterate to create convolution layer for each filter
        for filter_size in self.filter_sizes:
            with tf.name_scope("conv-maxpool-%d" % filter_size):
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
        with tf.name_scope('dense-%d' % self.dense_unit_num):
            h_full = tf.layers.dense(
                h_pool_flat,
                units=self.dense_unit_num,
                use_bias=True,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                bias_initializer=tf.constant_initializer(0.1)
            )
            h_full = tf.layers.dropout(h_full, rate=self.dropout_keep_prob)
            h_full = tf.nn.relu(h_full)
        # =========================================================================

        # Output layer
        with tf.name_scope('output'):
            score = tf.layers.dense(
                h_full,
                units=self.class_num,
                activation=None,
                use_bias=True,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                bias_initializer=tf.constant_initializer(0.1)
            )
            self.score = tf.multiply(score, 1, name='score')
            self.prediction = tf.argmax(score, 1, name='prediction')

        # Loss function
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=score, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        # Calculate accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.prediction, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    def convert_input(self, lines):
        """
        将训练集数据转换为id或词向量表示
        """
        batch_x = []
        batch_y = []
        title = ""

        for line in lines:
            line_ = line.decode("gbk").strip().split(',')
            title = ''.join(line_[0:-1])    # 逗号前段为标题
            label = ''.join(line_[-1])      # 最后一项为标签
            batch_x.append(preprocess.to_id(title, self.vocab, self.train_mode))
            batch_y.append(label)

        batch_x = np.stack(batch_x)
        return batch_x, batch_y

    def convert_test_input(self, titles):
        """
        将测试集tsv数据转为id或词向量表示
        :param titles:
        :return:
        """
        batch_x = []
        # 1.id
        for title in titles:
            valid_title = title.decode('gb18030').strip('\t')
            batch_x.append(preprocess.to_id(valid_title, self.vocab, self.train_mode))

        batch_x = np.stack(batch_x)
        return batch_x

    def prepare_data(self):
        # Data preparation.
        # =======================================================
        if self.train_mode == 'CHAR-RANDOM':
            # 1.字符级
            # 读取词汇表
            self.vocab = preprocess.read_vocab(os.path.join('data',preprocess.CHAR_VOCAB_PATH))

        elif self.train_mode == 'WORD-NON-STATIC' or self.train_mode == 'MULTI':
            # 把预训练词向量的值读到变量中
            self.vocab = preprocess.read_vocab(os.path.join('data', preprocess.WORD_VOCAB_PATH))
            self.vecs_dict = preprocess.load_vecs(os.path.join('data', preprocess.SGNS_WORD_PATH))
            self.embedding_W = np.ndarray(shape=[self.vocab_size, self.embedding_dim], dtype=np.float32)
            for word in self.vocab:
                # 第n行对应id为n的词的词向量
                if word not in self.vecs_dict:
                    preprocess.add_word(word, self.vecs_dict)
                self.embedding_W[self.vocab[word]] = self.vecs_dict[word]

        self.dataset = TextLineDataset(os.path.join('data', preprocess.TRAIN_WITH_ID_PATH))

        return
        # =============================================================
        # Date preparation ends.

    def shuffle_datset(self):
        # 打乱数据集
        # ==========================================================
        print('Shuffling dataset...')
        self.dataset = self.dataset.shuffle(preprocess.TOTAL_TRAIN_SIZE)

        # 分割数据集
        # 取前VALID_SIZE个样本给验证集
        valid_dataset = self.dataset.take(preprocess.VALID_SIZE).batch(self.valid_batch_size)
        # 剩下的给训练集
        train_dataset = self.dataset.skip(preprocess.VALID_SIZE).batch(self.train_batch_size)

        # Create a reinitializable iterator
        train_iterator = train_dataset.make_initializable_iterator()
        valid_iterator = valid_dataset.make_initializable_iterator()

        train_init_op = train_iterator.initializer
        valid_init_op = valid_iterator.initializer

        # 要获取元素，先sess.run(train_init_op)初始化迭代器
        # 再sess.run(next_train_element)
        next_train_element = train_iterator.get_next()
        next_valid_element = valid_iterator.get_next()

        return train_init_op, valid_init_op, next_train_element, next_valid_element
        # ==============================================================

    def prepare_test_data(self):
        # 读取词汇表
        if self.train_mode == 'CHAR-RANDOM':
            # 1.字符级
            self.vocab = preprocess.read_vocab(os.path.join('data',preprocess.CHAR_VOCAB_PATH))

        elif self.train_mode == 'WORD-NON-STATIC' or self.train_mode == 'MULTI':
            self.vocab = preprocess.read_vocab(os.path.join('data', preprocess.WORD_VOCAB_PATH))

        # 测试集有标题，读取时注意跳过第一行
        dataset = TextLineDataset(os.path.join('data',preprocess.TEST_PATH))
        dataset = dataset.shuffle(preprocess.TOTAL_TEST_SIZE).batch(self.test_batch_size)

        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        return dataset, next_element




