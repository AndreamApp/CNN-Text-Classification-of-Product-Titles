import tensorflow as tf
from tensorflow.data.experimental import CsvDataset
from tensorflow.data import TextLineDataset
import numpy as np
from data import preprocess
from data import cut
import os


class RNNConfig(object):
    """
    # TODO: 在此修改RNN以及训练的参数
    """
    train_mode = 'CHAR'     # 训练模式，'CHAR'为字符级，样本分割为字符并使用自训练词嵌入
                            # 'WORD'为词级，样本分词并使用word2vec预训练的词向量

    class_num = 1258        # 输出类别的数目
    embedding_dim = 128      # 词向量维度，'CHAR'模式适用，
                            # 'WORD'模式默认为preprocess.py中定义的vec_dim

    layer_num = 4      # rnn层数
    units_num = 128    # rnn神经元数目

    dense_unit_num = 128        # 全连接层神经元

    vocab_size = preprocess.VOCAB_SIZE      # 词汇表大小

    dropout_keep_prob = 0.5     # dropout保留比例
    learning_rate = 5e-3    # 学习率

    train_batch_size = 128         # 每批训练大小
    valid_batch_size = 3000       # 每批验证大小
    test_batch_size = 5000        # 每批测试大小
    valid_per_batch = 500           # 每多少批进行一次验证
    epoch_num = 25001        # 总迭代轮次


class RNN(object):
    def __init__(self, config):
        self.class_num = config.class_num
        self.layer_num = config.layer_num
        self.units_num = config.units_num
        self.dense_unit_num = config.dense_unit_num

        self.vocab_size = config.vocab_size

        self.train_batch_size = config.train_batch_size
        self.valid_batch_size = config.valid_batch_size
        self.test_batch_size = config.test_batch_size

        if config.train_mode == 'CHAR':
            # 文本长度
            self.text_length = preprocess.MAX_CHAR_TEXT_LENGTH
            # 词嵌入维度
            self.embedding_dim = config.embedding_dim

        elif config.train_mode == 'WORD':
            self.text_length = preprocess.MAX_WORD_TEXT_LENGTH
            self.embedding_dim = preprocess.vec_dim

        self.train_mode = config.train_mode

        self.input_x = None
        self.input_y = None
        self.labels = None
        self.dropout_keep_prob = None
        self.training = None
        self.embedding_inputs = None
        self.loss = None
        self.accuracy = None
        self.prediction = None
        self.vocab = None
        self.vecs_dict = {}

    def setRNN(self):
        # 输入层
        # Placeholders for input, output and dropout
        if self.train_mode == 'CHAR':
            self.input_x = tf.placeholder(tf.int32, [None, self.text_length], name="input_x")
        elif self.train_mode == 'WORD':
            self.input_x = tf.placeholder(tf.float32, [None, self.text_length, self.embedding_dim], name="input_x")

        self.labels = tf.placeholder(tf.int32, [None], name="input_y")
        self.input_y = tf.one_hot(self.labels, self.class_num)
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # 训练时batch_normalization的Training参数应为True,
        # 验证或测试时应为False
        self.training = tf.placeholder(tf.bool, name='training')

        if self.train_mode == 'CHAR':
            # 词嵌入层
            with tf.device('/cpu:0'), tf.name_scope('embedding'):
                W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_dim], -1.0, 1.0))
                self.embedding_inputs = tf.nn.embedding_lookup(W, self.input_x)
        elif self.train_mode == 'WORD':
            # 不通过词嵌入层
            self.embedding_inputs = self.input_x

        with tf.name_scope("batch_norm"):
            self.embedding_inputs = tf.layers.batch_normalization(self.embedding_inputs, training=self.training)

        def basic_rnn_cell():
            bcell = tf.nn.rnn_cell.BasicRNNCell(self.units_num)
            return tf.nn.rnn_cell.DropoutWrapper(bcell, output_keep_prob=self.dropout_keep_prob)

        with tf.name_scope("RNN"):
            # 多层RNN网络，每层有units_num个神经元
            # =======================================================================================
            cells = [basic_rnn_cell() for _ in range(self.layer_num)]
            cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
            # output的形状为[batch_size, text_length, units_num]
            output, states = tf.nn.dynamic_rnn(cell=cell, inputs=self.embedding_inputs, dtype=tf.float32)
            rnn_output = output[:, -1, :]  # 取最后一个时序作为输出结果
            # =========================================================================================
        with tf.name_scope("dense"):
            # 全连接层
            # ======================================================================================
            h_full = tf.layers.dense(inputs=rnn_output,
                                   units=self.dense_unit_num,
                                   activation=tf.nn.relu,
                                   use_bias=True,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                   bias_initializer=tf.constant_initializer(0.1)
                                   )
            # ==========================================================================================
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

            self.prediction = tf.argmax(score, 1, name='prediction')

        # Loss function
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=score, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        # Calculate accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.prediction, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    def convert_input(self, titles, labels):
        """
        将训练集数据转换为id或词向量表示
        """
        batch_x = []
        if self.train_mode == 'CHAR':
            # 1.id
            for title in titles:
                batch_x.append(preprocess.to_id(title.decode('gbk'), self.vocab, self.train_mode))

        elif self.train_mode == 'WORD':
            # 2.词向量
            for title in titles:
                t = cut.cut_and_filter(title.decode('gbk'))
                batch_x.append(preprocess.get_word_vecs(title, self.vecs_dict))

        batch_x = np.stack(batch_x)
        batch_y = labels
        return batch_x, batch_y

    def convert_test_input(self, titles):
        """
        将测试集tsv数据转为id或词向量表示
        :param titles:
        :return:
        """
        batch_x = []
        if self.train_mode == 'CHAR':
            # 1.id
            for title in titles:
                valid_title = title.decode('gb18030').strip('\t')
                batch_x.append(preprocess.to_id(valid_title, self.vocab, self.train_mode))

        elif self.train_mode == 'WORD':
            # 2.词向量
            for title in titles:
                valid_title = title.decode('gb18030').strip('\t')
                t = cut.cut_and_filter(valid_title)
                batch_x.append(preprocess.get_word_vecs(title, self.vecs_dict))

        batch_x = np.stack(batch_x)
        return batch_x

    def prepare_data(self):
        # Data preparation.
        # =======================================================
        if self.train_mode == 'CHAR':
            # 1.字符级
            # 读取词汇表
            self.vocab = preprocess.read_vocab(os.path.join('data',preprocess.CHAR_VOCAB_PATH))
        elif self.train_mode == 'WORD':
            # 2.词级
            # self.vocab = preprocess.read_vocab(os.path.join('data', preprocess.WORD_VOCAB_PATH))
            # 读取词向量文件
            self.vecs_dict = preprocess.load_vecs(os.path.join('data', preprocess.SGNS_WORD_NGRAM_PATH))

        # CsvDataset类加载csv文件
        dataset = CsvDataset(os.path.join('./data',preprocess.TRAIN_WITH_ID_PATH),
                             [tf.string, tf.int32]).shuffle(preprocess.TOTAL_TRAIN_SIZE)

        # 分割数据集
        # TODO: 使用k折交叉验证
        # 取前VALID_SIZE个样本给验证集
        valid_dataset = dataset.take(preprocess.VALID_SIZE).batch(self.valid_batch_size).repeat()
        # 剩下的给训练集
        train_dataset = dataset.skip(preprocess.VALID_SIZE).batch(self.train_batch_size).repeat()

        # Create a reinitializable iterator
        train_iterator = train_dataset.make_initializable_iterator()
        valid_iterator = valid_dataset.make_initializable_iterator()

        train_init_op = train_iterator.initializer
        valid_init_op = valid_iterator.initializer

        # 要获取元素，先sess.run(train_init_op)初始化迭代器
        # 再sess.run(next_train_element)
        next_train_element = train_iterator.get_next()
        next_valid_element = valid_iterator.get_next()

        return train_dataset, valid_dataset, train_init_op, valid_init_op, next_train_element, next_valid_element
        # =============================================================
        # Date preparation ends.

    def prepare_test_data(self):
        # 读取词汇表
        if self.train_mode == 'CHAR':
            # 1.字符级
            self.vocab = preprocess.read_vocab(os.path.join('data',preprocess.CHAR_VOCAB_PATH))
        elif self.train_mode == 'WORD':
            # 2.词级
            # self.vocab = preprocess.read_vocab(os.path.join('data', preprocess.WORD_VOCAB_PATH))
            self.vecs_dict = preprocess.load_vecs(os.path.join('data', preprocess.SGNS_WORD_NGRAM_PATH))

        # 测试集有标题，读取时注意跳过第一行
        dataset = TextLineDataset(os.path.join('data',preprocess.TEST_PATH))
        dataset = dataset.shuffle(preprocess.TOTAL_TEST_SIZE).batch(self.test_batch_size)

        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        return dataset, next_element




