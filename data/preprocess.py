import csv
import cut
import numpy as np
import tensorflow as tf
from tensorflow.data.experimental import CsvDataset
import datetime
import collections
import re

SGNS_WORD_NGRAM_PATH = 'sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5'
SGNS_WORD_PATH = ''
LABEL_ID_PATH = './data/level3_id.txt'
TRAIN_PATH = './data/train.csv'
TRAIN_WITH_ID_PATH = './data/train_with_id.csv'
CHAR_VOCAB_PATH = './data/char_vocab.txt'
WORD_VOCAB_PATH = './data/word_vocab.txt'
TEMP_PATH = './data/temp.csv'

TOTAL_SIZE = 500000
TRAIN_SIZE = int(TOTAL_SIZE * 0.7)
VALID_SIZE = int(TOTAL_SIZE * 0.3)

MAX_TEXT_LENGTH = 86

word_vecs = {}


# 给标签分配id
def assign_id():
    readfile = open('./data/level3_stat.txt', 'r')
    writefile = open('./data/level3_id.txt', 'w', encoding='utf-8')

    line = ''
    ID = 0
    label = ''
    while(True):
        line = readfile.readline()
        if line == '':
            break
        labelwords = line.split()[0:-1]     # 获取标签(个别标签出现了空格，所以特殊处理)
        for i in range(len(labelwords)):
            label += labelwords[i]
            if i == len(labelwords) - 1:
                break   # 标签最后不加空格
            label += ' '
        writefile.write(label+' '+str(ID)+'\n')
        ID += 1
        label = ''

    readfile.close()
    writefile.close()

    # ----------------------------------------------------------------------

    # 重新生成tag为id的训练集
    ids = {}
    label = ''
    ID = 0
    line = ''

    with open(LABEL_ID_PATH, 'r', encoding='utf-8') as f:
        print('Reading tag id file...')
        while (True):
            line = f.readline()
            label = ''
            if line == '':
                break
            labelwords = line.split()[0:-1]  # 获取标签(个别标签出现了空格，所以特殊处理)
            for i in range(len(labelwords)):
                label += labelwords[i]
                if i == len(labelwords) - 1:
                    break  # 标签最后不加空格
                label += ' '
            ID = line.split()[-1]  # 获取ID
            ids[label] = ID

    writefile = open(TRAIN_WITH_ID_PATH, 'w', newline='')
    writer = csv.writer(writefile)
    readfile = open(TRAIN_PATH, 'r')
    reader = csv.reader(readfile)
    next(reader)
    print('Writing traing file with id...')

    for row in reader:
        title = row[0]
        tag = row[1]
        try:
            writer.writerow([title, ids[tag]])
        except KeyError as e:
            print('KeyError occur!', title, tag)
    writefile.close()
    readfile.close()


def load_vecs(fname):
    """
    加载词向量文件。\n
    :return: 加载好的词向量dict
    """
    line = ''
    # 加载词向量表
    with open(fname, 'r', encoding='utf-8') as f:
        print('Reading word vectors file...')
        info = f.readline().split()  # 跳过第一行
        print('Total:', info[0], 'dimension:', info[1])
        starttime = datetime.datetime.now()
        while (True):
            line = f.readline()
            if line == '':
                break
            linewords = line.split()
            word = linewords[0]  # 词
            vec = linewords[1:]
            for i in range(len(vec)):
                vec[i] = float(vec[i])
            vec = np.asarray(vec, dtype=np.float32)  # 向量
            word_vecs[word] = vec  # 保存到vecs中
        endtime = datetime.datetime.now()
        dt = endtime - starttime
        print('Reading succeed. Total time:', str(int(dt.seconds / 60)) + 'min' + str(dt.seconds) + 'sec')


def add_word(word):
    # If a word isn't in the vocabulary, assign a random vector to it.
    # Referring to https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    # TODO: Uncertain method to generate random vector
    vec =  np.random.uniform(-0.25, 0.25, [300])
    word_vecs[word] = vec
    return vec


def get_word_vecs(string):
    # Return a list of word vectors of a string.
    # Note: The string read from this .csv file must be decoded with 'gbk' first
    words = cut.cut_and_filter(string)

    # TODO: Pad the string with spaces to fix text_length
    if len(words) < MAX_TEXT_LENGTH:
        for _ in range(MAX_TEXT_LENGTH - len(words)):
            words.append('<PAD>')

    vec = []
    for word in words:
        if word not in word_vecs:
            add_word(word)
        vec.append(word_vecs[word])
    return np.asarray(vec, np.float32)


def get_max_text_length(fname):
    """
    查看训练集文本的最长长度
    :param fname:
    :return:
    """
    rf = open(fname, 'r')
    reader = csv.reader(rf)

    max_len = 0
    for row in reader:
        # 只保留中文
        title = re.sub(r'[^\u4e00-\u9fa5]', '', row[0]).strip()
        if len(title) > max_len:
            max_len = len(title)

    return max_len

# 以下代码参考自 https://github.com/gaussic/text-classification-cnn-rnn
# ====================================================================
def build_vocab(train_path, vocab_path, vocab_size=4000):
    """
    根据训练集构建词汇表，保存为文件（1.字符级词汇表 2.词级词汇表），并分配id

    :param train_path:
    :param vocab_path:
    :param vocab_size:
    :return:
    """
    rf = open(train_path, 'r', encoding='gbk')
    reader = csv.reader(rf)
    wf = open(vocab_path, 'w', encoding='utf-8')

    # 1.字符级词汇表
    all_chars = []
    for row in reader:
        # 只保留中文
        title = re.sub(r'[^\u4e00-\u9fa5]', '', row[0]).strip()
        # 将字符存入列表
        chars = list(title)
        all_chars.extend(chars)

    # Counter类继承了dict，统计列表每个元素出现的次数
    counter = collections.Counter(all_chars)
    # 保存出现次数最多的vocab_size - 1个字
    count_pairs = counter.most_common(vocab_size - 1)
    chars, counts = list(zip(*count_pairs))
    # '<PAD>'用来填充每个文本至定长
    chars = ['<PAD>'] + list(chars)

    for i in range(len(chars)):
        wf.write(chars[i] + ' ' + str(i) + '\n')

    rf.close()
    wf.close()


def read_vocab(vocab_path):
    """
    读取词汇表文件，转换为{词：id}表示

    :param vocab_path:
    :return vocab:
    """
    vocab = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if line == '':
                break
            word, id = line.strip().split()
            vocab[word] = int(id)
    return vocab


def read_label(label_ids_path):
    """
    读取标签文件，转化为{标签: id}表示

    :param label_ids_path:
    :return label:
    """


def to_id(content, vocab):
    """
    将数据集从文字转换为固定长度的id序列表示
    :param content:
    :param vocab:
    :return:
    """

    title_id = [vocab[x] for x in content if x in vocab]

    # 将文本pad为固定长度
    if len(title_id) < MAX_TEXT_LENGTH:
        padding = [0 for _ in range(MAX_TEXT_LENGTH - len(title_id))]
        title_id.extend(padding)

    return title_id
# ===========================================================================


if __name__ == '__main__':
    #dataset = CsvDataset(
    #    TRAIN_WITH_ID_PATH,
    #    [tf.string, tf.int32],
    #).batch(32)
#
    #iterator = dataset.make_initializable_iterator()
    #next_element = iterator.get_next()
    #with tf.Session() as sess:
    #    sess.run(iterator.initializer)
    #    titles, labels = sess.run(next_element)
    #    batch_x = []
    #    for title in titles:
    #        vecs = get_word_vecs(title)
    #        batch_x.append(vecs)
    #    batch_x = np.stack(batch_x)
    #    batch_x = tf.constant(batch_x)
    #    labels = tf.one_hot(labels, depth=1258)
    #    print(batch_x.shape)
    #    print(labels.eval())

    #build_vocab(TRAIN_WITH_ID_PATH, CHAR_VOCAB_PATH)
    #vocab = read_vocab(CHAR_VOCAB_PATH)
    #print(to_id('你好啊', vocab))
    print(get_max_text_length(TRAIN_WITH_ID_PATH))