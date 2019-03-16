import csv
from data import cut
from tensorflow.data.experimental import CsvDataset
import numpy as np
import tensorflow as tf
import datetime
import collections
import re
import os

SGNS_WORD_NGRAM_PATH = 'sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5'
SGNS_WORD_PATH = 'sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5'
LABEL_ID_PATH = 'level3_id.txt'
TRAIN_PATH = 'train.csv'
TRAIN_WITH_ID_PATH = 'train_with_id.csv'
TEST_PATH = 'test.tsv'
CHAR_VOCAB_PATH = 'char_vocab.txt'
WORD_VOCAB_PATH = 'word_vocab.txt'
TEMP_PATH = 'temp.csv'

TOTAL_TRAIN_SIZE = 500000
TRAIN_SIZE = int(TOTAL_TRAIN_SIZE * 0.7)
VALID_SIZE = int(TOTAL_TRAIN_SIZE * 0.3)

TOTAL_TEST_SIZE = 4500000

# 字符级的文本最长长度
# 平均样本长度为29.95003
MAX_CHAR_TEXT_LENGTH = 35
# 词级的文本最长长度
# 平均样本长度为15.200604
MAX_WORD_TEXT_LENGTH = 20

VOCAB_SIZE = 4000

vec_dim = 300       # 预训练词向量的维度


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


def recreate_data_with_id_label(train_path, train_with_id_path):
    # 重新生成标签为id的训练集
    ids = {}
    label = ''
    ID = 0
    line = ''

    with open(LABEL_ID_PATH, 'r', encoding='utf-8') as f:
        print('Reading tag id file...')
        while True:
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

    wf = open(train_with_id_path, 'w', newline='', encoding='gbk')
    rf = open(train_path, 'r', encoding='gbk', errors='ignore')
    print('Writing traing file with id label...')

    while True:
        line = rf.readline()
        if line == '':
            break
        line = line.strip().split('\t')
        title = ''.join(line[0:-1])
        tag = ''.join(line[-1])
        try:
            wf.write(','.join([title, ids[tag]]) + '\n')
        except KeyError as e:
            print('KeyError occur!', title, tag)
        except IndexError as e:
            print('IndexError occur!', line)

    wf.close()
    rf.close()


# def recreate_data_with_id_title():
#     """
#     重新生成标题为id的训练集，转换成Deep Learning Studio规定的训练集csv格式
#     文本为分号分隔的字符id，标签为id
#     即:1;0;0;5;555;999;888;777,1
#     """
#     vocab = read_vocab(CHAR_VOCAB_PATH)
#     writefile = open(TEMP_PATH, 'w', newline='')
#     writer = csv.writer(writefile)
#     readfile = open(TRAIN_WITH_ID_PATH, 'r')
#     reader = csv.reader(readfile)
#     print('Writing training file with id title...')
#
#     for row in reader:
#         id_title = [str(x) for x in to_id(row[0], vocab, 'CHAR')]
#         id_title = ';'.join(id_title)
#         label = row[1]
#         writer.writerow([id_title, label])
#
#     writefile.close()
#     readfile.close()
#
#     os.rename(TEMP_PATH, 'train_id_with_id.csv')


def load_vecs(fname=SGNS_WORD_NGRAM_PATH):
    """
    加载词向量文件。\n
    :return: 加载好的词向量dict
    """
    vecs_dict = {}
    # 加载词向量表
    with open(fname, 'r', encoding='utf-8') as f:
        print('Reading word vectors file...')
        info = f.readline().split()  # 跳过第一行
        print('Total:', info[0], 'dimension:', info[1])

        starttime = datetime.datetime.now()

        while True:
            line = f.readline()
            if line == '':
                break
            linewords = line.split()
            word = linewords[0]  # 词
            vec = [float(x) for x in linewords[1:]]
            vec = np.asarray(vec, dtype=np.float32)  # 向量
            vecs_dict[word] = vec  # 保存到word_vecs中

        endtime = datetime.datetime.now()
        dt = endtime - starttime
        print('Reading succeed. Total time:', str(int(dt.seconds / 60)) + 'min' + str(dt.seconds) + 'sec')

    return vecs_dict


def add_word(word, vecs_dict):
        # 如果出现预训练词向量中没有的词，随机生成词向量
        # Referring to https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        vec = np.random.uniform(-0.25, 0.25, [300])
        vecs_dict[word] = vec
        return vec


def get_word_vecs(string, vecs_dict):
    # 将一个字符串转换为n*300的词向量列表

    words = cut.cut_and_filter(string.strip())

    # Pad the string with spaces to fix text_length
    if len(words) < MAX_WORD_TEXT_LENGTH:
        # 文本小于max_length, 将文本pad为固定长度
        padding = ['<PAD>' for _ in range(MAX_WORD_TEXT_LENGTH - len(words))]
        words.extend(padding)
    elif len(words) > MAX_WORD_TEXT_LENGTH:
        # 文本大于max_length，将文本裁剪
        words = words[:MAX_WORD_TEXT_LENGTH]

    vecs = []
    for word in words:
        if word not in vecs_dict:
            add_word(word, vecs_dict)
        vecs.append(vecs_dict[word])
    return np.asarray(vecs, np.float32)


def get_average_text_length(fname):
    """
    查看训练集文本的最长长度
    :param fname:
    :return:
    """
    rf = open(fname, 'r')
    reader = csv.reader(rf)

    length_sum = 0
    i = 0
    for row in reader:
        # 1.字符长度
        # title = re.sub(r'[^\u4e00-\u9fa5]', '', row[0]).strip()
        # 2.词长度
        title = cut.cut_and_filter(row[0].strip())
        print(title)
        length_sum += len(title)
        i += 1
    print('total sample number:', i)
    return length_sum/i


# 以下代码参考自 https://github.com/gaussic/text-classification-cnn-rnn
# ====================================================================
def build_vocab(train_path, vocab_path, vocab_size=VOCAB_SIZE):
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

    all_chars = []
    for row in reader:
        # 1.字符级词汇表
        # 只保留中文
        # title = re.sub(r'[^\u4e00-\u9fa5]', '', row[0]).strip()
        # 2.词级词汇表
        title = cut.cut_and_filter(row[0].strip())
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
    print('Reading vocabulary from:', vocab_path)
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
    读取标签文件，转化为列表表示

    :param label_ids_path:
    :return labels:
    """
    print('Reading label id from:', label_ids_path)
    labels = []
    with open(label_ids_path, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline().strip()
            if line == '':
                break
            line = line.split()
            label = ''.join(line[:-1])
            labels.append(label)
    return labels


def to_id(content, vocab, mode='CHAR-RANDOM'):
    """
    将数据集从文字转换为固定长度的id序列表示
    """

    title_id = [vocab[x] for x in content if x in vocab]

    max_length = 0
    if mode == 'CHAR-RANDOM':
        max_length = MAX_CHAR_TEXT_LENGTH
    elif mode == 'WORD-NON-STATIC' or mode == 'MULTI':
        max_length = MAX_WORD_TEXT_LENGTH

    if len(title_id) < max_length:
        # 文本小于max_length, 将文本pad为固定长度
        padding = [vocab['<PAD>'] for _ in range(max_length - len(title_id))]
        title_id.extend(padding)
    elif len(title_id) > max_length:
        # 文本大于max_length，将文本裁剪
        title_id = title_id[:max_length]

    return title_id
# ===========================================================================


if __name__ == '__main__':
    # with tf.Session() as sess:
    #     titles, labels = sess.run(next_element)
    #     batch_x = []
    #     for title in titles:
    #         t = cut.cut_and_filter(title.decode('gbk'))
    #         batch_x.append(get_word_vecs(title, vecs_dict))
    #     batch_x = np.stack(batch_x)
    #     batch_x = tf.constant(batch_x)
    #     print(batch_x.shape)
    # print(to_id('ansevi(安视威) IC卡/M1卡/门禁卡/考勤卡/异形卡 蓝色IC方牌', vocab, 'CHAR'))
    #print(get_average_text_length(TRAIN_WITH_ID_PATH))
    #recreate_data_with_id_label('holdout37\\3.tsv', 'holdout37\\train_with_id_3.csv')
    #recreate_data_with_id_label('holdout37\\7.tsv', 'holdout37\\train_with_id_7.csv')
    print(str(['1', '2']))

