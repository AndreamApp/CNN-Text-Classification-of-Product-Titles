# CNN-Text-Classification-of-Product-Title
使用TextCNN对商品名称进行分类。
使用自训练的词嵌入。

# Requirements
* Python 3
* Tensorflow 1.12以上
* Numpy

# Training dataset
训练集包含50万条样本及1258种类别。
下载链接：https://pan.baidu.com/s/1tv3yh6-H2cNTxuicJA5ykA 
提取码：qlxm 

# Project structure 
`data/`目录下包含了训练集以及字符级、词级词汇表，标签id表
`data/proprocess.py`包含一系列预处理数据的函数
`cnn_model.py`包含TextCNN模型的定义以及可调节的参数
`train.py`包含训练的代码
`test.py`包含测试的代码

# Train
    python train.py

# Referrence
CNN-RNN中文文本分类，基于TensorFlow: gaussic/text-classification-cnn-rnn https://github.com/gaussic/text-classification-cnn-rnn

dennybritz的TextCNN实现教程: http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

dennybritz/cnn-text-classification-tf: https://github.com/dennybritz/cnn-text-classification-tf

原论文: *Convolutional Neural Networks for Sentence Classification* https://github.com/yoonkim/CNN_sentence

预训练中文word2vec词向量: Embedding/Chinese-Word-Vectors https://github.com/Embedding/Chinese-Word-Vectors

jieba中文分词模块: fxsjy/jieba https://github.com/fxsjy/jieba

