import tensorflow as tf
from tensorflow.data.experimental import CsvDataset
from tensorflow.data import TextLineDataset
from cnn_model import TextCNN
from cnn_model import CNNConfig
from data import preprocess
import os
import datetime
import time
import numpy as np

CHECK_POINT_PATH = 'model-30001.meta'


def predict():
    """
    读取模型，预测商品标题
    :param titles: 列表，商品标题的字符串
    :return: results
    """
    # Test procedure
    # ======================================================
    with tf.Session() as sess:
        # 读取保存的模型
        checkpoint_dir = os.path.abspath("checkpoints")
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
        saver = tf.train.import_meta_graph(os.path.join(checkpoint_dir, 'model-25001.meta'))
        saver.restore(sess, checkpoint_file)
        graph = tf.get_default_graph()

        config = CNNConfig()
        cnn = TextCNN(CNNConfig)
        # 读取测试集数据
        dataset, next_element = cnn.prepare_test_data()

        # 从图中读取变量
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        prediction = graph.get_operation_by_name("output/prediction").outputs[0]
        training = graph.get_operation_by_name("training").outputs[0]

        def predict_step(batch_x):
            feed_dict = {
                input_x: batch_x,
                dropout_keep_prob: 1.0,
                training: False
            }
            pre = sess.run(prediction, feed_dict)
            return pre

        # 自定义批量查询
        # ================================================================================
        label = preprocess.read_label(os.path.join('data', preprocess.LABEL_ID_PATH))
        titles = ['特步男鞋运动鞋男2019春夏新款跑步鞋休闲鞋网面透气鞋子男士跑鞋',
                  'Nike Air Zoom Grade 气垫跑步鞋924465-002-003-300-400-001-004',
                  'NewBalance NB官方2019新款男鞋女鞋运动鞋997HCA复古休闲小白鞋',
                  '海南菠萝蜜40斤现摘现发新鲜水果包邮孕妇时令假榴莲非红心30 25',
                  '暴龙太阳镜男复古时尚蛤蟆镜女高清偏光开车驾驶墨镜可配近视眼镜',
                  '飞利浦电动剃须刀S300 S301 S330 S1010充电式全身水洗男士刮胡刀',]

        batch_x = []
        for title in titles:
            batch_x.append(preprocess.to_id(title, cnn.vocab, cnn.train_mode))
        batch_x = np.stack(batch_x)
        pre = predict_step(batch_x)
        results = [label[x] for x in pre]
        print(results)
        # =====================================================================================
        # 给测试集打标签
        # ====================================================================================
        # 跳过测试集的标题
        # sess.run(next_element)
        # i = 0
        # t1 = datetime.datetime.now()
        # while True:
        #     try:
        #         titles = sess.run(next_element)
        #         batch_x = cnn.convert_test_input(titles)
        #         predict_step(batch_x)
        #         i += 1
        #     except tf.errors.OutOfRangeError:
        #         break
#
        # t2 = datetime.datetime.now()
        # dt = (t2-t1).min
#
        # print('查询总耗时: %fmin' % dt)
        # print('平均每条耗时: %fmin' % (dt/i))
        # 450w条数据约15分钟

    # ==================================================================


if __name__ == '__main__':
    predict()






