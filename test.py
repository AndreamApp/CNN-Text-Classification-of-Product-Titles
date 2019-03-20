# coding=utf-8
import sklearn as sk
import sklearn.metrics as metrics
import tensorflow as tf
from bilstm_model import BiLSTM
from bilstm_model import BiLSTMConfig
from cnn_model import TextCNN
from cnn_model import CNNConfig
from rnn_model import TextRNN
from rnn_model import RNNConfig
from tensorflow.data import TextLineDataset
from data import preprocess
import os
import datetime
import numpy as np



def predict():
    """
    读取模型，预测商品标题
    :param titles: 列表，商品标题的字符串
    :return: results
    """
    # Test procedure
    # ======================================================
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # TODO: 读取不同模型，修改此处参数
        # 要读取的模型路径
        checkpoint_dir = os.path.abspath("checkpoints/textcnn")
        # 模型的文件名放在这，不含后缀
        checkpoint_file = os.path.join(checkpoint_dir, "MULTI-71110")
        # 这要加.meta后缀
        saver = tf.train.import_meta_graph(os.path.join(checkpoint_dir, 'MULTI-71110.meta'))
        saver.restore(sess, checkpoint_file)
        graph = tf.get_default_graph()

        # 这里的train_mode参数要和模型一致
        config = CNNConfig('MULTI')
        cnn = TextCNN(config)
        # 读取测试集及词汇表数据
        dataset, next_element = cnn.prepare_test_data()

        # 从图中读取变量
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        input_y = graph.get_operation_by_name("input_y").outputs[0]
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

        def customize_predict(titles):
            # 自定义批量查询
            # ================================================================================
            label = preprocess.read_label(os.path.join('data', preprocess.LABEL_ID_PATH))
            batch_x = []
            if cnn.train_mode == 'CHAR-RANDOM' or cnn.train_mode == 'WORD-NON-STATIC':
                # 1.id
                for title in titles:
                    batch_x.append(preprocess.to_id(title, cnn.vocab, cnn.train_mode))
            batch_x = np.stack(batch_x)
            pre = predict_step(batch_x)
            results = [label[x] for x in pre]
            final_results = list(zip(titles, results))
            print(final_results)
            # ====================================================================================

        def predict_test_set():
            # 给测试集打标签
            # ====================================================================================
            # 跳过测试集的标题
            sess.run(next_element)
            i = 0
            t1 = datetime.datetime.now()
            while True:
                try:
                    titles = sess.run(next_element)
                    batch_x = cnn.convert_test_input(titles)
                    predict_step(batch_x)
                    i += 1
                except tf.errors.OutOfRangeError:
                    break

            t2 = datetime.datetime.now()
            dt = (t2-t1).min
            print('查询总耗时: %fmin' % dt)
            print('平均每条耗时: %fmin' % (dt/i))
            # 450w条数据约15分钟
            # ==================================================================

        def predict_train_set():
            # 查看训练集的表现
            dataset = TextLineDataset(os.path.join('data', preprocess.TRAIN_WITH_ID_PATH)).batch(cnn.valid_batch_size)
            next_element = dataset.make_one_shot_iterator().get_next()

            valid_precision = 0.0
            valid_recall = 0.0
            valid_f1_score = 0.0
            i = 0
            while True:
                try:
                    titles = sess.run(next_element)
                    batch_x, batch_y = cnn.convert_input(titles)
                    feed_dict = {
                        input_x: batch_x,
                        input_y: batch_y,
                        dropout_keep_prob: 1.0,
                        training: False
                    }
                    y_true, y_pred = sess.run([input_y, prediction], feed_dict)
                    precision = sk.metrics.precision_score(y_true=y_true, y_pred=y_pred, average='weighted')
                    recall = sk.metrics.recall_score(y_true=y_true, y_pred=y_pred, average='weighted')
                    f1_score = sk.metrics.f1_score(y_true=y_true, y_pred=y_pred, average='weighted')

                    valid_precision += precision
                    valid_recall += recall
                    valid_f1_score += f1_score
                    i += 1

                except tf.errors.OutOfRangeError:
                    # 遍历完验证集，然后对loss和accuracy求平均值
                    valid_precision /= i
                    valid_recall /= i
                    valid_f1_score /= i

                    log = 'Training set: precision: %0.6f, recall: %0.6f, f1_score: %0.6f' % (
                        valid_precision, valid_recall, valid_f1_score)
                    print(log)
                    return

        titles = ['可莉丝汀桶装10低盐全期天然猫粮成猫幼猫粮包邮5斤2.5kg美短美毛',
                  '猫粮20斤10KG深海洋鱼味幼猫成猫老年猫 皇仕英短蓝猫天然奶糕粮5',
                  '29省包邮 皇家i27室内成猫猫粮10kg化毛球现货蓝猫美短波斯猫',
                  '猫粮成猫幼猫粮鱼肉味猫咪主粮流浪老年猫食10kg20斤5猫主粮大包',
                  '耀目 幼猫1-4-12个月成猫猫粮通用英短蓝猫深海鱼天然粮补钙20斤',
                  '金士顿 骇客神条DDR4 2400 16g台式机电脑 四代内存条 2666 单条',
                  'CRUCIAL镁光英睿达DDR4 8G 2400 2666笔记本电脑内存条联想华硕16',
                  '金士顿 骇客神条ddr4 8G 2400 2666 3000 3200 3600 4000 台式机电脑游戏内存条',
                  '瑞达威刚XPG 8G 3200 3000 2666 2400 DDR4台式机电脑RGB内存水冷灯条',
                  'AOC C27B1H 27英寸电脑电竞游戏曲面高清屏幕游戏液晶显示器27',
                  '55寸46/42/40寸三星液晶拼接屏电视墙无缝超窄边大屏幕监控显示器',
                  'HKC G271F 27英寸曲面电竞游戏显示器电脑144Hz/1ms可旋转升降',
                  '海悦源牡蛎干生蚝干500克海鲜干货海蛎干开袋即食海蛎子干海产品',
                  '【抢！再送3袋】蒜蓉粉丝扇贝肉48枚海鲜活冷冻大扇贝粉丝贝即食',
                  '新鲜大红扇贝肉鲜活冷冻海鲜水产超大鲜贝即食蒜蓉粉丝蒸扇贝1斤',
                  '原膳新西兰半壳青口贝350g 海鲜水产 海产贝类 熟冻青口贝',
                  '球队可定做！PGM 高尔夫伸缩球包 男款 多功能托运航空球包',
                  '升级版！高尔夫球练习网 专业打击笼 挥杆练习器 配推杆果岭 套装',
                  '英国DUNLOP官方正品儿童高尔夫球杆全套杆男女童初学者3至12岁',
                  '高尔夫 发球机 半自动发球机 多功能发球盒 大容量 高尔夫球设备',
                  '森马牛仔裤男修身小脚裤子黑色春季时尚弹力男士休闲男裤韩版长裤',
                  '男装 EZY DENIM牛仔裤(水洗产品) 413157 优衣库UNIQLO',
                  'JackJones杰克琼斯秋季男士时尚做旧复古浅色休闲牛仔裤长裤子',
                  '花花公子春秋季牛仔裤男士修身裤子男韩版潮流弹力加绒黑色小脚裤',
                  '琅酷 全包边休眠硅胶保护背壳 适用于ipad2/ipad3/ipad4 TPU蔷薇']
        # customize_predict(titles)
        predict_train_set()


if __name__ == '__main__':
    predict()






