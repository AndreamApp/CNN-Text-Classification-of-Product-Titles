# coding=utf-8
import tensorflow as tf
from data import preprocess
import os
import datetime
import numpy as np


class Predictor(object):
    def __init__(self):
        self.model = 'bilstm'
        self.pred_mode = 'WORD-NON-STATIC'
        self.state = 'created' # created/loading/prepared/predicting

        self.sess = None
        self.graph = None
        self.input_x = None
        self.input_y = None
        self.dropout_keep_prob = None
        self.prediction = None
        self.training = None
        self.vocab = None
        self.label = None

    def setModel(self, model, pred_mode):
        if model != self.model or pred_mode != self.pred_mode:
            self.model = model
            self.pred_mode = pred_mode
            if self.sess:
                self.sess.close()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.graph = tf.Graph()
            self.sess = tf.Session(graph=self.graph, config=config)
            self.initModel()
    
    def initModel(self):
        print('Loading model:', self.model, 'pred_mode:', self.pred_mode)
        checkpoint_file = self.__getCkptfile(self.model, self.pred_mode)
        if not checkpoint_file:
            return

        # 加载模型，这要加.meta后缀
        with self.graph.as_default():
            saver = tf.train.import_meta_graph(checkpoint_file + '.meta')
            saver.restore(self.sess, checkpoint_file)
        # self.graph = tf.get_default_graph()
        # 从图中读取变量
        self.input_x = self.graph.get_operation_by_name("input_x").outputs[0]
        self.input_y = self.graph.get_operation_by_name("input_y").outputs[0]
        self.dropout_keep_prob = self.graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        self.prediction = self.graph.get_operation_by_name("output/prediction").outputs[0]
        self.training = self.graph.get_operation_by_name("training").outputs[0]
        
        # 加载词向量
        if self.pred_mode == 'CHAR-RANDOM':
            self.vocab = preprocess.read_vocab(os.path.join('data', preprocess.CHAR_VOCAB_PATH))
        elif self.pred_mode == 'WORD-NON-STATIC' or self.pred_mode == 'MULTI':
            self.vocab = preprocess.read_vocab(os.path.join('data', preprocess.WORD_VOCAB_PATH))
        # 加载标签
        self.label = preprocess.read_label(os.path.join('data', preprocess.LABEL_ID_PATH))

    def predict(self, titles):
        # 自定义批量查询
        # ================================================================================
        batch_x = []
        # 1.id
        for title in titles:
            batch_x.append(preprocess.to_id(title, self.vocab, self.pred_mode))
        batch_x = np.stack(batch_x)
        pre = self.predictStep(batch_x)
        results = [self.label[x] for x in pre]
        final_results = list(zip(titles, results))
        return final_results
    
    def predictStep(self, batch_x):
        feed_dict = {
            self.input_x: batch_x,
            self.dropout_keep_prob: 1.0,
            self.training: False
        }
        pre = self.sess.run(self.prediction, feed_dict)
        return pre

    def __getCkptfile(self, model, pred_mode):
        checkpoint_file = ''
        # 模型的文件名放在这，不含后缀
        if model == 'textcnn':
            if pred_mode == 'CHAR-RANDOM':
                checkpoint_file = os.path.abspath("checkpoints/textcnn/CHAR-RANDOM-62905")
            elif pred_mode == 'WORD-NON-STATIC':
                checkpoint_file = os.path.abspath("checkpoints/textcnn/WORD-NON-STATIC-76580")
            elif pred_mode == 'MULTI':
                checkpoint_file = os.path.abspath("checkpoints/textcnn/MULTI-82050")
        elif model == 'bilstm':
            if pred_mode == 'CHAR-RANDOM':
                checkpoint_file = os.path.abspath("checkpoints/bilstm/CHAR-RANDOM-68375")
            elif pred_mode == 'WORD-NON-STATIC':
                checkpoint_file = os.path.abspath("checkpoints/bilstm/WORD-NON-STATIC-76580")
            elif pred_mode == 'MULTI':
                print('model', model, 'does not have pred_mode', pred_mode)
        elif model == 'textrnn':
            if pred_mode == 'CHAR-RANDOM':
                checkpoint_file = os.path.abspath("checkpoints/textrnn/CHAR-RANDOM-68375")
            elif pred_mode == 'WORD-NON-STATIC':
                checkpoint_file = os.path.abspath("checkpoints/textrnn/WORD-NON-STATIC-76580")
            elif pred_mode == 'MULTI':
                print('model', model, 'does not have pred_mode', pred_mode)
        return checkpoint_file


if __name__ == '__main__':
    model = ['textcnn', 'textrnn', 'bilstm']
    pred_mode = ['CHAR-RANDOM', 'WORD-NON-STATIC', 'MULTI']
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
    predictor = Predictor()
    predictor.setModel(model[0], pred_mode[0])
    results = predictor.predict(titles)
    print(results)






