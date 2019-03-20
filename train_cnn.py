# coding=utf-8
import sklearn.metrics as metrics
import sklearn as sk
import tensorflow as tf
from cnn_model import TextCNN
from cnn_model import CNNConfig
import datetime
import time
import os

def train():
    # Training procedure
    # ======================================================
    # 设定最小显存使用量
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        config = CNNConfig('MULTI')
        cnn = TextCNN(config)
        cnn.prepare_data()
        cnn.setCNN()

        print('Setting Tensorboard and Saver...')
        # 设置Saver和checkpoint来保存模型
        # ===================================================
        checkpoint_dir = os.path.join(os.path.abspath("checkpoints"), "textcnn")
        checkpoint_prefix = os.path.join(checkpoint_dir, cnn.train_mode)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())
        # =====================================================

        # 配置Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
        # ====================================================================
        train_tensorboard_dir = 'tensorboard/textcnn/train/' + config.train_mode
        valid_tensorboard_dir = 'tensorboard/textcnn/valid/' + config.train_mode
        if not os.path.exists(train_tensorboard_dir):
            os.makedirs(train_tensorboard_dir)
        if not os.path.exists(valid_tensorboard_dir):
            os.makedirs(valid_tensorboard_dir)

        # 训练结果记录
        log_file = open(valid_tensorboard_dir+'/log.txt', mode='w')

        merged_summary = tf.summary.merge([tf.summary.scalar('loss', cnn.loss),
                                            tf.summary.scalar('accuracy', cnn.accuracy)])

        train_summary_writer = tf.summary.FileWriter(train_tensorboard_dir, sess.graph)
        # =========================================================================

        global_step = tf.Variable(0, trainable=False)

        # 保证Batch normalization的执行
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):  # 保证train_op在update_ops执行之后再执行。
            train_op = tf.train.AdamOptimizer(config.learning_rate).minimize(cnn.loss, global_step)

        # 训练步骤
        def train_step(batch_x, batch_y, keep_prob=config.dropout_keep_prob):
            feed_dict = {
                cnn.input_x: batch_x,
                cnn.labels: batch_y,
                cnn.dropout_keep_prob: keep_prob,
                cnn.training: True
            }
            sess.run(train_op, feed_dict=feed_dict)
            step, loss, accuracy, summery = sess.run(
                [global_step, cnn.loss, cnn.accuracy, merged_summary],
                feed_dict={cnn.input_x: batch_x,
                cnn.labels: batch_y,
                cnn.dropout_keep_prob: 1.0,
                cnn.training: False})
            t = datetime.datetime.now().strftime('%m-%d %H:%M')
            print('%s: epoch: %d, step: %d, loss: %f, accuracy: %f' % (t, epoch, step, loss, accuracy))
            # 把结果写入Tensorboard中
            train_summary_writer.add_summary(summery, step)

        # 验证步骤
        def valid_step(next_valid_element):
            # 把valid_loss和valid_accuracy归0
            valid_loss = 0.0
            valid_accuracy = 0.0
            valid_precision = 0.0
            valid_recall = 0.0
            valid_f1_score = 0.0
            i = 0
            while True:
                try:
                    lines = sess.run(next_valid_element)
                    batch_x, batch_y = cnn.convert_input(lines)
                    feed_dict = {
                        cnn.input_x: batch_x,
                        cnn.labels: batch_y,
                        cnn.dropout_keep_prob: 1.0,
                        cnn.training: False
                    }
                    loss, accuracy, prediction, y_true = sess.run([cnn.loss, cnn.accuracy, cnn.prediction, cnn.labels], feed_dict)

                    precision = sk.metrics.precision_score(y_true=y_true, y_pred=prediction, average='weighted')
                    recall = sk.metrics.recall_score(y_true=y_true, y_pred=prediction, average='weighted')
                    f1_score = sk.metrics.f1_score(y_true=y_true, y_pred=prediction, average='weighted')

                    valid_loss += loss
                    valid_accuracy += accuracy
                    valid_precision += precision
                    valid_recall += recall
                    valid_f1_score += f1_score
                    i += 1

                except tf.errors.OutOfRangeError:
                    # 遍历完验证集，然后对loss和accuracy求平均值
                    valid_loss /= i
                    valid_accuracy /= i
                    valid_precision /= i
                    valid_recall /= i
                    valid_f1_score /= i

                    t = datetime.datetime.now().strftime('%m-%d %H:%M')
                    log = '%s: epoch %d, validation loss: %0.6f, accuracy: %0.6f' % (
                        t, epoch, valid_loss, valid_accuracy)
                    log = log + '\n' + ('precision: %0.6f, recall: %0.6f, f1_score: %0.6f' % (
                        valid_precision, valid_recall, valid_f1_score))
                    print(log)
                    log_file.write(log+'\n')
                    time.sleep(3)
                    return

        print('Start training TextCNN, training mode='+cnn.train_mode)
        sess.run(tf.global_variables_initializer())

        # Training loop
        for epoch in range(config.epoch_num):
            train_init_op, valid_init_op, next_train_element, next_valid_element = cnn.shuffle_datset()
            sess.run(train_init_op)
            while True:
                try:
                    lines = sess.run(next_train_element)
                    batch_x, batch_y = cnn.convert_input(lines)
                    train_step(batch_x, batch_y, config.dropout_keep_prob)
                except tf.errors.OutOfRangeError:
                    # 初始化验证集迭代器
                    sess.run(valid_init_op)
                    # 计算验证集准确率
                    valid_step(next_valid_element)
                    break

        train_summary_writer.close()
        log_file.close()
        # 训练完成后保存参数
        path = saver.save(sess, checkpoint_prefix, global_step=global_step)
        print("Saved model checkpoint to {}\n".format(path))
    # ==================================================================


if __name__ == '__main__':
    train()






