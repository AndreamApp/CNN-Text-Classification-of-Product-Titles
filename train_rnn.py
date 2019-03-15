import tensorflow as tf
from rnn_model import RNNConfig
from rnn_model import TextRNN
import os
import datetime
import time


def train():
    # Training procedure
    # ======================================================
    with tf.Session() as sess:
        config = RNNConfig()
        rnn = TextRNN(config)
        train_dataset, valid_dataset, train_init_op, valid_init_op, next_train_element, next_valid_element = rnn.prepare_data()
        rnn.setRNN()

        print('Setting Tensorboard and Saver...')
        # 设置Saver和checkpoint来保存模型
        # ===================================================
        checkpoint_dir = os.path.join(os.path.abspath("checkpoints"), "textrnn")
        checkpoint_prefix = os.path.join(checkpoint_dir, rnn.train_mode)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())
        # =====================================================

        # 配置Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
        # ====================================================================
        train_tensorboard_dir = 'tensorboard/textrnn/train/' + config.train_mode
        valid_tensorboard_dir = 'tensorboard/textrnn/valid/' + config.train_mode
        if not os.path.exists(train_tensorboard_dir):
            os.makedirs(train_tensorboard_dir)
        if not os.path.exists(valid_tensorboard_dir):
            os.makedirs(valid_tensorboard_dir)

        merged_summary = tf.summary.merge([tf.summary.scalar('loss', rnn.loss),
                                            tf.summary.scalar('accuracy', rnn.accuracy)])

        merged_valid_summary = tf.summary.merge([tf.summary.scalar('valid_loss', rnn.valid_loss),
                                                 tf.summary.scalar('valid_accuracy', rnn.valid_accuracy)])

        train_summary_writer = tf.summary.FileWriter(train_tensorboard_dir, sess.graph)
        valid_summary_writer = tf.summary.FileWriter(valid_tensorboard_dir, sess.graph)
        # =========================================================================

        global_step = tf.Variable(0, trainable=False)

        # 保证Batch normalization的执行
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):  # 保证train_op在update_ops执行之后再执行。
            train_op = tf.train.AdamOptimizer(config.learning_rate).minimize(rnn.loss, global_step)

        # 训练步骤
        def train_step(batch_x, batch_y, keep_prob=config.dropout_keep_prob):
            feed_dict = {
                rnn.input_x: batch_x,
                rnn.labels: batch_y,
                rnn.dropout_keep_prob: keep_prob,
                rnn.training: True
            }
            sess.run(train_op, feed_dict=feed_dict)
            step, loss, accuracy, summery = sess.run(
                [global_step, rnn.loss, rnn.accuracy, merged_summary],
                feed_dict={rnn.input_x: batch_x,
                rnn.labels: batch_y,
                rnn.dropout_keep_prob: 1.0,
                rnn.training: False})
            time = datetime.datetime.now().isoformat()
            print('%s: step: %d, loss: %f, accuracy: %f' % (time, step, loss, accuracy))
            # 把结果写入Tensorboard中
            train_summary_writer.add_summary(summery, step)

        # 验证步骤
        def valid_step(next_valid_element):
            # 把valid_loss和valid_accuracy归零
            sess.run(tf.assign(rnn.valid_loss, 0.0))
            sess.run(tf.assign(rnn.valid_accuracy, 0.0))
            i = 0
            while True:
                try:
                    lines = sess.run(next_valid_element)
                    batch_x, batch_y = rnn.convert_input(lines)
                    feed_dict = {
                        rnn.input_x: batch_x,
                        rnn.labels: batch_y,
                        rnn.dropout_keep_prob: 1.0,
                        rnn.training: False
                    }
                    loss, accuracy = sess.run([rnn.loss, rnn.accuracy], feed_dict)
                    # cnn.valid_loss += loss
                    sess.run(rnn.valid_loss.assign_add(loss))
                    # cnn.valid_accuracy += accuracy
                    sess.run(rnn.valid_accuracy.assign_add(accuracy))
                    i += 1

                except tf.errors.OutOfRangeError:
                    # 遍历完验证集，然后对loss和accuracy求平均值
                    # cnn.valid_loss /= i
                    sess.run(rnn.valid_loss.assign(tf.math.divide(rnn.valid_loss, i)))
                    # cnn.valid_accuracy /= i
                    sess.run(rnn.valid_accuracy.assign(tf.math.divide(rnn.valid_accuracy, i)))
                    step, valid_loss, valid_accuracy, valid_summary = sess.run([global_step, rnn.valid_loss,
                                                                                rnn.valid_accuracy,
                                                                                merged_valid_summary], feed_dict)
                    print('Validation loss: %f, accuracy: %f' % (valid_loss, valid_accuracy))
                    time.sleep(3)
                    # 把结果写入Tensorboard中
                    valid_summary_writer.add_summary(valid_summary, step)
                    break
        print('Start training RNN, training mode='+rnn.train_mode)
        sess.run(tf.global_variables_initializer())

        # 初始化训练集、验证集迭代器
        sess.run(train_init_op)

        # Training loop
        for epoch in range(config.epoch_num):
            lines = sess.run(next_train_element)
            batch_x, batch_y = rnn.convert_input(lines)
            train_step(batch_x, batch_y, config.dropout_keep_prob)
            if epoch % config.valid_per_batch == 0:
                # 重新初始化验证集迭代器
                sess.run(valid_init_op)
                # 计算验证集准确率
                valid_step(next_valid_element)
        train_summary_writer.close()
        valid_summary_writer.close()

        # 训练完成后保存参数
        path = saver.save(sess, checkpoint_prefix, global_step=global_step)
        print("Saved model checkpoint to {}\n".format(path))
    # ==================================================================


if __name__ == '__main__':
    train()







