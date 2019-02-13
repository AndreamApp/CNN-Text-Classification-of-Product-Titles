import tensorflow as tf
from cnn_model import TextCNN
from cnn_model import CNNConfig
import os
import datetime
import time


def train():
    # Training procedure
    # ======================================================
    with tf.Session() as sess:

        config = CNNConfig()
        cnn = TextCNN(config)
        train_dataset, valid_dataset, train_init_op, valid_init_op, next_train_element, next_valid_element = cnn.prepare_data()
        cnn.setCNN()

        global_step = tf.Variable(0, trainable=False)

        # 保证Batch normalization的执行
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):  # 保证train_op在update_ops执行之后再执行。
            train_op = tf.train.AdamOptimizer(config.learning_rate).minimize(cnn.loss, global_step)

        def train_step(batch_x, batch_y, keep_prob=config.dropout_keep_prob):
            feed_dict = {
                cnn.input_x: batch_x,
                cnn.labels: batch_y,
                cnn.dropout_keep_prob: keep_prob,
                cnn.training: True
            }
            sess.run(train_op, feed_dict=feed_dict)
            step, loss, accuracy = sess.run(
                [global_step, cnn.loss, cnn.accuracy],
                feed_dict={cnn.input_x: batch_x,
                cnn.labels: batch_y,
                cnn.dropout_keep_prob: 1.0,
                cnn.training: False})
            time = datetime.datetime.now().isoformat()
            print('%s: step: %d, loss: %f, accuracy: %f' % (time, step, loss, accuracy))

        def valid_step(batch_x, batch_y):
            feed_dict = {
                cnn.input_x: batch_x,
                cnn.labels: batch_y,
                cnn.dropout_keep_prob: 1.0,
                cnn.training: False
            }

            total_loss = 0.0
            total_accuracy = 0.0
            # 验证10次取平均值
            for _ in range(10):
                loss, accuracy = sess.run([cnn.loss, cnn.accuracy],
                                            feed_dict)
                total_loss += loss
                total_accuracy += accuracy
            total_loss /= 10
            total_accuracy /= 10

            print('Validation loss: %f, accuracy: %f' % (total_loss, total_accuracy))

        # 设置checkpoint来保存模型
        checkpoint_dir = os.path.abspath("checkpoints")
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        sess.run(tf.global_variables_initializer())

        # 初始化训练集、验证集迭代器
        sess.run(train_init_op)
        sess.run(valid_init_op)

        # Training loop
        for epoch in range(config.epoch_num):
            titles, labels = sess.run(next_train_element)
            batch_x, batch_y = cnn.convert_input(titles, labels)
            train_step(batch_x, batch_y, config.dropout_keep_prob)
            if epoch % 500 == 0:
                titles, labels = sess.run(next_valid_element)
                batch_x, batch_y = cnn.convert_input(titles, labels)
                valid_step(batch_x, batch_y)
                time.sleep(3)

        # 训练完成后保存参数
        path = saver.save(sess, checkpoint_prefix, global_step=global_step)
        print("Saved model checkpoint to {}\n".format(path))
    # ==================================================================


if __name__ == '__main__':
    train()







