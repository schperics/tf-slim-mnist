import tensorflow as tf
from tensorflow.contrib.slim.python.slim.data.prefetch_queue import prefetch_queue

from datasets import mnist
from model import lenet, load_batch
import time

slim = tf.contrib.slim

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/tmp/mnist',
                    'Directory with the mnist data.')
flags.DEFINE_integer('batch_size', 5, 'Batch size.')
flags.DEFINE_integer('num_batches', None,
                     'Num of batches to train (epochs).')
flags.DEFINE_string('log_dir', './log/train',
                    'Directory with the log data.')
FLAGS = flags.FLAGS


def main(args):
  # use RMSProp to optimize
  optimizer = tf.train.RMSPropOptimizer(0.001, 0.9)
  
  # load the dataset
  dataset = mnist.get_split('train', FLAGS.data_dir)
  
  # load batch of dataset
  images, labels = load_batch(dataset,
                              FLAGS.batch_size,
                              is_training=True)
  
  queue = prefetch_queue([images, labels], capacity=10)
  
  # run the image through the model
  images, labels = queue.dequeue()
  predictions = lenet(images)
  
  # get the cross-entropy loss
  one_hot_labels = slim.one_hot_encoding(labels,
                                         dataset.num_classes)
  slim.losses.softmax_cross_entropy(predictions,
                                    one_hot_labels)
  total_loss = slim.losses.get_total_loss()
  
  # create train op
  train_op = slim.learning.create_train_op(
    total_loss,
    optimizer,
    summarize_gradients=True)
  
  sess_conf = tf.ConfigProto(allow_soft_placement=True,
                             log_device_placement=False)
  sess = tf.Session(config=sess_conf)
  tf.train.start_queue_runners(sess=sess)
  sess.run(tf.global_variables_initializer())
  
  # warm up
  for _ in range(100):
    sess.run(train_op)
  
  start = time.time()
  for i in range(10000):
    val = sess.run(train_op)
    if i % 100 == 0:
      print("{} : {}".format(i, val))
  elapsed = time.time() - start
  print("last loss = {}, {} sec".format(val, elapsed))


if __name__ == '__main__':
  tf.app.run()
