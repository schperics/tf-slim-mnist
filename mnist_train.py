import tensorflow as tf
from tensorflow.contrib.slim.python.slim.data.prefetch_queue import prefetch_queue

from datasets import mnist
from net import lenet, load_batch
import time
import model_deploy

slim = tf.contrib.slim

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/tmp/mnist',
                    'Directory with the mnist data.')
flags.DEFINE_integer('batch_size', 32, 'Batch size.')
flags.DEFINE_integer('num_batches', None,
                     'Num of batches to train (epochs).')
flags.DEFINE_string('log_dir', './log/train',
                    'Directory with the log data.')
FLAGS = flags.FLAGS


def main(args):
  config = model_deploy.DeploymentConfig(num_clones=2)
  
  with tf.device(config.variables_device()):
    global_step = tf.train.get_or_create_global_step()
  
  with tf.device(config.inputs_device()):
    # load the dataset
    dataset = mnist.get_split('train', FLAGS.data_dir)
    
    # load batch of dataset
    images, labels = load_batch(dataset,
                                FLAGS.batch_size,
                                width=300,
                                height=300,
                                is_training=True)
    
    queue = prefetch_queue([images, labels], capacity=10)
  
  with tf.device(config.optimizer_device()):
    optimizer = tf.train.RMSPropOptimizer(0.001, 0.9)
  
  def model_fn(queue):
    images, labels = queue.dequeue()
    predictions = lenet(images)
    one_hot_labels = slim.one_hot_encoding(labels,
                                           dataset.num_classes)
    slim.losses.softmax_cross_entropy(predictions,
                                      one_hot_labels)
  
  model_dp = model_deploy.deploy(config, model_fn, [queue], optimizer=optimizer)
  
  sess_conf = tf.ConfigProto(allow_soft_placement=True,
                             log_device_placement=False)
  sess = tf.Session(config=sess_conf)
  tf.train.start_queue_runners(sess=sess)
  sess.run(tf.global_variables_initializer())
  s = time.time()
  for i in range(100):
    var = sess.run([model_dp.train_op, global_step])
    if i % 10 == 0:
      print("{}, {}".format(i, var))
  
  print("last : {}, {}s elapsed".format(var, time.time() - s))


if __name__ == '__main__':
  tf.app.run()
