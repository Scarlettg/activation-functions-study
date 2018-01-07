from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import scipy.io as sio
import tensorflow as tf
import numpy as np

import resnet_model

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--data_dir', type=str, default=os.path.join(os.path.dirname(__file__), '../datasets'),
                    help='The path to the SVHN data directory.')

parser.add_argument('--model_dir', type=str, default='/tmp/svhn_resnet_model',
                    help='The directory where the model will be stored.')

parser.add_argument('--resnet_size', type=int, default=32,
                    help='The size of the ResNet model to use.')

parser.add_argument('--train_epochs', type=int, default=160,
                    help='The number of epochs to train.')

parser.add_argument('--epochs_per_eval', type=int, default=1,
                    help='The number of epochs to run in between evaluations.')

parser.add_argument('--batch_size', type=int, default=128,
                    help='The number of images per batch.')

parser.add_argument('--activation', type=str, default='relu',
                    help='activation function swish,relu,lrelu,tanh,elu')

parser.add_argument(
    '--data_format', type=str, default=None,
    choices=['channels_first', 'channels_last'],
    help='A flag to override the data format used in the model. channels_first '
         'provides a performance boost on GPU but is not always compatible '
         'with CPU. If left unspecified, the data format will be chosen '
         'automatically based on whether TensorFlow was built for CPU or GPU.')

_HEIGHT = 32
_WIDTH = 32
_DEPTH = 3
_NUM_IMAGES = {'train': 73257, 'test': 26032}
_NUM_CLASSES = 10
_WEIGHT_DECAY = 2e-4
_MOMENTUM = 0.9

def get_data(is_training, data_dir):
  """Read the .mat file, do data conversions, and return
     TF dataset
  """
  if is_training:
    filename = 'train_32x32.mat'
  else:
    filename = 'test_32x32.mat'

  filepath = os.path.join(data_dir, filename)
  assert (os.path.exists(filepath))
  
  #roll the image# axis backwards to be the first axis
  data = sio.loadmat(filepath)
  X = np.rollaxis(data['X'], 3)
  y = data['y'].reshape((X.shape[0], 1))

  num_images = _NUM_IMAGES['train'] if is_training else _NUM_IMAGES['test']
  assert(X.shape[0] == num_images)

  dataset = tf.data.Dataset.from_tensor_slices((X, y))
  dataset = dataset.map(
      lambda image, label: (tf.cast(image, tf.float32), 
          tf.squeeze(
              tf.one_hot(tf.cast(label, tf.int32), _NUM_CLASSES)
      )))
  
  return dataset

def preprocess_image(is_training, image):
  if is_training:
    image = tf.image.resize_image_with_crop_or_pad(
        image, _HEIGHT + 8, _WIDTH + 8)
    
    image = tf.random_crop(image, [_HEIGHT, _WIDTH, _DEPTH])
    image = tf.image.random_flip_left_right(image)
  
  image = tf.image.per_image_standardization(image)
 
  return image

def input_fn(is_training, data_dir, batch_size, num_epochs=1):
  """input function to the network"""
  dataset = get_data(is_training, data_dir)

  if is_training:
    dataset = dataset.shuffle(buffer_size = _NUM_IMAGES['train'])

  dataset = dataset.map(
    lambda image, label: (preprocess_image(is_training, image), label))
  
  dataset = dataset.prefetch(2 * batch_size)
  #Repeat the dataset N epochs before evaluation
  dataset = dataset.repeat(num_epochs)

  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()
  images, labels = iterator.get_next()

  return images, labels

def svhn_model_fn(features, labels, mode, params):
  tf.summary.image('images', features, max_outputs=6)
  network = resnet_model.cifar10_resnet_v2_generator(
    params['resnet_size'], _NUM_CLASSES, params['data_format'], FLAGS.activation)
  
  inputs = tf.reshape(features, [-1, _HEIGHT, _WIDTH, _DEPTH])
  logits = network(inputs, mode == tf.estimator.ModeKeys.TRAIN)
  
  predictions = {
      'classes': tf.argmax(logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  #calculate loss
  cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits, onehot_labels=labels)

  #logging cross entropy
  tf.identity(cross_entropy, name='cross_entropy')
  tf.summary.scalar('cross_entropy', cross_entropy)

  #Add weight decay to the loss
  loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

  if mode == tf.estimator.ModeKeys.TRAIN:
    # Scale the learning rate linearly with the batch size. When the batch size
    # is 128, the learning rate should be 0.1.
    initial_learning_rate = 0.1 * params['batch_size'] / 128
    batches_per_epoch = _NUM_IMAGES['train'] / params['batch_size']
    global_step = tf.train.get_or_create_global_step()

    # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
    boundaries = [int(batches_per_epoch * epoch) for epoch in [80, 120]]
    values = [initial_learning_rate * decay for decay in [1, 0.1, 0.01]]
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32), boundaries, values)

    # Create a tensor named learning_rate for logging purposes
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=_MOMENTUM)

    # Batch norm requires update ops to be added as a dependency to the train_op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step)
  else:
    train_op = None

  accuracy = tf.metrics.accuracy(
      tf.argmax(labels, axis=1), predictions['classes'])
  metrics = {'accuracy': accuracy}

  # Create a tensor named train_accuracy for logging purposes
  tf.identity(accuracy[1], name='train_accuracy')
  tf.summary.scalar('train_accuracy', accuracy[1])

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)

def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  # Set up a RunConfig to only save checkpoints once per training cycle.
  run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9)
  svhn_classifier = tf.estimator.Estimator(
      model_fn=svhn_model_fn, model_dir=FLAGS.model_dir, config=run_config,
      params={
          'resnet_size': FLAGS.resnet_size,
          'data_format': FLAGS.data_format,
          'batch_size': FLAGS.batch_size,
      })

  for _ in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
    tensors_to_log = {
        'learning_rate': 'learning_rate',
        'cross_entropy': 'cross_entropy',
        'train_accuracy': 'train_accuracy'
    }

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)

    svhn_classifier.train(
        input_fn=lambda: input_fn(
            True, FLAGS.data_dir, FLAGS.batch_size, FLAGS.epochs_per_eval),
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_results = svhn_classifier.evaluate(
        input_fn=lambda: input_fn(False, FLAGS.data_dir, FLAGS.batch_size))
    print(eval_results)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)

