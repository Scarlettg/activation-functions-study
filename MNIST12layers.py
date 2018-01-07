import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import function

mnist = input_data.read_data_sets("~/data/MNIST/", one_hot=True)

import argparse



def _swish_shape(op):
  """Shape helper function for swish and _swish_grad function below."""
  return [op.inputs[0].shape]


# Set noinline=True so that sigmoid(features) is re-computed during
# backprop, and we can free the sigmoid(features) expression immediately
# after use during the forward pass.
@function.Defun(shape_func=_swish_shape, func_name="swish_grad", noinline=True)
def _swish_grad(features, grad):
  """Gradient of Swish function defined below."""
  sigmoid_features = tf.sigmoid(features)
  activation_grad = (
      sigmoid_features * (1.0 + features * (1.0 - sigmoid_features)))
  return grad * activation_grad


@function.Defun(
    grad_func=_swish_grad,
    shape_func=_swish_shape,
    func_name="swish",
    noinline=True)
def swish(features):
  # pylint: disable=g-doc-args
  """Computes the Swish activation function: `x * sigmoid(x)`.
  Source: "Swish: a Self-Gated Activation Function" (Ramachandran et al. 2017)
  https://arxiv.org/abs/1710.05941
  Args:
    features: A `Tensor` representing preactivation values.
    name: A name for the operation (optional).
  Returns:
    The activation value.
  """
  # pylint: enable=g-doc-args
  features = tf.convert_to_tensor(features, name="features")
  return features * tf.sigmoid(features)

def _new_shape(op):
  """Shape helper function for new and _new_grad function below."""
  return [op.inputs[0].shape]

#@function.Defun(grad_func=_new_grad, shape_func=_new_shape, func_name="new", noinline=True)
#def _new_shape(op):
#  """Shape helper function for new and _new_grad function below."""
#  return [op.inputs[0].shape]

@function.Defun(shape_func=_new_shape, func_name="new_grad", noinline=True)
def _new_grad(features, grad):
  """Gradient of new function defined below."""
  activation_grad=(tf.minimum(tf.add(features, [1]) * tf.exp(features), 1))
  return grad * activation_grad

@function.Defun(
  grad_func=_new_grad,
  shape_func=_new_shape,
  func_name="new",
  noinline=True)
def new(features):
  # pylint: disable=g-doc-args
  """Computes the New activation function we created.
  Args:
  features: A Tensor representing preactivation values.
  name: A name for the operation (optional).
  Returns:
  The activation value.
  """
  # pylint: enable=g-doc-args
  features = tf.convert_to_tensor(features, name="features")
  return tf.maximum(features,features * tf.exp(-tf.abs(features)))


parser = argparse.ArgumentParser()
parser.add_argument('--activation', type=str, default='relu', help='swish/relu/elu/lrelu/tanh/new')
opt = parser.parse_args()
activation=opt.activation

actfun=None

_ACTIVATIONS = {
    None : None,
    'relu' : tf.nn.relu,
    'elu' : tf.nn.elu,
    'lrelu' : tf.nn.leaky_relu,
    'tanh' : tf.nn.tanh,
    'swish' : swish,
    'new' : new }
actfun = _ACTIVATIONS[activation] 
print("--------activation function using is :",actfun)

INPUT_NODE = 784     
OUTPUT_NODE = 10     
LAYER1_NODE = 500 
LAYER2_NODE = 500 
LAYER3_NODE = 500 
LAYER4_NODE = 500 
LAYER5_NODE = 500 
LAYER6_NODE = 500 
LAYER7_NODE = 500 
LAYER8_NODE = 300 
LAYER9_NODE = 200
LAYER10_NODE = 100 
                              
BATCH_SIZE = 100        

LEARNING_RATE_BASE = 0.008      
LEARNING_RATE_DECAY = 0.99    
REGULARAZTION_RATE = 0.0001   
TRAINING_STEPS = 10000        
MOVING_AVERAGE_DECAY = 0.99 
value=0.01

def inference(input_tensor, avg_class, W, B):
    if avg_class == None:
        layer1 = actfun(tf.matmul(input_tensor, W[0]) + B[0],value)
        layer2 = actfun(tf.matmul(layer1, W[1]) + B[1],value)
        layer3 = actfun(tf.matmul(layer2, W[2]) + B[2],value)
        layer4 = actfun(tf.matmul(layer3, W[3]) + B[3],value)
        layer5 = actfun(tf.matmul(layer4, W[4]) + B[4],value)
        layer6 = actfun(tf.matmul(layer5, W[5]) + B[5],value)
        layer7 = actfun(tf.matmul(layer6, W[6]) + B[6],value)
        layer8 = actfun(tf.matmul(layer7, W[7]) + B[7],value)
        layer9 = actfun(tf.matmul(layer8, W[8]) + B[8],value)
        layer10 = actfun(tf.matmul(layer9, W[9]) + B[9],value)
        return tf.matmul(layer10, W[10]) + B[10]
    else:  
        layer1 = actfun(tf.matmul(input_tensor, avg_class.average(W[0])) + avg_class.average(B[0]),value)
        layer2 = actfun(tf.matmul(layer1, avg_class.average(W[1])) + avg_class.average(B[1]),value)
        layer3 = actfun(tf.matmul(layer2, avg_class.average(W[2])) + avg_class.average(B[2]),value)
        layer4 = actfun(tf.matmul(layer3, avg_class.average(W[3])) + avg_class.average(B[3]),value)
        layer5 = actfun(tf.matmul(layer4, avg_class.average(W[4])) + avg_class.average(B[4]),value)
        layer6 = actfun(tf.matmul(layer5, avg_class.average(W[5])) + avg_class.average(B[5]),value)
        layer7 = actfun(tf.matmul(layer6, avg_class.average(W[6])) + avg_class.average(B[6]),value)
        layer8 = actfun(tf.matmul(layer7, avg_class.average(W[7])) + avg_class.average(B[7]),value)
        layer9 = actfun(tf.matmul(layer8, avg_class.average(W[8])) + avg_class.average(B[8]),value)
        layer10 = actfun(tf.matmul(layer9, avg_class.average(W[9])) + avg_class.average(B[9]),value)
        return tf.matmul(layer10, avg_class.average(W[10])) + avg_class.average(B[10])   
    
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, LAYER2_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[ LAYER2_NODE]))
    
    weights3 = tf.Variable(tf.truncated_normal([ LAYER2_NODE,  LAYER3_NODE], stddev=0.1))
    biases3 = tf.Variable(tf.constant(0.1, shape=[LAYER3_NODE]))
    
    weights4 = tf.Variable(tf.truncated_normal([LAYER3_NODE, LAYER4_NODE], stddev=0.1))
    biases4 = tf.Variable(tf.constant(0.1, shape=[LAYER4_NODE]))
    
    weights5 = tf.Variable(tf.truncated_normal([LAYER4_NODE, LAYER5_NODE], stddev=0.1))
    biases5 = tf.Variable(tf.constant(0.1, shape=[LAYER5_NODE]))
    
    weights6 = tf.Variable(tf.truncated_normal([LAYER5_NODE, LAYER6_NODE], stddev=0.1))
    biases6 = tf.Variable(tf.constant(0.1, shape=[LAYER6_NODE]))
    
    weights7 = tf.Variable(tf.truncated_normal([LAYER6_NODE, LAYER7_NODE], stddev=0.1))
    biases7 = tf.Variable(tf.constant(0.1, shape=[LAYER7_NODE]))
    
    weights8 = tf.Variable(tf.truncated_normal([LAYER7_NODE, LAYER8_NODE], stddev=0.1))
    biases8 = tf.Variable(tf.constant(0.1, shape=[LAYER8_NODE]))
    
    weights9 = tf.Variable(tf.truncated_normal([LAYER8_NODE, LAYER9_NODE], stddev=0.1))
    biases9 = tf.Variable(tf.constant(0.1, shape=[LAYER9_NODE]))
    
    weights10 = tf.Variable(tf.truncated_normal([LAYER9_NODE, LAYER10_NODE], stddev=0.1))
    biases10 = tf.Variable(tf.constant(0.1, shape=[LAYER10_NODE]))
    
    weights11 = tf.Variable(tf.truncated_normal([LAYER10_NODE, OUTPUT_NODE], stddev=0.1))
    biases11 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
    
    W=[weights1, weights2, weights3, weights4, weights5, weights6, weights7, weights8, weights9, weights10, weights11]
    B=[biases1, biases2, biases3, biases4, biases5, biases6, biases7, biases8, biases9, biases10, biases11]
    
    y = inference(x, None, W, B)
    
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    average_y = inference(x, variable_averages, W, B)
    
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    regularaztion = regularizer(W[0]) 
    for i in range(1,11):
        regularazation=regularaztion + regularizer(W[i]) 
    loss = cross_entropy_mean + regularaztion
    
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)
    
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels} 
        
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy using average model is %g " % (i, validate_acc))
            
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})
        validate_acc = sess.run(accuracy, feed_dict=validate_feed)
        print("After %d training step(s), validation accuracy using average model is %g " % (TRAINING_STEPS, validate_acc))
            
        test_acc=sess.run(accuracy,feed_dict=test_feed)
        print(("After %d training step(s), test accuracy using average model is %g" %(TRAINING_STEPS, test_acc)))

train(mnist)