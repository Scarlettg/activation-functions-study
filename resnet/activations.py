import tensorflow as tf
from tensorflow.python.framework import function

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

