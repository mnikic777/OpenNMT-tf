"""Defines common layers."""

import tensorflow as tf

from tensorflow.python.framework import function


@function.Defun(
    python_grad_func=lambda x, dy: tf.convert_to_tensor(dy),
    shape_func=lambda op: [op.inputs[0].get_shape()])
def convert_gradient_to_tensor(x):
  """Wraps :obj:`x` to convert its gradient to a tensor."""
  return x


def embedding_lookup(params, ids):
  """Wrapper around ``tf.nn.embedding_lookup``.

  This converts gradients of the embedding variable to tensors which allows
  to use of optimizers that don't support sparse gradients (e.g. Adafactor).

  Args:
    params: The embedding tensor.
    ids: The ids to lookup in :obj:`params`.

  Returns:
    A ``tf.Tensor``, the embeddings that correspond to :obj:`ids`.
  """
  params = convert_gradient_to_tensor(params)
  return tf.nn.embedding_lookup(params, ids)


def glu(inputs, axis=-1):
  """
  The gated linear unit.

  Args:
      inputs: Input tensor.
      axis: Dimension on which to split the input.

  Returns:
      A `Tensor`. Has the same type as `x`.
  """
  layer_size = int(inputs.get_shape().as_list()[axis] / 2)
  a = inputs[:, :, 0:layer_size]
  b = inputs[:, :, layer_size:]
  b = tf.sigmoid(b)
  return tf.multiply(a, b)


def scale_gradient(x, scale):
  """Scales gradient of :obj:`x` with :obj:`scale`."""
  @function.Defun(
      python_grad_func=lambda x, dy: dy * scale,
      shape_func=lambda op: [op.inputs[0].get_shape()])
  def _scale_grad(x):
    return x

  return _scale_grad(x)


def reuse_variable(var, scope=None):
  """
  Gets an existing variable from a specified variable scope.

  Args:
      var: The name of the existing variable.
      scope: string or VariableScope: the scope for getting variable from.

  Returns:
        The existing `Variable` (or `PartitionedVariable`, if a
        partitioner was used).
  """
  with tf.variable_scope(scope, default_name="", reuse=True):
    v = tf.get_variable(var)
  return v
