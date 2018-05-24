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


def scale_gradient(inputs, scale):
  """Scales gradient of :obj:`x` with :obj:`scale`."""
  dtype = inputs.dtype.base_dtype
  scale = tf.convert_to_tensor(scale, dtype=dtype)
  def backward_op(op, dy):
    return op.inputs[1] * dy, None

  def forward_op(x, scale):
    del scale
    return x

  def shape_op(op):
    return [op.inputs[0].get_shape()]

  _scale_func = function.Defun(
    dtype, dtype,
    python_grad_func=backward_op,
    shape_func=shape_op
  )(forward_op)

  output = _scale_func(inputs, scale)
  output.set_shape(inputs.get_shape())
  return output


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
  scope = "" if scope is None else scope
  with tf.variable_scope(scope, reuse=True):
    v = tf.get_variable(var)
  return v
