"""Define Weight Normalization class."""

import tensorflow as tf

from tensorflow.python.layers import base


class WeightNorm(base.Layer):
  """Weight normalization decorator. It applies weight normalization to a parameter
  in the decorated layer class. Weight normalization is described in
  https://arxiv.org/abs/1602.07868"""

  def __init__(self, base_layer, kernel_name, axis, **kwargs):
    """ Initializes a weight normalization layer.

    Args:
        base_layer: A layer whose weights need to be normalized.
        kernel_name: Name of weight_parameter.
        axis: The dimension over which to compute norm.
        **kwargs: Additional keyword arguments.
    """
    super().__init__(**kwargs)
    self.base_layer = base_layer
    self.kernel_name = kernel_name
    self.axis = axis

  def build(self, input_shape):
    self.base_layer.build(input_shape)
    normalized_kernel = self.normalize_variable(variable=getattr(
        self.base_layer, self.kernel_name), norm_axis=self.axis)
    setattr(self.base_layer, self.kernel_name,
            normalized_kernel)
    self.built = True

  def call(self, inputs, **kwargs):
    return self.base_layer.call(inputs, **kwargs)

  def compute_output_shape(self, input_shape):
    return self.base_layer.compute_output_shape(input_shape)

  def normalize_variable(self, variable, norm_axis):
    """
    Computes the norm over :obj:`norm_axis` dimension of :obj:`variable`.

    Args:
        variable: The variable to normalize.
        norm_axis: The dimension over which to compute the norm.

    Returns:
        Normalized variable.
    """
    w_norm = tf.norm(variable.initialized_value(), axis=norm_axis)
    g = self.add_variable(name='g', shape=None,
                          dtype=tf.float32, initializer=w_norm, trainable=True)

    return variable * tf.div(g, tf.norm(variable, axis=norm_axis))


def weight_norm(base_layer, kernel_name='kernel', axis=0, **kwargs):
  """
  Applies weight normalization to a parameter :obj:`kernel_name`
  in the given :obj:`base_layer`.

  Args:
      base_layer: A layer whose weights need to be normalized.
      kernel_name: Name of weight_parameter.
      axis: The dimension over which to compute norm.
      **kwargs: Additional keyword arguments.

  Returns:
      Original layer decorated with weight normalization decorator.
  """
  return WeightNorm(base_layer, kernel_name, axis, **kwargs)
