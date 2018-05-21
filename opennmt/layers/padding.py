"""Defines padding decorator class."""

import tensorflow as tf
from tensorflow.python.layers import base


class Padding(base.Layer):
  """Tensor padding decorator class. It is mostly a wrapper around ``tf.pad`` function.

  See Also:
      :meth:`tf.pad` that pads a tensor."""

  def __init__(self,
               base_layer,
               paddings,
               mode="CONSTANT",
               op_name=None,
               constant_values=0,
               **kwargs):
    """Initializes a tensor padding decorator.
    Args:
        base_layer: A layer to be padded.
        paddings: A `Tensor` of type `int32`.
        mode: One of "CONSTANT", "REFLECT", or "SYMMETRIC" (case-insensitive).
        op_name: A name for the operation (optional).
        constant_values: In "CONSTANT" mode, the scalar pad value to use. Must be
          same type as `tensor`.
        **kwargs: Additional keyword arguments.
    """
    super().__init__(**kwargs)
    self.base_layer = base_layer
    self.paddings = paddings
    self.mode = mode
    self.op_name = op_name
    self.constant_values = constant_values

  def build(self, input_shape):
    self.base_layer.build(input_shape)
    self.built = True

  def call(self, inputs, **kwargs):
    inputs = tf.pad(tensor=inputs, paddings=self.paddings, mode=self.mode,
                    name=self.op_name, constant_values=self.constant_values)
    return self.base_layer.call(inputs, **kwargs)

  def compute_output_shape(self, input_shape):
    return self.base_layer.compute_output_shape(input_shape)


def pad(base_layer,
        paddings,
        mode="CONSTANT",
        op_name=None,
        constant_values=0,
        **kwargs):
  """Adds padding to the given :obj:`base_layer`.

  Args:
      base_layer: A layer to be padded.
      paddings: A `Tensor` of type `int32`.
      mode: One of "CONSTANT", "REFLECT", or "SYMMETRIC" (case-insensitive).
      op_name: A name for the operation (optional).
      constant_values: In "CONSTANT" mode, the scalar pad value to use. Must be
            same type as `tensor`.
      **kwargs: Additional keyword arguments.

  Returns:
      Original layer decorate with padding decorator.
  """
  return Padding(
      base_layer,
      paddings,
      mode,
      op_name,
      constant_values,
      **kwargs)
