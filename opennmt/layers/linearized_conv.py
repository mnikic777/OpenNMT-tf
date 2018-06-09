from tensorflow.python.layers.convolutional import Conv1D
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops

class LinearizedConvolution(Conv1D):

  def __init__(self, filters, kernel_size, strides=1, padding='valid', data_format='channels_last', dilation_rate=1,
               activation=None, use_bias=True, kernel_initializer=None, bias_initializer=init_ops.zeros_initializer(),
               kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
               bias_constraint=None, trainable=True, name=None, **kwargs):
    if isinstance(padding, int):
      self.padding_value = padding
      padding = 'valid'
    else:
      self.padding_value = None
    super().__init__(filters, kernel_size, strides, padding, data_format, dilation_rate, activation, use_bias,
                     kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer,
                     kernel_constraint, bias_constraint, trainable, name, **kwargs)
    self._linearized_weight = None

  def call(self, inputs, **kwargs):
    cache = kwargs.get('cache')
    if cache is None:
      if self.padding_value is not None:
        inputs = array_ops.pad(tensor=inputs, paddings=[[0, 0], [self.padding_value, self.padding_value], [0, 0]])
      return super().call(inputs)[:, :-self.padding_value, :]

    weight = self._get_linearized_weight()
    batch_size = array_ops.shape(inputs)[0]
    kw = self.kernel_size[0]

    if kw > 1:
      cache["incremental_state"] = array_ops.concat([cache["incremental_state"][:, 1:, :], inputs[:, -1:, :]], axis=1)
      inputs = cache["incremental_state"]
    outputs = nn_ops.xw_plus_b(array_ops.reshape(inputs, [batch_size, -1]), array_ops.transpose(weight), self.bias)
    return array_ops.reshape(outputs, [batch_size, -1, self.filters])

  def _get_linearized_weight(self):
    if self._linearized_weight is None:
      weight = array_ops.transpose(self.kernel, perm=[2, 0, 1])
      self._linearized_weight = array_ops.reshape(weight, [self.filters, -1])
    return self._linearized_weight

