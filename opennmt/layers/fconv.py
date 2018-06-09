"""Define layers related to the Facebook's Convolutional sequence-to-sequence model."""
import math

import tensorflow as tf
from opennmt.layers.linearized_conv import LinearizedConvolution

from opennmt.layers.weight_norm import weight_norm


def build_linear_shared_weights(num_outputs, weights, biases=None, scope=None):
  """Builds a fully connected layer with custom weight and biases.

  Args:
      num_outputs: The layer output depth.
      weights: a 2D tensor.  Dimensions typically: num_inputs, num_outputs
      biases: a 1D tensor.  Dimensions: num_outputs
      scope: Optional scope for variable_scope.

  Returns:
    A function that takes input tensor and returns the tensor variable representing
    the result of the series of operations.
  """
  with tf.variable_scope(scope, default_name='linear_shared_weights'):
    if biases is None:
      biases = tf.get_variable(
          'b',
          shape=[num_outputs],
          initializer=tf.zeros_initializer(),
          trainable=True)

    def _simple_fc(inputs):
      shape = inputs.get_shape().as_list()
      if len(shape) > 2:
        outputs = tf.tensordot(inputs, weights, [[len(shape) - 1],
                                                 [0]])
        output_shape = shape[:-1] + [num_outputs]
        outputs.set_shape(output_shape)
      else:
        outputs = tf.matmul(inputs, weights)
      outputs = tf.nn.bias_add(outputs, biases)
      return outputs
    return _simple_fc


def build_linear_weight_norm(num_inputs,
                             num_outputs,
                             dropout=0.0,
                             dtype=None,
                             scope=None):
  """Builds the fully connected layer with normalized weights as described in
  https://arxiv.org/abs/1705.03122


  Args:
    num_inputs: The layer input depth.
    num_outputs: The layer output depth.
    dropout: The layer dropout, used for weight initialization.
    dtype: The layer dtype.
    scope: Optional scope for variable_scope.

  Returns:
    A ``tf.layers.Dense`` instance.

  Raises:
    ValueError: if :obj:`vocab_size` is ``None``.
  """
  if num_outputs is None:
      raise ValueError("num_outputs must be set to build the output layer")
  layer = weight_norm(
      tf.layers.Dense(
          num_outputs,
          use_bias=True,
          dtype=dtype,
          kernel_initializer=tf.random_normal_initializer(
              mean=0,
              stddev=math.sqrt(
                  (1 - dropout)) / num_inputs), _scope=scope))
  layer.build([None, num_inputs])
  return layer


def shift_padding_tokens_left(inputs, sequence_lengths):
  """
  Moves padding tokens from the right side of sentences to the left side.

  Args:
      inputs: A batch of sentences.
      sequence_lengths: Lengths of each sentence in the batch.

  Returns:
      A batch with padding tokens positioned on the left side of sentences.
  """
  return tf.reverse(
      tensor=tf.reverse_sequence(
          input=inputs,
          seq_lengths=sequence_lengths,
          batch_dim=0,
          seq_dim=1),
      axis=[1])


def linear_weight_norm(inputs,
                       num_outputs,
                       dropout=0.0,
                       dtype=None,
                       scope=None):
    """Builds the fully connected layer with normalized weights as described in
    https://arxiv.org/abs/1705.03122


    Args:
      num_inputs: The layer input depth.
      num_outputs: The layer output depth.
      dropout: The layer dropout, used for weight initialization.
      dtype: The layer dtype.
      scope: Optional scope for variable_scope.

    Returns:
      Output tensor.

    Raises:
      ValueError: if :obj:`vocab_size` is ``None``.
    """
    num_inputs = inputs.get_shape().as_list()[-1]
    layer = build_linear_weight_norm(num_inputs, num_outputs, dropout,
                                     dtype, scope)
    return layer(inputs)


def conv1d_weight_norm(inputs,
                       num_outputs,
                       kernel_size,
                       dropout=0.0,
                       padding="SAME",
                       scope=None):
    """Weight-normalized Conv1d layer as described in https://arxiv.org/abs/1705.03122.

    Args:
        inputs: Tensor input.
        num_outputs: The layer output depth.
        kernel_size: An integer or tuple/list of a single integer, specifying the length
      of the 1D convolution window.
        dropout: The layer dropout, used for weight initialization.
        padding: One of "valid" or "same" (case-insensitive) or single number.
        scope: Optional scope for variable_scope.

    Returns:
      Output tensor.
    """
    num_inputs = inputs.get_shape().as_list()[-1]
    if num_outputs is None:
        raise ValueError("num_outputs must be set to build the output layer")
    kernel_initializer = tf.random_normal_initializer(
        mean=0, stddev=math.sqrt((4 * (1.0 - dropout)) / (kernel_size * num_inputs)))
    layer = weight_norm(
        tf.layers.Conv1D(
            filters=num_outputs,
            kernel_size=kernel_size,
            padding=padding,
            use_bias=True,
            kernel_initializer=kernel_initializer,
            _scope=scope),
        axis=[0, 1])
    layer.build([None, None, num_inputs])
    return layer(inputs)

def linearized_conv1d(inputs,
                       num_outputs,
                       kernel_size,
                       dropout=0.0,
                       padding="SAME",
                       scope=None,
                       cache=None):
  num_inputs = inputs.get_shape().as_list()[-1]
  if num_outputs is None:
    raise ValueError("num_outputs must be set to build the output layer")
  kernel_initializer = tf.random_normal_initializer(
    mean=0, stddev=math.sqrt((4 * (1.0 - dropout)) / (kernel_size * num_inputs)), seed=301)
  layer = weight_norm(
    LinearizedConvolution(
      filters=num_outputs,
      kernel_size=kernel_size,
      padding=padding,
      use_bias=True,
      kernel_initializer=kernel_initializer,
      _scope=scope),
    axis=[0, 1])
  layer.build([None, None, num_inputs])
  return layer(inputs, cache=cache)

def multi_step_attention(inputs, target_embed, encoder_outs, mask=None, scope=None):
    """Computes the multi-step attention as described in
    https://arxiv.org/abs/1705.03122

    Args:
        inputs: Current decoder layer input.
        target_embed: Embedding matrix of target element.
        encoder_outs: A tuple of encoder outputs
        scope: Optional scope for variable_scope.

    Returns:
        A tuple (attention output, attention score)
    """
    with tf.variable_scope(scope, default_name="attention"):
      conv_channels = inputs.get_shape().as_list()[-1]
      embedding_dim = target_embed.get_shape().as_list()[-1]

      residual = inputs
      next_layer = (linear_weight_norm(inputs, embedding_dim) + target_embed) * tf.sqrt(0.5)

      next_layer = tf.matmul(next_layer, encoder_outs[0], transpose_b=True)
      if mask is not None:
        next_layer = next_layer * mask + ((mask - 1.0) * next_layer.dtype.max)
      next_layer = tf.nn.softmax(next_layer)
      attn_score = next_layer

      next_layer = tf.matmul(next_layer, encoder_outs[1])

      scale = tf.cast(tf.shape(encoder_outs[1]), tf.float32)[1]
      next_layer = next_layer * (scale * tf.sqrt(1.0 / scale))

      next_layer = (linear_weight_norm(next_layer, conv_channels) + residual) * tf.sqrt(0.5)

      return next_layer, attn_score