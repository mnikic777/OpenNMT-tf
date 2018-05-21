"""Define layers related to the Facebook's Convolutional sequence-to-sequence model."""

import tensorflow as tf

from opennmt.layers.weight_norm import weight_norm
from opennmt.layers.padding import pad


def build_linear_shared_weights(num_outputs, weights, biases=None, scope=None):
  """Builds the fully connected layer with custom weight and biases.

  Args:
      num_outputs: The layer output depth.
      weights: a 2D tensor.  Dimensions typically: num_inputs, num_outputs
      biases: a 1D tensor.  Dimensions: num_outputs
      scope: Optional scope for variable_scope.

  Returns:
    A function.
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
  with tf.variable_scope(scope, default_name='linear_weight_norm'):
    if num_outputs is None:
      raise ValueError("num_outputs must be set to build the output layer")
    layer = weight_norm(
        tf.layers.Dense(
            num_outputs,
            use_bias=True,
            dtype=dtype,
            kernel_initializer=tf.random_normal_initializer(
                mean=0,
                stddev=tf.sqrt(
                    (1 - dropout)) / num_inputs)),
        _scope=scope)
    layer.build([None, num_inputs])
    return layer


def build_conv1d_weight_norm(num_inputs,
                             num_outputs,
                             kernel_size,
                             dropout=0.0,
                             padding="SAME",
                             scope=None):
  """Weight-normalized Conv1d layer as described in https://arxiv.org/abs/1705.03122.

  Args:
      num_inputs: The layer input depth.
      num_outputs: The layer output depth.
      kernel_size: An integer or tuple/list of a single integer, specifying the length
    of the 1D convolution window.
      dropout: The layer dropout, used for weight initialization.
      padding: One of "valid" or "same" (case-insensitive) or single number.
      scope: Optional scope for variable_scope.

  Returns:
    A ``tf.layers.Dense`` instance.
  """
  with tf.variable_scope(scope, default_name='conv1d_weight_norm'):
    if num_outputs is None:
      raise ValueError("num_outputs must be set to build the output layer")
    kernel_initializer = tf.random_normal_initializer(
        mean=0, stddev=tf.sqrt((4 * (1.0 - dropout)) / (kernel_size * num_inputs)))
    if not (isinstance(padding, str) and padding.upper() in ["SAME", "VALID"]):
      if isinstance(padding, int):
        layer = pad(weight_norm(tf.layers.Conv1D(filters=num_outputs,
                                                 kernel_size=kernel_size,
                                                 padding="VALID",
                                                 use_bias=True,
                                                 kernel_initializer=kernel_initializer),
                                axis=[0, 1]),
                    paddings=[[0, 0], [padding, padding], [0, 0]],
                    _scope=scope)
        layer.build([None, None, num_inputs])

        # remove future information
        def _remove_future_info(inputs):
          return layer(inputs)[:, :-padding, :]
        return _remove_future_info
      else:
        raise ValueError("padding must be VALID, SAME, or integer!")
    else:
      layer = weight_norm(
          tf.layers.Conv1D(
              filters=num_outputs,
              kernel_size=kernel_size,
              padding=padding,
              use_bias=True,
              kernel_initializer=kernel_initializer),
          axis=[0, 1],
          _scope=scope)
      layer.build([None, None, num_inputs])
      return layer


def build_attention_layer(conv_channels, embedding_dim, scope=None):
  """Builds attention layer as described in https://arxiv.org/abs/1705.03122

  Args:
      conv_channels: Convolutional layer size.
      embedding_dim: Decoder embedding dimension.
      scope: Optional scope for variable_scope.

  Returns:
      Attention function.
  """
  with tf.variable_scope(scope, default_name='attention'):
    in_projection = build_linear_weight_norm(
        conv_channels, embedding_dim, scope="in_projection")
    out_projection = build_linear_weight_norm(
        embedding_dim, conv_channels, scope="out_projection")

  def _attn_stack(next_layer, target_embed, encoder_outs):
    with tf.variable_scope(scope, default_name='attention'):
      residual = next_layer
      next_layer = (in_projection(next_layer) + target_embed) * tf.sqrt(0.5)

      next_layer = tf.matmul(next_layer, encoder_outs[0], transpose_b=True)
      next_layer = tf.nn.softmax(next_layer)
      attn_score = next_layer

      next_layer = tf.matmul(next_layer, encoder_outs[1])

      scale = tf.cast(tf.shape(encoder_outs[1]), tf.float32)[1]
      next_layer = next_layer * (scale * tf.sqrt(1.0 / scale))

      next_layer = (out_projection(next_layer) + residual) * tf.sqrt(0.5)

      return next_layer, attn_score

  return _attn_stack


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
