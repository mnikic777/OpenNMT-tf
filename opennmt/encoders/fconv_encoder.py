"""Define convolution-based encoders."""

import tensorflow as tf
from opennmt.encoders.encoder import Encoder
from opennmt.layers.common import glu
from opennmt.layers.common import scale_gradient
from opennmt.layers.position import LearnedPositionalEmbedding

from opennmt.layers.fconv import linear_weight_norm, conv1d_weight_norm


class FConvEncoder(Encoder):
  """An encoder that applies a convolution over the input sequence
as described in https://arxiv.org/abs/1705.03122.
"""

  def __init__(self,
               convolutions=((512, 3),) * 20,
               dropout=0.1,
               position_encoder=LearnedPositionalEmbedding()):
    """Initializes the parameters of the encoder.

    Args:
      convolutions: Encoder layers [(dim, kernel_size), ...].
      dropout: The probability to drop units from the inputs.
      position_encoder: The :class:`opennmt.convolutions.position.PositionEncoder` to
        apply on inputs or ``None``.
    """
    self.dropout = dropout
    self.position_encoder = position_encoder
    self.convolutions = convolutions
    self.num_attention_layers = None

  def encode(self, inputs, sequence_length=None, mode=tf.estimator.ModeKeys.TRAIN):
    embedding_dim = inputs.get_shape().as_list()[-1]
    in_channels = self.convolutions[0][0]

    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, sequence_length=sequence_length)

    # Apply dropout to inputs.
    inputs = tf.layers.dropout(inputs, rate=self.dropout,
                               training=mode == tf.estimator.ModeKeys.TRAIN)

    cnn_a = inputs
    cnn_a = linear_weight_norm(cnn_a, in_channels, dropout=self.dropout,
                               scope="in_projection")
    for i, (out_channels, kernel_size) in enumerate(self.convolutions):
      with tf.variable_scope("layer_{}".format(i)):
        residual = cnn_a if in_channels == out_channels else linear_weight_norm(cnn_a, out_channels)
        cnn_a = tf.layers.dropout(inputs=cnn_a, rate=self.dropout,
                                  training=mode == tf.estimator.ModeKeys.TRAIN)
        cnn_a = conv1d_weight_norm(cnn_a, out_channels * 2, kernel_size,
                                   dropout=self.dropout)
        cnn_a = glu(cnn_a, axis=2)
        cnn_a = (cnn_a + residual) * tf.sqrt(0.5)
        in_channels = out_channels

    cnn_a = linear_weight_norm(cnn_a, embedding_dim,
                               scope="out_projection")
    cnn_a = scale_gradient(cnn_a, 1.0 / (2.0 * self.num_attention_layers))

    cnn_c = (cnn_a + inputs) * tf.sqrt(0.5)

    encoder_output = (cnn_a, cnn_c)
    encoder_state = tf.reduce_mean(cnn_c, axis=1)

    return (encoder_output, encoder_state, sequence_length)
