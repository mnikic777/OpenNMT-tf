"""Define convolution-based encoders."""

import tensorflow as tf
from opennmt.encoders.encoder import Encoder
from opennmt.layers.common import glu
from opennmt.layers.common import scale_gradient
from opennmt.layers.position import LearnedPositionalEmbedding

from opennmt.layers.fconv import build_linear_weight_norm, build_conv1d_weight_norm


class FConvEncoder(Encoder):
  """An encoder that applies a convolution over the input sequence
as described in https://arxiv.org/abs/1705.03122.
"""

  def __init__(self,
               embedding_dim=512,
               convolutions=((512, 3),) * 20,
               dropout=0.1,
               position_encoder=LearnedPositionalEmbedding()):
    """Initializes the parameters of the encoder.

    Args:
      embedding_dim: Encoder embedding dimension.
      convolutions: Encoder layers [(dim, kernel_size), ...].
      dropout: The probability to drop units from the inputs.
      position_encoder: The :class:`opennmt.convolutions.position.PositionEncoder` to
        apply on inputs or ``None``.
    """
    self.dropout = dropout
    self.position_encoder = position_encoder

    in_channels = convolutions[0][0]
    self.fc1 = build_linear_weight_norm(
        embedding_dim, in_channels, dropout=dropout, scope="proj_to_in_channels")
    self.projections = list()
    self.convolutions = list()
    for i, (out_channels, kernel_size) in enumerate(convolutions):
      self.projections.append(build_linear_weight_norm(in_channels, out_channels,
                                                       scope="proj_layer_" + str(i))
                              if in_channels != out_channels else None)
      self.convolutions.append(build_conv1d_weight_norm(in_channels, out_channels * 2, kernel_size,
                                                        dropout=dropout,
                                                        scope="conv1d_layer_" + str(i)))
      in_channels = out_channels
    self.fc2 = build_linear_weight_norm(
        in_channels, embedding_dim, scope="proj_to_embedding_dim")
    self.num_attention_layers = None

  def _cnn_stack(self, inputs, mode):
    next_layer = inputs

    for proj, conv in zip(self.projections, self.convolutions):
      residual = next_layer if proj is None else proj(next_layer)
      next_layer = tf.layers.dropout(inputs=next_layer, rate=self.dropout,
                                     training=mode == tf.estimator.ModeKeys.TRAIN)
      next_layer = conv(next_layer)
      next_layer = glu(next_layer, axis=2)
      next_layer = (next_layer + residual) * tf.sqrt(0.5)

    return next_layer

  def encode(self, inputs, sequence_length=None, mode=tf.estimator.ModeKeys.TRAIN):
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, sequence_length=sequence_length)

    # Apply dropout to inputs.
    inputs = tf.layers.dropout(inputs, rate=self.dropout,
                               training=mode == tf.estimator.ModeKeys.TRAIN)

    with tf.variable_scope("cnn_encoder"):
      cnn_a = inputs
      cnn_a = self.fc1(cnn_a)
      cnn_a = self._cnn_stack(cnn_a, mode)

      cnn_a = self.fc2(cnn_a)

      cnn_a = scale_gradient(cnn_a, 1.0 / (2.0 * self.num_attention_layers))

      cnn_c = (cnn_a + inputs) * tf.sqrt(0.5)

    encoder_output = (cnn_a, cnn_c)
    encoder_state = tf.reduce_mean(cnn_c, axis=1)

    return (encoder_output, encoder_state, sequence_length)
