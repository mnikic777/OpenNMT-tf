"""Convolutional sequence-to-sequence model."""

import tensorflow as tf

import opennmt.constants as constants
from opennmt.decoders.fconv_decoder import FConvDecoder
from opennmt.encoders.fconv_encoder import FConvEncoder
from opennmt.layers.position import LearnedPositionalEmbedding
from opennmt.models import SequenceToSequence
from opennmt.layers.fconv import shift_padding_tokens_left


def shift_source_sequence(inputter, data):
  """Prepares shifted target sequences.
Given a source sequence ``a b c``, the encoder input should be
``a b c </s>`` and also shifts padding tokens to the left side of source sequence.

Args:
  inputter: The :class:`opennmt.inputters.inputter.Inputter` that processed
    :obj:`data`.
  data: A dict of ``tf.Tensor`` containing ``ids`` and ``length`` keys.

Returns:
  The updated :obj:`data` dictionary with ``ids`` the sequence prefixed
  with the end token id and padding tokens shifted to the left. Additionally,
  the ``length`` is increased by 1 to reflect the added token on both sequences.
"""
  eos = tf.cast(tf.constant([constants.END_OF_SENTENCE_ID]), tf.int64)

  ids = data["ids"]
  length = data["length"]

  data = inputter.set_data_field(data, "ids", tf.concat([ids, eos], axis=0))

  # Increment length accordingly.
  inputter.set_data_field(data, "length", length + 1)

  return data



class FConvModel(SequenceToSequence):
  """A convolutional sequence-to-sequence model as described in
  https://arxiv.org/abs/1705.03122."""

  def __init__(self,
               source_inputter,
               target_inputter,
               encoder_convolutions=((512, 3),) * 20,
               decoder_convolutions=((512, 3),) * 20,
               attention=True,
               dropout=0.1,
               encoder_position_encoder=LearnedPositionalEmbedding(),
               decoder_position_encoder=LearnedPositionalEmbedding(),
               out_embedding_dim=512,
               share_embeddings=False,
               name="fairseq"):
    """ Initializes a Convolutional sequence to sequence model.

    Args:
        source_inputter: A :class:`opennmt.inputters.inputter.Inputter` to process
            the source data.
        target_inputter: A :class:`opennmt.inputters.inputter.Inputter` to process
            the target data. Currently, only the
            :class:`opennmt.inputters.text_inputter.WordEmbedder` is supported.
        encoder_convolutions: Encoder layers [(dim, kernel_size), ...].
        decoder_convolutions: Decoder layers [(dim, kernel_size), ...].
        attention: Decoder attention [True, ...].
        dropout: The probability to drop units in each layer output.
        encoder_position_encoder: A :class:`opennmt.layers.position.PositionEncoder` to
            apply on the encoder inputs.
        decoder_position_encoder: A :class:`opennmt.layers.position.PositionEncoder` to
            apply on the decoder inputs.
        out_embedding_dim: Decoder output embedding dimension.
        share_embeddings: Share input and output embeddings (requires decoder embedding dimension
            and :obj:`out_embedding_dim` to be equal).
        daisy_chain_variables:
        name: The name of this model.
    """
    with tf.variable_scope("encoder"):
        encoder = FConvEncoder(convolutions=encoder_convolutions,
                               dropout=dropout,
                               position_encoder=encoder_position_encoder)
    with tf.variable_scope("decoder"):
        decoder = FConvDecoder(out_embedding_dim=out_embedding_dim,
                               convolutions=decoder_convolutions,
                               attention=attention,
                               dropout=dropout,
                               position_encoder=decoder_position_encoder,
                               share_embedding=share_embeddings)
    encoder.num_attention_layers = sum(
        layer is not None for layer in decoder.attention)
    super(FConvModel, self). __init__(source_inputter,
                                      target_inputter,
                                      encoder,
                                      decoder,
                                      daisy_chain_variables=False,
                                      name=name)
    self.source_inputter.add_process_hooks(
        [shift_source_sequence])
