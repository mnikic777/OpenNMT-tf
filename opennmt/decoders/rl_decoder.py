import abc
import six
import tensorflow as tf
from opennmt.decoders.decoder import Decoder

@six.add_metaclass(abc.ABCMeta)
class RLDecoder(Decoder):

  def greedy_decode(self,
                    embedding,
                    start_tokens,
                    end_token,
                    sequence_length,
                    vocab_size=None,
                    initial_state=None,
                    output_layer=None,
                    mode=tf.estimator.ModeKeys.TRAIN,
                    memory=None,
                    memory_sequence_length=None,
                    dtype=None):
    raise NotImplementedError()

  def sampling_decode(self,
                      embedding,
                      start_tokens,
                      end_token,
                      sequence_length,
                      vocab_size=None,
                      initial_state=None,
                      output_layer=None,
                      mode=tf.estimator.ModeKeys.TRAIN,
                      memory=None,
                      memory_sequence_length=None,
                      dtype=None):
    raise NotImplementedError()
