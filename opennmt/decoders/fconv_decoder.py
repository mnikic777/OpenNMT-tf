"""Define convolution-based decoders."""

import tensorflow as tf
from opennmt.decoders.rl_decoder import RLDecoder

from opennmt.decoders.decoder import Decoder, get_embedding_fn
from opennmt.layers.common import glu, reuse_variable
from opennmt.layers.fconv import build_linear_weight_norm, build_linear_shared_weights, \
  linear_weight_norm, conv1d_weight_norm, multi_step_attention, linearized_conv1d
from opennmt.layers.position import LearnedPositionalEmbedding
from opennmt.utils import beam_search


class FConvDecoder(RLDecoder):
  """Convolutional decoder as described in https://arxiv.org/abs/1705.03122."""

  def __init__(self,
               out_embedding_dim=256,
               convolutions=((512, 3),) * 20,
               attention=True,
               dropout=0.1,
               position_encoder=LearnedPositionalEmbedding(),
               share_embedding=False):
    """Initializes the parameters of the decoder.

    Args:
        out_embedding_dim: Decoder output embedding dimension.
        convolutions: Decoder layers [(dim, kernel_size), ...].
        attention: Decoder attention [True, ...].
        dropout: The probability to drop units from the inputs.
        position_encoder: The :class:`opennmt.convolutions.position.PositionEncoder` to
        apply on inputs or ``None``.
        share_embedding: Share input and output embeddings (requires decoder embedding dimension
            and :obj:`out_embedding_dim` to be equal).
    """
    self.dropout = dropout
    self.convolutions = convolutions
    self.position_encoder = position_encoder
    self.out_embedding_dim = out_embedding_dim
    self.share_embedding = share_embedding

    if isinstance(attention, bool):
      # expand True into [True, True, ...] and do the same with False
      attention = [attention] * len(convolutions)
    if not isinstance(attention, list) or len(attention) != len(convolutions):
      raise ValueError('Attention is expected to be a list of booleans of '
                       'length equal to the number of layers.')
    self.attention = attention

  def _build_memory_mask(self, memory, memory_sequence_length=None, maxlen=None):
    if memory_sequence_length is None:
      return None
    else:
      if maxlen is None:
        maxlen = tf.shape(memory[0])[1]
      mask = tf.sequence_mask(lengths=memory_sequence_length,
                              maxlen=maxlen,
                              dtype=memory[0].dtype)
      mask = tf.expand_dims(mask, axis=1)
      return mask

  def _init_cache(self, memory, memory_sequence_length=None):

    memory_shape = tf.shape(memory[0])
    batch_size = memory_shape[0]
    src_len = memory_shape[1]
    cache = {
      "memory": memory,
      "memory_mask": self._build_memory_mask(
          memory, memory_sequence_length, maxlen=src_len),
      "avg_attn_scores": tf.zeros([batch_size, 0, src_len])
    }

    for l, (out_channels, kernel_size) in enumerate(self.convolutions):
      cache["layer_{}".format(l)] = {
        "incremental_state": tf.zeros([batch_size, kernel_size, out_channels])
      }

    return cache

  def _symbols_to_logits_fn(self, embedding, vocab_size, mode, output_layer=None, dtype=None):
    embedding_fn = get_embedding_fn(embedding)
    if self.share_embedding:
      w_embs = reuse_variable("w_embs")
      output_layer = build_linear_shared_weights(
          vocab_size, w_embs, scope="proj_to_vocab_size")
    elif output_layer is None:
      output_layer = build_linear_weight_norm(self.out_embedding_dim, vocab_size,
                                              dropout=self.dropout,
                                              dtype=dtype,
                                              scope="proj_to_vocab_size")

    def _impl(ids, step, cache):
      inputs = embedding_fn(ids[:, -1:])
      if self.position_encoder is not None:
        inputs = self.position_encoder.apply_one(inputs, step + 1)
      outputs = self._cnn_stack(
          inputs,
          memory=cache["memory"],
          mode=mode,
          cache=cache)
      outputs = outputs[:, -1:, :]
      logits = output_layer(outputs)
      return logits, cache
    return _impl

  def _cnn_stack(self, inputs, memory, mode, memory_sequence_length=None, cache=None):
    in_channels = self.convolutions[0][0]
    next_layer = inputs
    num_attn_layers = len(self.attention)
    avg_attn_scores = None
    memory_mask = None

    if memory is not None:
      if cache is not None:
        memory_mask = cache["memory_mask"]
      elif memory_sequence_length is not None:
        memory_mask = self._build_memory_mask(
            memory, memory_sequence_length=memory_sequence_length)

    next_layer = linear_weight_norm(next_layer, in_channels, dropout=self.dropout,
                                    scope="in_projection")
    for l, (out_channels, kernel_size) in enumerate(self.convolutions):
      layer_name = "layer_{}".format(l)
      conv_layer_cache = cache[layer_name] if cache is not None else None
      with tf.variable_scope(layer_name):
        residual = next_layer if in_channels == out_channels else linear_weight_norm(next_layer, out_channels)

        next_layer = tf.layers.dropout(next_layer, self.dropout,
                                       training=mode == tf.estimator.ModeKeys.TRAIN)
        next_layer = linearized_conv1d(next_layer, out_channels * 2, kernel_size,
                                        padding=(kernel_size - 1),
                                        dropout=self.dropout, cache=conv_layer_cache)
        next_layer = glu(next_layer, axis=2)
        if self.attention[l]:
          next_layer, attn_scores = multi_step_attention(next_layer, inputs, memory, mask=memory_mask)
          if cache is not None:
            attn_scores /= num_attn_layers
            if avg_attn_scores is None:
              avg_attn_scores = attn_scores
            else:
              avg_attn_scores += attn_scores

        next_layer = (next_layer + residual) * tf.sqrt(0.5)

    next_layer = linear_weight_norm(next_layer, self.out_embedding_dim,
                                    scope="out_projection")
    next_layer = tf.layers.dropout(next_layer, rate=self.dropout,
                                   training=mode == tf.estimator.ModeKeys.TRAIN)
    if cache is not None:
      cache["avg_attn_scores"] = tf.concat([cache["avg_attn_scores"], avg_attn_scores], axis=1)

    return next_layer

  def _rl_decode(self,
                 is_multinomial,
                 embedding,
                 start_tokens,
                 sequence_length,
                 maximum_length,
                 vocab_size=None,
                 output_layer=None,
                 mode=tf.estimator.ModeKeys.PREDICT,
                 memory=None,
                 memory_sequence_length=None,
                 dtype=None):
    batch_size = tf.shape(start_tokens)[0]
    finished = tf.tile([False], [batch_size])
    step = tf.constant(0)
    inputs = tf.expand_dims(start_tokens, 1)
    lengths = tf.zeros([batch_size], dtype=tf.int32)
    cache = self._init_cache(memory, memory_sequence_length)
    symbols_to_logits_fn = self._symbols_to_logits_fn(
      embedding, vocab_size, mode, output_layer=output_layer, dtype=dtype)
    batch_nums = tf.expand_dims(tf.range(0, limit=batch_size), axis=1)
    loss = tf.zeros([batch_size], dtype=dtype)


    def _condition(unused_step, finished, unused_inputs, unused_lengths, unused_logits, unused_cache):
      return tf.logical_not(tf.reduce_all(finished))

    def _body(step, finished, inputs, lengths, loss, cache):
      inputs_lengths = tf.add(lengths, 1 - tf.cast(finished, lengths.dtype))

      logits, _ = symbols_to_logits_fn(inputs, step, cache)
      probs = tf.nn.softmax(logits)
      sample_ids = tf.multinomial(tf.squeeze(probs, axis=1), 1) if is_multinomial else tf.argmax(probs, axis=-1)
      sample_ids = tf.cast(sample_ids, inputs.dtype)

      indices = tf.concat([batch_nums, sample_ids], axis=1)
      gold_probs = tf.gather_nd(tf.squeeze(probs, axis=1), indices)


      # Accumulate log probabilities
      next_inputs = tf.concat([inputs, sample_ids], -1)
      next_loss = loss + tf.log(gold_probs)
      next_loss.set_shape(loss.get_shape())
      next_lengths = inputs_lengths
      step = step + 1

      next_finished = tf.logical_or(
        finished, step >= maximum_length)

      return step, next_finished, next_inputs, next_lengths, next_loss, cache

    step, _, outputs, lengths, loss, _ = tf.while_loop(
      _condition,
      _body,
      loop_vars=(step, finished, inputs, lengths, loss, cache),
      shape_invariants=(
        tf.TensorShape([]),
        finished.get_shape(),
        tf.TensorShape([None, None]),
        lengths.get_shape(),
        tf.TensorShape([None]),
        tf.contrib.framework.nest.map_structure(
          beam_search.get_state_shape_invariants, cache)
      ),
      parallel_iterations=1)

    out_mask = tf.sequence_mask(sequence_length, maxlen=maximum_length, dtype=outputs.dtype)
    out_mask = tf.Print(out_mask, [tf.shape(out_mask)], "out_mask = ")
    outputs = outputs[:, :-1]
    outputs = outputs * out_mask

    return outputs, loss

  def decode(self,
             inputs,
             sequence_length,
             vocab_size=None,
             initial_state=None,
             sampling_probability=None,
             embedding=None,
             output_layer=None,
             mode=tf.estimator.ModeKeys.TRAIN,
             memory=None,
             memory_sequence_length=None):
    if sampling_probability is not None:
      raise ValueError(
          "Scheduled sampling is not supported with FairseqConvDecoder")

    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, sequence_length=sequence_length)

    outputs = self._cnn_stack(inputs, memory, mode, memory_sequence_length)

    if self.share_embedding:
      w_embs = reuse_variable("w_embs")
      output_layer = build_linear_shared_weights(
          vocab_size, w_embs, scope="proj_to_vocab_size")
    elif output_layer is None:
      output_layer = build_linear_weight_norm(self.out_embedding_dim, vocab_size,
                                              dropout=self.dropout,
                                              dtype=inputs.dtype,
                                              scope="proj_to_vocab_size")

    logits = output_layer(outputs)  # fc3

    return (logits, None, sequence_length)

  def dynamic_decode(self,
                     embedding,
                     start_tokens,
                     end_token,
                     vocab_size=None,
                     initial_state=None,
                     output_layer=None,
                     maximum_iterations=250,
                     mode=tf.estimator.ModeKeys.PREDICT,
                     memory=None,
                     memory_sequence_length=None,
                     dtype=None,
                     return_alignment_history=False):
    batch_size = tf.shape(start_tokens)[0]
    finished = tf.tile([False], [batch_size])
    step = tf.constant(0)
    inputs = tf.expand_dims(start_tokens, 1)
    lengths = tf.zeros([batch_size], dtype=tf.int32)
    log_probs = tf.zeros([batch_size])
    cache = self._init_cache(memory, memory_sequence_length=memory_sequence_length)

    symbols_to_logits_fn = self._symbols_to_logits_fn(
        embedding, vocab_size, mode, output_layer=output_layer, dtype=dtype)

    def _condition(unused_step, finished, unused_inputs, unused_lengths, unused_log_probs,
                   unused_cache):
      return tf.logical_not(tf.reduce_all(finished))

    def _body(step, finished, inputs, lengths, log_probs, cache):
      inputs_lengths = tf.add(lengths, 1 - tf.cast(finished, lengths.dtype))

      logits, cache = symbols_to_logits_fn(inputs, step, cache)
      probs = tf.nn.log_softmax(logits)
      sample_ids = tf.argmax(probs, axis=-1)

      # Accumulate log probabilities
      sample_probs = tf.reduce_max(probs, axis=-1)
      masked_probs = tf.squeeze(sample_probs, -1) * \
          (1.0 - tf.cast(finished, sample_probs.dtype))
      log_probs = tf.add(log_probs, masked_probs)

      next_inputs = tf.concat([inputs, tf.cast(sample_ids, inputs.dtype)], -1)
      next_lengths = inputs_lengths
      next_finished = tf.logical_or(
          finished,
          tf.equal(tf.squeeze(sample_ids, axis=[1]), end_token))
      step = step + 1

      if maximum_iterations is not None:
        next_finished = tf.logical_or(
            next_finished, step >= maximum_iterations)

      return step, next_finished, next_inputs, next_lengths, log_probs, cache

    step, _, outputs, lengths, log_probs, cache = tf.while_loop(
        _condition,
        _body,
        loop_vars=(step, finished, inputs, lengths, log_probs, cache),
        shape_invariants=(
            tf.TensorShape([]),
            finished.get_shape(),
            tf.TensorShape([None, None]),
            lengths.get_shape(),
            log_probs.get_shape(),
            tf.contrib.framework.nest.map_structure(
                beam_search.get_state_shape_invariants, cache)
        ),
        parallel_iterations=1)

    outputs = tf.slice(outputs, [0, 1], [-1, -1])  # Ignore <s>

    # Make shape consistent with beam search
    outputs = tf.expand_dims(outputs, 1)
    lengths = tf.expand_dims(lengths, 1)
    log_probs = tf.expand_dims(log_probs, 1)

    if return_alignment_history:
      cache["avg_attn_scores"] = cache["avg_attn_scores"][:, :, :-1]
      alignment_history = tf.expand_dims(cache["avg_attn_scores"], 1)
      return (outputs, None, lengths, log_probs, alignment_history)
    return (outputs, None, lengths, log_probs)

  def greedy_decode(self,
                    embedding,
                    start_tokens,
                    end_token,
                    sequence_length,
                    maximum_length,
                    vocab_size=None,
                    initial_state=None,
                    output_layer=None,
                    mode=tf.estimator.ModeKeys.TRAIN,
                    memory=None,
                    memory_sequence_length=None,
                    dtype=None):
    outputs, _ = self._rl_decode(is_multinomial=False, embedding=embedding, start_tokens=start_tokens,
                                 sequence_length=sequence_length, maximum_length=maximum_length, vocab_size=vocab_size,
                                 output_layer=output_layer, mode=mode, memory=memory,
                                 memory_sequence_length=memory_sequence_length, dtype=dtype)
    return outputs

  def sampling_decode(self,
                      embedding,
                      start_tokens,
                      end_token,
                      sequence_length,
                      maximum_length,
                      vocab_size=None,
                      initial_state=None,
                      output_layer=None,
                      mode=tf.estimator.ModeKeys.TRAIN,
                      memory=None,
                      memory_sequence_length=None,
                      dtype=None,
                      sampling_function=tf.multinomial):
    return self._rl_decode(is_multinomial=True, embedding=embedding, start_tokens=start_tokens,
                           sequence_length=sequence_length, maximum_length=maximum_length, vocab_size=vocab_size,
                           output_layer=output_layer, mode=mode, memory=memory,
                           memory_sequence_length=memory_sequence_length, dtype=dtype)

  def dynamic_decode_and_search(self,
                                embedding,
                                start_tokens,
                                end_token,
                                vocab_size=None,
                                initial_state=None,
                                output_layer=None,
                                beam_width=5,
                                length_penalty=0.0,
                                maximum_iterations=250,
                                mode=tf.estimator.ModeKeys.PREDICT,
                                memory=None,
                                memory_sequence_length=None,
                                dtype=None,
                                return_alignment_history=False):
    cache = self._init_cache(memory, memory_sequence_length=memory_sequence_length)
    symbols_to_logits_fn = self._symbols_to_logits_fn(
        embedding, vocab_size, mode, output_layer=output_layer, dtype=dtype)

    outputs, log_probs, cache = beam_search.beam_search(
        symbols_to_logits_fn,
        start_tokens,
        beam_width,
        maximum_iterations,
        vocab_size,
        length_penalty,
        states=cache,
        eos_id=end_token,
        return_states=True)
    outputs = tf.slice(outputs, [0, 0, 1], [-1, -1, -1])  # Ignore <s>.

    lengths = tf.not_equal(outputs, 0)
    lengths = tf.cast(lengths, tf.int32)
    lengths = tf.reduce_sum(lengths, axis=-1)

    if return_alignment_history:
      cache["avg_attn_scores"] = cache["avg_attn_scores"][:, :, :-1]
      alignment_history = cache["avg_attn_scores"]
      return (outputs, None, lengths, log_probs, alignment_history)
    return (outputs, None, lengths, log_probs)
