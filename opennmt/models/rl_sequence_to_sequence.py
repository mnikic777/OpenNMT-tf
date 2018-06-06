import tensorflow as tf
from opennmt.utils.losses import cross_entropy_sequence_loss_rl

from opennmt.decoders.decoder import get_sampling_probability
from tensor2tensor.utils.bleu_hook import compute_bleu
from opennmt.models.sequence_to_sequence import replace_unknown_target

from opennmt import constants, inputters
from opennmt.models import SequenceToSequence


class RLSequenceToSequence(SequenceToSequence):

  def _build(self, features, labels, params, mode, config=None):
    features_length = self._get_features_length(features)
    log_dir = config.model_dir if config is not None else None

    with tf.variable_scope("encoder"):
      source_inputs = self.source_inputter.transform_data(
          features,
          mode=mode,
          log_dir=log_dir)
      encoder_outputs, encoder_state, encoder_sequence_length = self.encoder.encode(
          source_inputs,
          sequence_length=features_length,
          mode=mode)

    target_vocab_size = self.target_inputter.vocabulary_size
    target_dtype = self.target_inputter.dtype

    with tf.variable_scope("decoder") as decoder_scope:
      if labels is not None:
        sampling_probability = get_sampling_probability(
            tf.train.get_or_create_global_step(),
            read_probability=params.get("scheduled_sampling_read_probability"),
            schedule_type=params.get("scheduled_sampling_type"),
            k=params.get("scheduled_sampling_k"))

        target_inputs = self.target_inputter.transform_data(
            labels,
            mode=mode,
            log_dir=log_dir)
        logits, _, _ = self.decoder.decode(
            target_inputs,
            self._get_labels_length(labels),
            vocab_size=target_vocab_size,
            initial_state=encoder_state,
            sampling_probability=sampling_probability,
            embedding=self._scoped_target_embedding_fn(mode, decoder_scope),
            mode=mode,
            memory=encoder_outputs,
            memory_sequence_length=encoder_sequence_length)
        if params.get("rl_mode", False):
          with tf.variable_scope(decoder_scope, reuse=labels is not None) as decoder_scope:
            batch_size = tf.shape(encoder_sequence_length)[0]
            start_tokens = tf.fill([batch_size], constants.START_OF_SENTENCE_ID)
            end_token = constants.END_OF_SENTENCE_ID
            outputs_greedy = self.decoder.greedy_decode(
              self._scoped_target_embedding_fn(mode, decoder_scope),
              start_tokens,
              end_token,
              self._get_labels_length(labels),
              tf.shape(labels["ids_out"])[1],
              vocab_size=target_vocab_size,
              initial_state=encoder_state,
              mode=mode,
              memory=encoder_outputs,
              memory_sequence_length=encoder_sequence_length,
              dtype=target_dtype)
            outputs_sample, logits_sample = self.decoder.sampling_decode(
              self._scoped_target_embedding_fn(mode, decoder_scope),
              start_tokens,
              end_token,
              self._get_labels_length(labels),
              tf.shape(labels["ids_out"])[1],
              vocab_size=target_vocab_size,
              initial_state=encoder_state,
              mode=mode,
              memory=encoder_outputs,
              memory_sequence_length=encoder_sequence_length,
              dtype=target_dtype)
            reward_sample = approximate_bleu(outputs_sample, labels["ids"])
            reward_true = approximate_bleu(outputs_greedy, labels["ids"])
            logits = (logits, logits_sample, reward_true, reward_sample)
      else:
        logits = None

    if mode != tf.estimator.ModeKeys.TRAIN:
      with tf.variable_scope(decoder_scope, reuse=labels is not None) as decoder_scope:
        batch_size = tf.shape(encoder_sequence_length)[0]
        beam_width = params.get("beam_width", 1)
        maximum_iterations = params.get("maximum_iterations", 250)
        start_tokens = tf.fill([batch_size], constants.START_OF_SENTENCE_ID)
        end_token = constants.END_OF_SENTENCE_ID

        if beam_width <= 1:
          sampled_ids, _, sampled_length, log_probs, alignment = self.decoder.dynamic_decode(
              self._scoped_target_embedding_fn(mode, decoder_scope),
              start_tokens,
              end_token,
              vocab_size=target_vocab_size,
              initial_state=encoder_state,
              maximum_iterations=maximum_iterations,
              mode=mode,
              memory=encoder_outputs,
              memory_sequence_length=encoder_sequence_length,
              dtype=target_dtype,
              return_alignment_history=True)
        else:
          length_penalty = params.get("length_penalty", 0)
          sampled_ids, _, sampled_length, log_probs, alignment = (
              self.decoder.dynamic_decode_and_search(
                  self._scoped_target_embedding_fn(mode, decoder_scope),
                  start_tokens,
                  end_token,
                  vocab_size=target_vocab_size,
                  initial_state=encoder_state,
                  beam_width=beam_width,
                  length_penalty=length_penalty,
                  maximum_iterations=maximum_iterations,
                  mode=mode,
                  memory=encoder_outputs,
                  memory_sequence_length=encoder_sequence_length,
                  dtype=target_dtype,
                  return_alignment_history=True))

      target_vocab_rev = tf.contrib.lookup.index_to_string_table_from_file(
          self.target_inputter.vocabulary_file,
          vocab_size=target_vocab_size - self.target_inputter.num_oov_buckets,
          default_value=constants.UNKNOWN_TOKEN)
      target_tokens = target_vocab_rev.lookup(tf.cast(sampled_ids, tf.int64))

      if params.get("replace_unknown_target", False):
        if alignment is None:
          raise TypeError("replace_unknown_target is not compatible with decoders "
                          "that don't return alignment history")
        if not isinstance(self.source_inputter, inputters.WordEmbedder):
          raise TypeError("replace_unknown_target is only defined when the source "
                          "inputter is a WordEmbedder")
        source_tokens = features["tokens"]
        if beam_width > 1:
          source_tokens = tf.contrib.seq2seq.tile_batch(source_tokens, multiplier=beam_width)
        # Merge batch and beam dimensions.
        original_shape = tf.shape(target_tokens)
        target_tokens = tf.reshape(target_tokens, [-1, original_shape[-1]])
        attention = tf.reshape(alignment, [-1, tf.shape(alignment)[2], tf.shape(alignment)[3]])
        replaced_target_tokens = replace_unknown_target(target_tokens, source_tokens, attention)
        target_tokens = tf.reshape(replaced_target_tokens, original_shape)

      predictions = {
          "tokens": target_tokens,
          "length": sampled_length,
          "log_probs": log_probs
      }
      if alignment is not None:
        predictions["alignment"] = alignment
    else:
      predictions = None


    return logits, predictions

  def _compute_loss(self, features, labels, outputs, params, mode):
    if params.get("rl_mode", False):
      logits, logits_sample, reward_true, reward_sample = outputs
      logits = tf.Print(logits, [tf.shape(logits), tf.shape(logits_sample)], "shapes = ", summarize=1000)
      return cross_entropy_sequence_loss_rl(
        logits,
        logits_sample,
        reward_true,
        reward_sample,
        labels["ids_out"],
        self._get_labels_length(labels),
        scaling_factor=params.get("rl_scaling_factor", 0.5),
        label_smoothing=params.get("label_smoothing", 0.0),
        average_in_time=params.get("average_loss_in_time", False),
        mode=mode)
    else:
      return super(RLSequenceToSequence, self)._compute_loss(features, labels,
                                                             outputs, params, mode)

def approximate_bleu(outputs, labels):
  bleu = tf.py_func(compute_bleu, (labels, outputs), tf.float32)
  bleu.set_shape(1)
  return bleu

