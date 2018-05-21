import opennmt.layers.fconv as fconv
import tensorflow as tf


class FConvTest(tf.test.TestCase):

  def testBuildLinearSharedWeights(self):
    batch_size = 5
    max_len = 4
    num_inputs = 2
    num_outputs = 3

    inputs = tf.random_normal(shape=[batch_size, max_len, num_inputs])
    weights = tf.random_normal(shape=[num_inputs, num_outputs])
    biases = tf.zeros(shape=[num_outputs])
    layer = fconv.build_linear_shared_weights(
        weights.get_shape()[-1], weights, biases)
    with self.test_session() as sess:
      output = sess.run(layer(inputs))
      self.assertAllEqual([batch_size, max_len, num_outputs], output.shape)

  def testBuildLinearWeightNorm(self):
    batch_size = 5
    max_len = 4
    num_inputs = 2
    num_outputs = 3

    inputs = tf.random_normal(shape=[batch_size, max_len, num_inputs])
    layer = fconv.build_linear_weight_norm(num_inputs, num_outputs)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      output = sess.run(layer(inputs))
      self.assertAllEqual([batch_size, max_len, num_outputs], output.shape)

  def testBuildConv1DWeightNorm(self):
    batch_size = 5
    max_len = 4
    num_inputs = 2
    kernel_size = 3
    num_outputs = 3

    inputs = tf.random_normal(shape=[batch_size, max_len, num_inputs])
    layer = fconv.build_conv1d_weight_norm(
        num_inputs, num_outputs, kernel_size)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      output = sess.run(layer(inputs))
      self.assertAllEqual([batch_size, max_len, num_outputs], output.shape)

  def testBuildConv1DWeightNormWithCustomPadding(self):
    batch_size = 5
    max_len = 4
    num_inputs = 2
    kernel_size = 3
    padding = 2
    num_outputs = 3

    inputs = tf.random_normal(shape=[batch_size, max_len, num_inputs])
    layer = fconv.build_conv1d_weight_norm(
        num_inputs, num_outputs, kernel_size, padding=padding)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      output = sess.run(layer(inputs))
      self.assertAllEqual([batch_size, max_len, num_outputs], output.shape)

  def testBuildAttentionLayer(self):
    batch_size = 5
    max_src_len = 3
    max_tgt_len = 4
    num_inputs = 6

    embedding_dim = 5

    inputs = tf.random_normal(shape=[batch_size, max_tgt_len, num_inputs])
    target_embedding = tf.random_normal(
        shape=[batch_size, max_tgt_len, embedding_dim])
    encoder_outputs = (
        tf.random_normal(
            shape=[
                batch_size,
                max_src_len,
                embedding_dim]),
        tf.random_normal(
            shape=[
                batch_size,
                max_src_len,
                embedding_dim]))

    layer = fconv.build_attention_layer(num_inputs, embedding_dim)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      output, attn_score = sess.run(
          layer(inputs, target_embedding, encoder_outputs))
      self.assertAllEqual([batch_size, max_tgt_len, num_inputs], output.shape)
      self.assertAllEqual(
          [batch_size, max_tgt_len, max_src_len], attn_score.shape)

  def testShiftPaddingTokensLeft(self):
    inputs = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 0, 0],
        [1, 2, 0, 0, 0],
    ]
    expected = [
        [1, 2, 3, 4, 5],
        [0, 0, 1, 2, 3],
        [0, 0, 0, 1, 2],
    ]
    inputs = tf.constant(inputs)
    with self.test_session() as sess:
      outputs = sess.run(fconv.shift_padding_tokens_left(
          inputs, tf.constant([5, 3, 2])))
      self.assertAllEqual(outputs, expected)


if __name__ == "__main__":
  tf.test.main()
