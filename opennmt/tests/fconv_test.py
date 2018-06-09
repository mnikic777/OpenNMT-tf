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

  def testConv1DWeightNorm(self):
    batch_size = 5
    max_len = 4
    num_inputs = 2
    kernel_size = 3
    num_outputs = 3

    inputs = tf.random_normal(shape=[batch_size, max_len, num_inputs])
    layer = fconv.conv1d_weight_norm(
        inputs, num_outputs, kernel_size)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      output = sess.run(layer)
      self.assertAllEqual([batch_size, max_len, num_outputs], output.shape)

  def testConv1DWeightNormWithCustomPadding(self):
    batch_size = 5
    max_len = 4
    num_inputs = 2
    kernel_size = 3
    padding = 2
    num_outputs = 3

    inputs = tf.random_normal(shape=[batch_size, max_len, num_inputs])
    layer = fconv.conv1d_weight_norm(
        inputs, num_outputs, kernel_size, padding=padding)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      output = sess.run(layer)
      self.assertAllEqual([batch_size, max_len, num_outputs], output.shape)

  def testMultiStepAttention(self):
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

    layer = fconv.multi_step_attention(inputs, target_embedding, encoder_outputs)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      output, attn_score = sess.run(layer)
      self.assertAllEqual([batch_size, max_tgt_len, num_inputs], output.shape)
      self.assertAllEqual(
          [batch_size, max_tgt_len, max_src_len], attn_score.shape)


if __name__ == "__main__":
  tf.test.main()
