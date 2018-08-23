import tensorflow as tf


class RnnAttentionModel(object):
    def __init__(self, conf):
        self.config = conf

    def _placeholder_layers(self):
        self.inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="inputs")
        self.targets = tf.placeholder(dtype=tf.int32, shape=[None], name="targets")
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=None, name="keep_prob")

        self.seq_length = tf.reduce_sum(
            tf.cast(tf.not_equal(tf.cast(0, self.inputs.dtype), self.inputs), tf.int32), axis=-1
        )

    def _embedding_layers(self):
        with tf.variable_scope(name_or_scope="embedding_layers"):
            embedding_matrix = tf.get_variable(
                name="embedding_matrix", shape=[self.config.vocab_size, self.config.embedding_size], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
            )
            self.embedded_inputs = tf.nn.embedding_lookup(params=embedding_matrix, ids=self.inputs)
            # [B * T * D]
            self.origin_shape = tf.shape(self.embedded_inputs)

    def _encoder_layers(self):
        with tf.variable_scope(name_or_scope="sentence_encoder_layers"):
            cell_fw = tf.nn.rnn_cell.GRUCell(num_units=self.config.gru_num)
            cell_bw = tf.nn.rnn_cell.GRUCell(num_units=self.config.gru_num)

            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw, cell_bw=cell_bw, inputs=self.embedded_inputs,
                sequence_length=self.seq_length,
                dtype=tf.float32, time_major=False
            )
            self.encoder_outputs = tf.nn.dropout(tf.concat([output_fw, output_bw], axis=2), keep_prob=self.keep_prob)

    def _attention_layers(self):
        with tf.variable_scope("sentence_attention_layers"):
            w_1 = tf.get_variable(
                name="w_1", shape=[2 * self.config.gru_num, self.config.attention_size],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            b_1 = tf.get_variable(name="b_1", shape=[self.config.attention_size], initializer=tf.constant_initializer(0.))
            u = tf.get_variable(
                name="w_2", shape=[self.config.attention_size, 1], initializer=tf.truncated_normal_initializer(stddev=0.1))
            v = tf.nn.xw_plus_b(tf.reshape(self.encoder_outputs, [-1, 2 * self.config.gru_num]), w_1, b_1)  # B*T*A
            s = tf.matmul(v, u)
            alphas = tf.nn.softmax(tf.reshape(s, [self.origin_shape[0], 1, self.origin_shape[1]]))
            self.sentence_attention_output = tf.reduce_sum(tf.matmul(alphas, self.encoder_outputs), axis=1)
