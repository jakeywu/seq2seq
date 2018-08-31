import tensorflow as tf


class RnnAttentionModel(object):
    def __init__(self, conf):
        self.config = conf
        assert self.config.encoder_rnn_cell in ["GRU", "LSTM"]
        assert self.config.attention_option in ["Luong", "Bahdanau"]
        self._placeholder_layers()
        self._embedding_layers()
        self._encoder_layers()
        self._create_attention_mechanism()
        self._decoder_rnn_cell()
        self.sess = tf.Session()

    def _placeholder_layers(self):
        self.encoder_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="encoder_inputs")
        self.decoder_target = tf.placeholder(dtype=tf.int32, shape=[None, None], name="decoder_target")
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=None, name="keep_prob")

        self.encoder_seq_len = tf.reduce_sum(
            tf.cast(tf.not_equal(tf.cast(0, self.encoder_inputs.dtype), self.encoder_inputs), tf.int32), axis=-1
        )
        self.decoder_seq_len = tf.reduce_sum(
            tf.cast(tf.not_equal(tf.cast(0, self.decoder_target.dtype), self.decoder_target), tf.int32), axis=-1
        )

    def _embedding_layers(self):
        with tf.variable_scope(name_or_scope="embedding_layers"):
            encoder_emb_matrix = tf.get_variable(
                name="encoder_emb_matrix", shape=[self.config.encoder_vocab_size, self.config.encoder_embedding_size],
                dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
            )
            self.encoder_emb_inp = tf.nn.embedding_lookup(params=encoder_emb_matrix, ids=self.encoder_inputs)
            decoder_emb_matrix = tf.get_variable(
                name="decoder_emb_matrix", shape=[self.config.decoder_vocab_size, self.config.decoder_embedding_size],
                dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
            )
            self.decoder_emb_inp = tf.nn.embedding_lookup(params=decoder_emb_matrix, ids=self.decoder_target)

    def _encoder_layers(self):
        with tf.variable_scope(name_or_scope="encoder_layers"):
            if self.config.encoder_rnn_cell == "GRU":
                cell_fw = tf.nn.rnn_cell.GRUCell(num_units=self.config.encoder_rnn_num)
                cell_bw = tf.nn.rnn_cell.GRUCell(num_units=self.config.encoder_rnn_num)
            else:
                cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=self.config.encoder_rnn_num)
                cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=self.config.encoder_rnn_num)
            (output_fw, output_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw, cell_bw=cell_bw, inputs=self.encoder_emb_inp,
                sequence_length=self.encoder_seq_len,
                dtype=tf.float32, time_major=False
            )
            self.encoder_state = tf.concat([state_fw.h, state_bw.h], axis=1)
            self.encoder_outputs = tf.nn.dropout(tf.concat([output_fw, output_bw], axis=2), keep_prob=self.keep_prob)

    def _create_attention_mechanism(self):
        if self.config.attention_option == "Luong":
            self.attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units=self.config.attention_layer_size, memory=self.encoder_outputs,
                memory_sequence_length=self.encoder_seq_len)
        else:
            self.attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=self.config.attention_layer_size, memory=self.encoder_outputs,
                memory_sequence_length=self.encoder_seq_len)

    def _single_cell(self):
        if self.config.cellType == "LSTM":
            single_cell = tf.contrib.rnn.BasicLSTMCell(self.config.decoder_rnn_num)
        else:
            single_cell = tf.contrib.rnn.GRUCell(self.config.decoder_rnn_num)
        self.single_cell = tf.contrib.rnn.DropoutWrapper(single_cell, self.config.keep_prob)

    def _decoder_rnn_cell(self):
        decoder_cell = tf.nn.rnn_cell.MultiRNNCell(self.single_cell)
        self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
            decoder_cell, self.attention_mechanism, self.config.attention_layer_size)
        helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=self.decoder_emb_inp, sequence_length=self.decoder_seq_len, time_major=False)
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=self.decoder_cell, helper=helper,
            initial_state=self.decoder_cell.zero_state(self.config.batch_size, tf.float32).clone(self.encoder_state)
        )
        self.logits = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, output_time_major=False)

    def train(self, encoder_inputs, decoder_target):
        self.sess.run(fetches=[], feed_dict={
            self.encoder_inputs: encoder_inputs, self.decoder_target: decoder_target, self.keep_prob: 0.5
        })
