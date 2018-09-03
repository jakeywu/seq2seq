import tensorflow as tf


class RnnAttentionModel(object):
    def __init__(self, conf):
        self.config = conf
        assert self.config.encoder_rnn_cell in ["GRU", "LSTM"]
        assert self.config.decoder_rnn_cell in ["GRU", "LSTM"]
        assert self.config.attention_option in ["Luong", "Bahdanau"]
        self._placeholder_layers()
        self._embedding_layers()
        self._encoder_layers()
        self._inference()
        self._build_train_op()
        self.sess = tf.Session()

    def _placeholder_layers(self):
        self.encoder_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="encoder_inputs")
        self.decoder_target = tf.placeholder(dtype=tf.int32, shape=[None, None], name="decoder_target")
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=None, name="keep_prob")

        self.batch_size = tf.shape(self.encoder_inputs)[0]
        self.encoder_seq_len = tf.reduce_sum(
            tf.cast(tf.not_equal(tf.cast(0, self.encoder_inputs.dtype), self.encoder_inputs), tf.int32), axis=-1
        )
        self.decoder_seq_len = tf.reduce_sum(
            tf.cast(tf.not_equal(tf.cast(0, self.decoder_target.dtype), self.decoder_target), tf.int32), axis=-1
        )
        self.mask_seq = tf.sequence_mask(self.decoder_seq_len, tf.reduce_max(self.decoder_seq_len, name="max target length"))

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
            encoder_state = tf.concat([state_fw.h, state_bw.h], axis=1)
            encoder_outputs = tf.nn.dropout(tf.concat([output_fw, output_bw], axis=2), keep_prob=self.keep_prob)
            if not self.config.beam_search:
                self.encoder_state = encoder_state
                self.encoder_outputs = encoder_outputs
            else:
                self.encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=self.config.beam_width)
                self.encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=self.config.beam_width)
                self.encoder_seq_len = self.encoder_seq_len * self.config.beam_width
                self.batch_size = self.batch_size * self.config.beam_width

    def __create_attention_mechanism(self):
        if self.config.attention_option == "Luong":
            return tf.contrib.seq2seq.LuongAttention(
                num_units=self.config.attention_layer_size, memory=self.encoder_outputs,
                memory_sequence_length=self.encoder_seq_len)
        return tf.contrib.seq2seq.BahdanauAttention(
            num_units=self.config.attention_layer_size, memory=self.encoder_outputs,
            memory_sequence_length=self.encoder_seq_len)

    def __create_decoder_cell(self):
        if self.config.decoder_rnn_cell == "GRU":
            single_cell = tf.contrib.rnn.GRUCell(self.config.decoder_rnn_num)
        else:
            single_cell = tf.contrib.rnn.BasicLSTMCell(self.config.decoder_rnn_num)
        return tf.nn.rnn_cell.MultiRNNCell(tf.contrib.rnn.DropoutWrapper(single_cell, self.config.keep_prob))

    def _inference(self):
        decoder_cell = self.__create_decoder_cell()
        attention_mechanism = self.__create_attention_mechanism()
        self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
            decoder_cell, attention_mechanism, self.config.attention_layer_size)
        self.decoder_initial_state = self.decoder_cell.zero_state(self.batch_size, tf.float32).clone(self.encoder_state)
        self.output_layer = tf.layers.Dense(
            self.config.decoder_vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        training_helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=self.decoder_emb_inp, sequence_length=self.decoder_seq_len, time_major=False)
        training_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=self.decoder_cell, helper=training_helper,
            initial_state=self.decoder_initial_state, output_layer=self.output_layer
        )
        final_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder, output_time_major=False)
        self.logits = final_outputs[0]

    def _build_train_op(self):
        self.loss = tf.contrib.seq2seq.sequence_loss(
            logits=self.logits, targets=self.decoder_target, weights=self.mask_seq)
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.config.max_gradient_norm)
        optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(clipped_gradients, params))

    def __search_decoder(self):
        start_tokens = tf.ones([self.batch_size, 1], tf.int32)
        end_token = tf.ones([self.batch_size, 1], tf.int32)
        if self.config.beam_search:
            inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=self.decoder_cell, embedding=self.decoder_emb_inp, start_token=start_tokens,
                end_token=end_token, initial_state=self.decoder_initial_state, beam_width=self.config.beam_width,
                output_layer=self.output_layer
            )
        else:
            decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding=self.decoder_emb_inp, start_tokens=start_tokens, end_token=end_token)
            inference_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=self.decoder_cell, helper=decoding_helper, initial_state=self.decoder_initial_state,
                output_layer=self.output_layer)

        decoder_outputs = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder, maximum_iterations=self.decoder_seq_len)

    def train(self, encoder_inputs, decoder_target):
        self.sess.run(fetches=[], feed_dict={
            self.encoder_inputs: encoder_inputs, self.decoder_target: decoder_target, self.keep_prob: 0.5
        })

    def test(self):
        pass

    def predict(self):
        pass
