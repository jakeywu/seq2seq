import tensorflow as tf
from data_utils import PrePareQaData


class RnnAttentionModel(object):
    def __init__(self, conf):
        self.config = conf
        assert self.config.unit_type in ["GRU", "LSTM"]
        assert self.config.encoder_type in ["Single", "Bi"]
        assert self.config.attention_option in ["Luong", "Bahdanau"]
        self.sess = tf.Session()
        self.SOS = 1
        self.EOS = 2
        self.PAD = 0
        self._placeholder_layers()
        self._embedding_layers()
        self._inference()
        self._build_train_op()

    def _placeholder_layers(self):
        """
        接收输入信息
        """
        self.encoder_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="encoder_inputs")
        self.decoder_train = tf.placeholder(dtype=tf.int32, shape=[None, None], name="decoder_targets")
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=None, name="keep_prob")
        self.batch_size = tf.shape(self.encoder_inputs)[0]
        # <PAD> <SOS> <EOS> <UNKNOWN> 分别对应0, 1, 2, 3 //填充符-开始符-结束符-未登录符
        self.start_token = tf.fill([self.batch_size, 1], self.SOS)
        self.end_token = tf.fill([self.batch_size, 1], self.EOS)

        self.encoder_seq_len = tf.reduce_sum(
            tf.cast(tf.not_equal(tf.cast(self.PAD, self.encoder_inputs.dtype), self.encoder_inputs), tf.int32), axis=-1
        )
        self.decoder_seq_len = tf.reduce_sum(
            tf.cast(tf.not_equal(tf.cast(self.PAD, self.decoder_train.dtype), self.decoder_train), tf.int32), -1
        )

        self.decoder_train_inputs = tf.concat([self.start_token, self.decoder_train], axis=1)
        self.decoder_train_targets = tf.concat([self.decoder_train, self.end_token], axis=1)
        self.mask_seq_len = tf.sequence_mask(
            self.decoder_seq_len, tf.shape(self.decoder_train_targets)[1], dtype=tf.float32)

    def _embedding_layers(self):
        """转换字向量"""
        with tf.variable_scope(name_or_scope="embedding_layers"):
            encoder_emb_matrix = tf.get_variable(
                name="encoder_emb_matrix", shape=[self.config.vocab_size, self.config.embedding_size],
                dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
            )
            self.encoder_emb_inp = tf.nn.embedding_lookup(params=encoder_emb_matrix, ids=self.encoder_inputs)

            decoder_emb_matrix = tf.get_variable(
                name="decoder_emb_matrix", shape=[self.config.vocab_size, self.config.embedding_size],
                dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
            )
            self.decoder_emb_inp = tf.nn.embedding_lookup(params=decoder_emb_matrix, ids=self.decoder_train_inputs)

    def _inference(self):
        """
        推理
        1. encoder
        2. decoder
        """
        with tf.variable_scope("encoder_attention"):
            self._build_encoder_layers()  # encoder layers
            attention_mechanism = self._create_attention_mechanism()  # attention mechanism

        with tf.variable_scope("prepare_decoder"):
            decoder_cell = self._combine_single_cell()  # 构建decoder cell
            self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                decoder_cell, attention_mechanism, self.config.num_units)

            self.decoder_initial_state = self.decoder_cell.zero_state(
                self.batch_size, tf.float32).clone(cell_state=self.encoder_state)

            self.output_layer = tf.layers.Dense(
                self.config.vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        with tf.variable_scope("training_decoder"):
            training_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=self.decoder_emb_inp, sequence_length=self.decoder_seq_len+1, time_major=False)
            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=self.decoder_cell, helper=training_helper,
                initial_state=self.decoder_initial_state, output_layer=self.output_layer
            )
            # 返回 final_outputs, final_state, final_sequence_lengths
            final_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder, impute_finished=True)
        self.logits = final_outputs[0]

    def _build_train_op(self):
        self.loss = tf.contrib.seq2seq.sequence_loss(
            logits=self.logits, targets=self.decoder_train_targets, weights=self.mask_seq_len)
        tf.summary.scalar('loss', self.loss)
        self.summary_op = tf.summary.merge_all()
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.config.max_gradient_norm)
        optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(clipped_gradients, params))

    def _build_encoder_layers(self):
        with tf.variable_scope(name_or_scope="encoder_layers"):
            if self.config.encoder_type == "Single":
                encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                    cell=self._combine_single_cell(), inputs=self.encoder_emb_inp,
                    sequence_length=self.encoder_seq_len, dtype=tf.float32, time_major=False
                )
            elif self.config.encoder_type == "Bi":
                (output_fw, output_bw), encoder_state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=self._combine_single_cell()[0],
                    cell_bw=self._combine_single_cell()[0],
                    inputs=self.encoder_emb_inp, sequence_length=self.encoder_seq_len,
                    dtype=tf.float32, time_major=False
                )
                encoder_outputs = tf.nn.dropout(tf.concat([output_fw, output_bw], axis=2), keep_prob=self.keep_prob)
            else:
                raise ValueError("encoder type must be in [Single, Bi]")

            if not self.config.beam_search:
                self.encoder_state = encoder_state
                self.encoder_outputs = encoder_outputs
            else:
                self.encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=self.config.beam_width)
                self.encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=self.config.beam_width)
                # from tensorflow.python.util import nest
                # self.encoder_state = nest.map_structure(lambda s: tf.contrib.seq2seq.tile_batch(s, self.config.beam_width), encoder_state)
                self.encoder_seq_len = tf.contrib.seq2seq.tile_batch(self.encoder_seq_len, multiplier=self.config.beam_width)
                self.batch_size = self.batch_size * self.config.beam_width

    def _combine_single_cell(self):
        return tf.contrib.rnn.MultiRNNCell([self._single_cell() for _ in range(self.config.num_layers)])

    def _single_cell(self):
        with tf.variable_scope(name_or_scope="single_cell"):
            if self.config.unit_type == "GRU":
                single_cell = tf.contrib.rnn.GRUCell(num_units=self.config.num_units)
            elif self.config.unit_type == "LSTM":
                single_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.config.num_units)
            else:
                raise ValueError("unit_type must in [GRU, LSTM]")
            return tf.contrib.rnn.DropoutWrapper(single_cell, self.keep_prob)

    def _create_attention_mechanism(self):
        if self.config.attention_option == "Luong":
            return tf.contrib.seq2seq.LuongAttention(
                num_units=self.config.num_units, memory=self.encoder_outputs,
                memory_sequence_length=self.encoder_seq_len)
        return tf.contrib.seq2seq.BahdanauAttention(
            num_units=self.config.num_units, memory=self.encoder_outputs,
            memory_sequence_length=self.encoder_seq_len)

    def __search_decoder(self):
        if self.config.beam_search:
            inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=self.decoder_cell, embedding=self.decoder_emb_inp, start_token=self.start_token,
                end_token=self.EOS, initial_state=self.decoder_initial_state, beam_width=self.config.beam_width,
                output_layer=self.output_layer
            )
        else:
            decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding=self.decoder_emb_inp, start_tokens=self.start_token, end_token=self.EOS)
            inference_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=self.decoder_cell, helper=decoding_helper, initial_state=self.decoder_initial_state,
                output_layer=self.output_layer)

        final_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=inference_decoder)

    def train(self, flag):
        self.sess.run(tf.global_variables_initializer())
        print("begin train ...")
        step = 0
        _iter = 0
        for i in range(flag.epoch):
            pqd = PrePareQaData(flag, "train")
            for encoder_input, decoder_target in pqd:
                step += len(encoder_input)
                _iter += 1
                loss = self.sess.run(fetches=[self.logits], feed_dict={
                    self.encoder_inputs: encoder_input, self.decoder_train: decoder_target, self.keep_prob: 0.5
                })
                import pdb
                pdb.set_trace()
                print("<Train>\t Epoch:[%d] Iter[%d] Step:[%d] Loss[%.3f]" % (i+1, _iter, step, loss))

    def test(self, flag):
        print("begin train ...")
        step = 0
        _iter = 0
        pqd = PrePareQaData(flag, "test")
        for encoder_input, decoder_target in pqd:
            step += len(encoder_input)
            _iter += 1
            loss = self.sess.run(fetches=[self.loss], feed_dict={
                self.encoder_inputs: encoder_input, self.decoder_train: decoder_target, self.keep_prob: 1.
            })
            print("<Train>\t Iter[%d] Step:[%d] Loss[%.3f]" % (_iter, step, loss))

    def predict(self):
        pass
