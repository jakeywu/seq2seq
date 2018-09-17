import tensorflow as tf
from data_utils import PrePareQaData


class RnnAttentionModel(object):
    def __init__(self, conf):
        self.config = conf
        assert self.config.unit_type in ["GRU", "LSTM"]
        assert self.config.encoder_type in ["Single", "Bi"]
        assert self.config.attention_option in ["Luong", "Bahdanau"]
        self.checkpoint_path = "model/dialog"
        self.sess = tf.Session()
        self.SOS = 1
        self.EOS = 2
        self.PAD = 0
        self._placeholder_layers()
        self._embedding_layers()
        self._train_inference()
        self._prediction_decoder()
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
        self.encoder_seq_len = tf.reduce_sum(
            tf.cast(tf.not_equal(tf.cast(self.PAD, self.encoder_inputs.dtype), self.encoder_inputs), tf.int32), axis=-1
        )
        self.decoder_seq_len = tf.reduce_sum(
            tf.cast(tf.not_equal(tf.cast(self.PAD, self.decoder_train.dtype), self.decoder_train), tf.int32), -1
        )

        self.decoder_train_inputs = tf.concat([tf.fill([self.batch_size, 1], self.SOS), self.decoder_train], axis=1)
        self.decoder_train_targets = tf.concat([self.decoder_train, tf.fill([self.batch_size, 1], self.EOS)], axis=1)
        self.max_target_sequence_length = tf.reduce_max(self.decoder_seq_len)
        self.mask_seq_len = tf.sequence_mask(
            self.decoder_seq_len, tf.shape(self.decoder_train_targets)[1], dtype=tf.float32)

    def _embedding_layers(self):
        """转换字向量"""
        with tf.variable_scope(name_or_scope="embedding_layers"):
            self.emb_matrix = tf.get_variable(
                name="encoder_emb_matrix", shape=[self.config.vocab_size, self.config.embedding_size],
                dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
            )
            self.encoder_emb_inp = tf.nn.embedding_lookup(params=self.emb_matrix, ids=self.encoder_inputs)
            self.decoder_emb_inp = tf.nn.embedding_lookup(params=self.emb_matrix, ids=self.decoder_train_inputs)

    def _train_infer(self):
        with tf.variable_scope("train_encoder_attention"):
            self._build_encoder_layers()  # encoder layers
            attention_mechanism = self._create_attention_mechanism()  # attention mechanism
        with tf.variable_scope("train_prepare_decoder"):
            decoder_cell = self._create_rnn_cells()  # 构建decoder cell
            self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                decoder_cell, attention_mechanism, self.config.num_units)
            self.decoder_initial_state = self.decoder_cell.zero_state(
                self.batch_size, tf.float32).clone(cell_state=self.encoder_state)

            self.output_layer = tf.layers.Dense(
                self.config.vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

    def _train_inference(self):
        """
        推理
        1. encoder
        2. decoder
        """
        self._train_infer()
        with tf.variable_scope("training_decoder"):
            training_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=self.decoder_emb_inp, sequence_length=self.decoder_seq_len+1, time_major=False)
            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=self.decoder_cell, helper=training_helper,
                initial_state=self.decoder_initial_state, output_layer=self.output_layer
            )
            # 返回 final_outputs, final_state, final_sequence_lengths
            final_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=training_decoder
            )
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
                    cell=self._create_rnn_cells(), inputs=self.encoder_emb_inp,
                    sequence_length=self.encoder_seq_len, dtype=tf.float32, time_major=False
                )
            elif self.config.encoder_type == "Bi":
                (output_fw, output_bw), encoder_state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=tf.contrib.rnn.LSTMCell(num_units=self.config.num_units),
                    cell_bw=tf.contrib.rnn.LSTMCell(num_units=self.config.num_units),
                    inputs=self.encoder_emb_inp, sequence_length=self.encoder_seq_len,
                    dtype=tf.float32, time_major=False
                )
                encoder_outputs = tf.nn.dropout(tf.concat([output_fw, output_bw], axis=2), keep_prob=self.keep_prob)
            else:
                raise ValueError("encoder type must be in [Single, Bi]")

            self.encoder_state = encoder_state
            self.encoder_outputs = encoder_outputs

    def _saver(self):
        saver = tf.train.Saver()
        saver.save(sess=self.sess, save_path=self.checkpoint_path)

    def _create_rnn_cells(self):
        def single_rnn_cell():
            if self.config.unit_type == "GRU":
                single_cell = tf.contrib.rnn.GRUCell(num_units=self.config.num_units)
            elif self.config.unit_type == "LSTM":
                single_cell = tf.contrib.rnn.LSTMCell(num_units=self.config.num_units)
            else:
                raise ValueError("unit_type must in [GRU, LSTM]")
            single_cell = tf.contrib.rnn.DropoutWrapper(single_cell, self.keep_prob)
            return single_cell

        with tf.variable_scope(name_or_scope="create_rnn_cells"):
            return tf.contrib.rnn.MultiRNNCell([single_rnn_cell() for _ in range(self.config.num_layers)])

    def _create_attention_mechanism(self):
        if self.config.attention_option == "Luong":
            return tf.contrib.seq2seq.LuongAttention(
                num_units=self.config.num_units, memory=self.encoder_outputs,
                memory_sequence_length=self.encoder_seq_len)
        return tf.contrib.seq2seq.BahdanauAttention(
            num_units=self.config.num_units, memory=self.encoder_outputs,
            memory_sequence_length=self.encoder_seq_len)

    def _prediction_decoder(self):
        start_tokens = tf.fill([self.batch_size], self.SOS)
        with tf.variable_scope("greedy_search"):
            decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding=self.emb_matrix, start_tokens=start_tokens, end_token=self.EOS)
            inference_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=self.decoder_cell, helper=decoding_helper, initial_state=self.decoder_initial_state,
                output_layer=self.output_layer)

            final_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder, maximum_iterations=30)
            self.greedy_predict = tf.expand_dims(final_outputs.sample_id, axis=-1, name="predictions")

    def train(self, flag):
        self.sess.run(tf.global_variables_initializer())
        print("begin train ...")
        step = 0
        _iter = 0
        summary_writer = tf.summary.FileWriter(flag.log_dir, graph=self.sess.graph)
        for i in range(flag.epoch):
            pqd = PrePareQaData(flag, "train")
            for encoder_input, decoder_target in pqd:
                step += len(encoder_input)
                _iter += 1
                _, summary, loss = self.sess.run(fetches=[self.train_op, self.summary_op, self.loss], feed_dict={
                    self.encoder_inputs: encoder_input, self.decoder_train: decoder_target, self.keep_prob: 0.5
                })
                summary_writer.add_summary(summary, global_step=step)
                print("<Train>\t Epoch:[%d] Iter[%d] Step:[%d] Loss[%.3f]" % (i+1, _iter, step, loss))
            self._saver()

    def test(self, flag):
        print("begin test ...")
        step = 0
        _iter = 0
        pqd = PrePareQaData(flag, "test")
        for encoder_input, decoder_target in pqd:
            step += len(encoder_input)
            _iter += 1
            loss = self.sess.run(fetches=self.loss, feed_dict={
                self.encoder_inputs: encoder_input, self.decoder_train: decoder_target, self.keep_prob: 1.
            })
            print("<Test>\t Iter[%d] Step:[%d] Loss[%.3f]" % (_iter, step, loss))
