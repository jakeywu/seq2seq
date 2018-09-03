import tensorflow as tf
from seq2seq_model import RnnAttentionModel

tf.flags.DEFINE_string("attention_option", "Luong", "choose one type of attention Luong |Bahdanau")
tf.flags.DEFINE_integer("attention_layer_size", 64, "attention size")
tf.flags.DEFINE_bool("beam_search", True, "beam search or greedy search")
tf.flags.DEFINE_integer("beam_width", 5, "beam search size")
tf.flags.DEFINE_float("learning_rate", 0.001, "beam search size")
tf.flags.DEFINE_float("max_gradient_norm", 0.001, "beam search size")

tf.flags.DEFINE_string("encoder_rnn_cell", "LSTM", "LSTM/GRU")
tf.flags.DEFINE_integer("encoder_rnn_num", 128, "encoder rnn hidden num")
tf.flags.DEFINE_integer("encoder_vocab_size", 5000, "encoder vocab size")
tf.flags.DEFINE_integer("encoder_embedding_size", 128, "dimension of vocab")

tf.flags.DEFINE_string("decoder_rnn_cell", "LSTM", "LSTM/GRU")
tf.flags.DEFINE_integer("decoder_vocab_size", 5000, "decoder vocab size")
tf.flags.DEFINE_integer("decoder_rnn_num", 128, "decoder vocab size")
tf.flags.DEFINE_integer("decoder_embedding_size", 128, "decoder rnn hidden num")

FLAG = tf.flags.FLAGS


def main(_):
    input_x = [
        [1, 2, 3], [2, 1, 4], [5, 6, 2], [6, 9, 8], [11, 3, 2]
    ]
    input_y = [
        [1, 2], [5, 6], [6, 8], [11, 3]
    ]
    ram = RnnAttentionModel(FLAG)
    ram.train(input_x, input_y)


if __name__ == "__main__":
    tf.app.run(main)
