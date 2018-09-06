import tensorflow as tf
from seq2seq_model import RnnAttentionModel

tf.flags.DEFINE_string("attention_option", "Luong", "choose one type of attention Luong |Bahdanau")
tf.flags.DEFINE_bool("beam_search", True, "if use beam search else greedy search")
tf.flags.DEFINE_integer("beam_width", 5, "beam search size")
tf.flags.DEFINE_float("learning_rate", 0.001, "beam search size")
tf.flags.DEFINE_integer("max_gradient_norm", 5, "beam search size")
tf.flags.DEFINE_integer("num_units", 64, "Network size")
tf.flags.DEFINE_string("unit_type", "LSTM", "LSTM | GRU ")
tf.flags.DEFINE_string("encoder_type", "Single", "Single|Bi")
tf.flags.DEFINE_integer("num_layers", 2, "if Single, we can use layers. Bi layers === 1")
tf.flags.DEFINE_integer("vocab_size", 5000, "vocab size")
tf.flags.DEFINE_integer("embedding_size", 64, "dimension of vocab")
tf.flags.DEFINE_integer("batch_size", 10, "sentence number of each sequence")
tf.flags.DEFINE_integer("epoch", 10, "epoch size")

FLAG = tf.flags.FLAGS


def main(_):
    ram = RnnAttentionModel(FLAG)
    ram.train(FLAG)


if __name__ == "__main__":
    tf.app.run(main)
