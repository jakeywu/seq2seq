import tensorflow as tf
from seq2seq_model import RnnAttentionModel

tf.flags.DEFINE_string("attention_option", "Luong", "choose one type of attention Luong |Bahdanau")

tf.flags.DEFINE_float("learning_rate", 0.001, "learning rate ")
tf.flags.DEFINE_integer("max_gradient_norm", 5, "gradient num")
tf.flags.DEFINE_integer("num_units", 128, "Network size")
tf.flags.DEFINE_string("unit_type", "LSTM", "LSTM | GRU ")
tf.flags.DEFINE_string("encoder_type", "Single", "Single|Bi")
tf.flags.DEFINE_integer("num_layers", 2, "if Single, we can use layers. Bi layers === 1")
tf.flags.DEFINE_integer("vocab_size", 5000, "vocab size")
tf.flags.DEFINE_integer("embedding_size", 128, "dimension of vocab")
tf.flags.DEFINE_integer("batch_size", 64, "sentence number of each sequence")
tf.flags.DEFINE_integer("epoch", 5, "epoch size")
tf.flags.DEFINE_string("log_dir", "log_dir/", "loss log")

FLAG = tf.flags.FLAGS


def main(_):
    ram = RnnAttentionModel(FLAG)
    ram.train(FLAG)
    ram.test(FLAG)


if __name__ == "__main__":
    tf.app.run(main)
