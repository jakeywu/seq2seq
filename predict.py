import os
import codecs
import tensorflow as tf


class PredictModel(object):
    def __init__(self):
        self.model_dir = "model/"
        self._currPath = os.path.dirname(__file__)
        self.word_2_id, self.id2word = self.__load_chinese_vocab()
        self.__cnn_by_meta_graph()

    def __cnn_by_meta_graph(self):
        checkpoint_file = tf.train.latest_checkpoint(self.model_dir)
        graph = tf.Graph()
        with graph.as_default():
            self.sess = tf.Session()
            with self.sess.as_default():
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(self.sess, checkpoint_file)
                self.inputs = graph.get_operation_by_name("encoder_inputs").outputs[0]
                self.keep_prob = graph.get_operation_by_name("keep_prob").outputs[0]
                self.predictions = graph.get_operation_by_name("greedy_search/predictions").outputs[0]

    def __load_chinese_vocab(self):
        word_2_id = dict()
        id_2_word = dict()
        with codecs.open(os.path.join(self._currPath, "data/chinese_vocab.txt"), "r", "utf8") as f:
            for i, line in enumerate(f.readlines()):
                word_2_id[line.strip()] = i
                id_2_word[i] = line.strip()
        return word_2_id, id_2_word

    def __convert_question(self, question):
        _id_lst = []
        for char in question:
            _id = self.word_2_id.get(char, 3)  # index(<UNKNOWN>) == 3
            _id_lst.append(_id)
        return _id_lst

    def __get_predict_result(self, predictions):
        result = ""
        for item in predictions[0]:
            result += self.id2word.get(item[0], "")
        return result

    def predict_by_meta_graph(self, question):
        assert isinstance(question, str)
        question = self.__convert_question(question)
        predictions = self.sess.run(self.predictions, {self.inputs: [question], self.keep_prob: 1.})
        return self.__get_predict_result(predictions)


if __name__ == "__main__":
    ques = "想问下"
    pm = PredictModel()
    print(pm.predict_by_meta_graph(ques))
