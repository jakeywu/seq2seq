import os
import codecs
import tensorflow as tf


class PredictModel(object):
    def __init__(self):
        self.model_dir = "model/"
        self._currPath = os.path.dirname(__file__)
        self._vocabDict = self.__load_chinese_vocab()
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
        cv = dict()
        with codecs.open(os.path.join(self._currPath, "data/chinese_vocab.txt"), "r", "utf8") as f:
            for i, line in enumerate(f.readlines()):
                cv[line.strip()] = i
        return cv

    def __convert_question(self, question):
        _id_lst = []
        for char in question:
            _id = self._vocabDict.get(char, 3)  # index(<UNKNOWN>) == 3
            _id_lst.append(_id)
        return _id_lst

    def predict_by_meta_graph(self, question):
        question = self.__convert_question(question)
        predictions = self.sess.run(self.predictions, {self.inputs: [question], self.keep_prob: 1.})
        return predictions


if __name__ == "__main__":
    ques = "你好，我想问下，月经干净了，过个四五天又来了一点点，是什么问题"
    pm = PredictModel()
    pm.predict_by_meta_graph(ques)
