import tensorflow as tf
from transformers import TFAutoModel


class LstmBertweet(tf.keras.Model):

    def __init__(self, num_classes):
        super(LstmBertweet, self).__init__()
        self.bert = TFAutoModel.from_pretrained("vinai/bertweet-base", from_pt=True)
        self.bi_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))
        self.global_pool = tf.keras.layers.GlobalMaxPool1D()
        self.hidden_layer = tf.keras.layers.Dense(50, activation='relu')
        self.droput = tf.keras.layers.Dropout(0.2)
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        ids = inputs['input_ids']
        masks = inputs['attention_mask']
        x = self.bert(ids, attention_mask=masks)[0]
        x = self.bi_lstm(x)
        x = self.global_pool(x)
        x = self.hidden_layer(x)
        x = self.droput(x)
        return self.classifier(x)

    def build_graph(self, seq_len):
        x1 = tf.keras.layers.Input(shape=(seq_len,), name='input_ids', dtype='int32')
        x2 = tf.keras.layers.Input(shape=(seq_len,), name='attention_mask', dtype='int32')
        return tf.keras.Model(inputs=[x1, x2], outputs=self.call({'input_ids': x1, 'attention_mask': x2}))
