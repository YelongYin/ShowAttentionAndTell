import tensorflow as tf
import tensorflow.contrib as tc
import math
from six.moves import xrange

class model():

    def init_weight(self, dim_in, dim_out, name=None, stddev=1.0):
        return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev / math.sqrt(float(dim_in))), name=name)

    def init_bias(self, dim_out, name=None):
        return tf.Variable(tf.zeros([dim_out]), name=name)

    def __init__(self,config):
        self.init_h_W = self.init_weight(self.D, self.hidden_unit, name='init_h_W')
        self.init_h_b = self.init_bias(self.hidden_unit, name='init_h_b')
        self.init_c_W = self.init_weight(self.D, self.hidden_unit, name='init_c_W')
        self.init_c_b = self.init_bias(self.hidden_unit, name='init_c_b')
        self.embedding_matrix=self.init_embedding()
        self._end = 2#EOS
        self._pad = 0#PAD

    def init_LSTM(self,a):
        '''
        Initialize the hidden state and memory state of LSTM
        :param:a feture vector of img[batch_size,L,D]
        :return:h0[batch_size,hidden_unit],c0[batch_size,hidden_unit]
        '''
        a = tf.reduce_mean(a,1)
        h = tf.nn.tanh(tf.matmul(a,self.init_h_W)+self.init_h_b)
        c = tf.nn.tanh(tf.matmul(a,self.init_c_W)+self.init_c_b)
        return h, c

    def init_embedding(self,use_glove=False):
        '''
        Initalize a embedding matrix
        :return: embedding matrix[vocab_size,embedding_size]
        '''
        if(use_glove):
            #TODO use glove
        else: init_embedding = self.init_weight(self.vocab_size,self.embedding_size,name='init_embedding')
        return init_embedding

    def word_embedding(self,embedding_matrix,caption):
        '''
        Embedding caption
        :param embedding_matrix:
        :param caption:the token of caption
        :return: word_embed[n_time_step,embedding_size]
        '''
        return tf.nn.embedding_lookup(embedding_matrix,caption)

    def loss(self,logit,captions,t):
        '''
        calculate the loss of step t
        :param logit:the predicted word of step t
        :param captions:the real caption of img
        :param t:the step of LSTM
        :return:
        '''
        captions_out = captions[:, 1:]
        mask = tf.to_float(tf.not_equal(captions_out, self._pad or self._end))
        labels = tf.expand_dims(captions[:, t], 1)
        indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
        concated = tf.concat(1, [indices, labels])
        onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.vocab_size]), 1.0, 0.0)
        loss=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logit,onehot_labels)*mask[:, t])
        loss=loss/tf.to_float(self.batch_size)
        return loss

    def optimizer(self,learning_rate,loss):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return  train_op
