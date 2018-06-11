import tensorflow as tf
import tensorflow.contrib as tc

from six.moves import xrange

class model():

    def init_weight(self, dim_in, dim_out, name=None, stddev=1.0):
        return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev / math.sqrt(float(dim_in))), name=name)

    def init_bias(self, dim_out, name=None):
        return tf.Variable(tf.zeros([dim_out]), name=name)

    def __init__(self,config):
        self.init_h_W = self.init_weight(config.D, config.hidden_unit, name='init_h_W')
        self.init_h_b = self.init_bias(config.hidden_unit, name='init_h_b')
        self.init_c_W = self.init_weight(config.D, config.hidden_unit, name='init_c_W')
        self.init_c_b = self.init_bias(config.hidden_unit, name='init_c_b')

    def init_LSTM(self,a):
        '''
        Initialize the hidden state and memory state of LSTM
        :param:a feture vector of img[batch_size*L,D]
        :return:h0[],c0[]
        '''
        a = tf.reduce_mean(a,1)
        h = tf.nn.tanh(tf.matmul(a,self.init_h_W)+self.init_h_b)
        c = tf.nn.tanh(tf.matmul(a,self.init_c_W)+self.init_c_b)
        return h, c