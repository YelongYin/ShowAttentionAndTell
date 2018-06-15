import tensorflow as tf
import tensorflow.contrib as tc
import math
import numpy as np
import pickle

import pandas as pd
import data_utils
from config.Deconfig import Deconfig

"""
FEATURES_PATH = "/home/yyl/PycharmProjects/ShowAttentionAndTell/data/f30k/f30k_0.features.pkl"
MODEL_PATH = "/home/yyl/PycharmProjects/ShowAttentionAndTell/model/"
VOCAB_PATH = "/home/yyl/PycharmProjects/ShowAttentionAndTell/data/vocab"
"""
FEATURES_PATH = "/home/lemin/1TBdisk/PycharmProjects/ShowAttentionAndTell/data/f30k/f30k_0.features.pkl"
MODEL_PATH = "/home/lemin/1TBdisk/PycharmProjects/ShowAttentionAndTell/model/"
VOCAB_PATH = "/home/lemin/1TBdisk/PycharmProjects/ShowAttentionAndTell/data/vocab"



class model(object):

    def __init__(self, config):
        self.D = config.D
        self.L = config.L
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.hidden_units = config.hidden_unit
        self.batch_size = config.batch_size
        self.n_time_step = config.n_time_step

        self.learning_rate = config.learning_rate
        self.epochs = config.epochs
        self._end = 2
        self._pad = 0

        # variable of initial lstm
        self.init_h_W = self.init_weight(self.D, self.hidden_units, name='init_h_W')
        self.init_h_b = self.init_bias(self.hidden_units, name='init_h_b')
        self.init_c_W = self.init_weight(self.D, self.hidden_units, name='init_c_W')
        self.init_c_b = self.init_bias(self.hidden_units, name='init_c_b')

        # embedding matrix
        self.embedding_matrix = self.init_embedding()

        # variable of attention
        self.image_att_W = self.init_weight(self.D, self.D, name="image_att_W")
        self.hidden_att_W = self.init_weight(self.hidden_units, self.D, name="hidden_att_W")

        self.pre_att_b = self.init_bias(self.D, name='pre_att_b')

        self.attn_W = self.init_weight(self.D, 1, name="attn_w")
        self.attn_b = self.init_bias(1, name="attn_b")

        # variable of lstm cell
        self.lstm_W = self.init_weight(self.embedding_size, self.hidden_units * 4, name="lstm_W")
        self.lstm_U = self.init_weight(self.hidden_units, self.hidden_units * 4, name="lstm_U")
        self.image_encode_W = self.init_weight(self.D, self.hidden_units * 4, name="image_encode_W")

        self.lstm_b = self.init_bias(self.hidden_units * 4, name="lstm_b")

        # variable of predict
        self.decode_W = self.init_weight(self.hidden_units, self.embedding_size, name="decode_W")
        self.decode_b = self.init_bias(self.embedding_size, name="decode_b")

        self.decode_word_W = self.init_weight(self.embedding_size, self.vocab_size, name='decode_word_W')
        self.decode_word_b = self.init_bias(self.vocab_size, name="decode_word_b")

    def init_weight(self, dim_in, dim_out, name=None, stddev=1.0):
        return tf.get_variable(name=name, dtype=tf.float32, initializer=tf.truncated_normal([dim_in, dim_out], stddev=stddev / math.sqrt(float(dim_in))))

    def init_bias(self, dim_out, name=None):
        return tf.get_variable(name=name, dtype=tf.float32, initializer=tf.zeros([dim_out]))

    def _attention(self, _h, context_encode):
        """
        This function is used to implement attention mechanism and the attention formula is
        ai = v * tanh(u*_context + w * _h + b1) + b2
        Args:
            _h: hidden state of lstm cell [batch_size, hidden_units]
            _context:feature of image, shape:[batch_size, L, D]
        Returns:
            A tensor ai [batch_size, L]
        """
        # make _h * attn_w to a shape of [batch_size, 1, D]
        context_encode = context_encode + tf.expand_dims(tf.matmul(_h, self.hidden_att_W), 1) + self.pre_att_b
        # shape [batch_size, L, D]
        context_encode = tf.tanh(context_encode)

        # shape [batch_size * L, D]
        context_encode_flat = tf.reshape(context_encode, [-1, self.D])
        # shape [batch_size * L, 1]
        alpha = tf.matmul(context_encode_flat, self.attn_W) + self.attn_b
        # shape [batch_size, L]
        alpha = tf.reshape(alpha, [-1, self.L])
        alpha = tf.nn.softmax(alpha)

        return alpha

    def _lstm_function(self, _xt, _h, _context, _c):
        """
        This function is used to implement lstm_cell function
        Args:
             _xt: caption after embedding (shape:[batch_size, embed_size])
             _h: hidden state of lstm (shape:[batch_size, hidden_units])
             _context: the context vector after weight sum (shape:[batch_size, D)
             _c: context vector of lstm (shape:[batch_size, hidden_units]
        Returns:
            _h: hidden state of lstm (shape:[batch_size, hidden_units])
            _c: memory vector of lstm (shape:[batch_size, hidden_units])
            o:  output of lstm (shape:[batch_size, hidden_units])
        """
        # shape: [batch_size, hidden_units * 4]
        _xt = tf.matmul(_xt, self.lstm_W)
        # shape: [batch_size, hidden_units * 4]
        _ht = tf.matmul(_h, self.lstm_U)
        # shape: [batch_size, hidden_units * 4]
        _context = tf.matmul(_context, self.image_encode_W)

        lstm_preactive = _xt + _ht + _context

        # shape [batch_size, hidden_units]
        i, f, o, new_c = tf.split(lstm_preactive, 4, 1)

        # calculate the input gate
        i = tf.nn.sigmoid(i)
        # calculate the forget gate
        f = tf.nn.sigmoid(f)
        # calculate the output gate
        o = tf.nn.sigmoid(o)
        # calculate the new context vector of lstm
        new_c = tf.nn.tanh(new_c)

        _c = f * _c + i * new_c
        h = o * tf.nn.tanh(new_c)

        return h, o, _c

    def _predict(self, _h, **kwargs):
        """
        This function is used to predict time-step word(one step) and the formula is
        y_hat = Relu(h * w + b) and we have used dropout here

        Args:
            _h: hidden state of lstm (shape:[batch_size, hidden_units])
        Returns:
            logits: a vector of prediction (shape:[batch_size, vocab_size])
        """
        # shape:[batch_size, embed_size]
        logits = tf.matmul(_h, self.decode_W) + self.decode_b
        logits = tf.nn.relu(logits)
        logits = tf.nn.dropout(logits, 0.5)

        # shape:[batch_size, vocab_size]
        logits_words = tf.matmul(logits, self.decode_word_W) + self.decode_word_b
        return logits_words

    def init_LSTM(self, a):
        """
        Initialize the hidden state and memory state of LSTM
        :param:a feture vector of img[batch_size,L,D]
        :return:h0[batch_size,hidden_unit],c0[batch_size,hidden_unit]
        """
        a = tf.reduce_mean(a, 1)
        h = tf.nn.tanh(tf.matmul(a, self.init_h_W) + self.init_h_b)
        c = tf.nn.tanh(tf.matmul(a, self.init_c_W) + self.init_c_b)
        return h, c

    def init_embedding(self, use_glove=False):
        """
        Initalize a embedding matrix
        :return: embedding matrix[vocab_size,embedding_size]
        """
        init_embedding = None
        if use_glove:
            "TODO use glove"
        else:
            init_embedding = self.init_weight(self.vocab_size, self.embedding_size, name='init_embedding')
        return init_embedding

    def word_embedding(self, embedding_matrix, caption):
        """
        Embedding caption
        :param embedding_matrix:
        :param caption:the token of caption
        :return:word_embed[n_time_step,embedding_size]
        """
        return tf.nn.embedding_lookup(embedding_matrix, caption)

    def loss(self, logit, captions, t):
        """
        calculate the loss of step t
        :param logit: the predict[batch_size,vocab_size]
        :param captions: the real captions[batch_size,n_time_step]
        :param t: step of LSTM
        :return: loss
        """
        captions_out = captions[:, :]
        mask = tf.to_float(tf.not_equal(captions_out, self._end or self._pad))
        # shape:[batch_size, 1]
        labels = tf.expand_dims(captions[:, t], 1)
        # shape:[batch_size, 1]
        indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)

        # shape:[batch_size, 2]
        concated = tf.concat([indices, labels], 1)

        onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.vocab_size]), 1.0, 0.0)

        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=onehot_labels) * mask[:, t])

        loss = loss/tf.to_float(self.batch_size)

        return loss

    def optimizer(self, learning_rate, loss):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return train_op

    def build_model(self):
        """
        This function is mainly used to trained the model.
        """
        context = tf.placeholder(tf.float32, [self.batch_size, self.L, self.D])
        sentence = tf.placeholder(tf.int32, [self.batch_size, self.n_time_step])

        # get hidden state vector and memory vector
        h, c = self.init_LSTM(context)

        context_flat = tf.reshape(context, [-1, self.D])
        # make context encode can be trained
        context_encode = tf.matmul(context_flat, self.image_att_W)
        # shape: [batch_size, L, D]
        context_encode = tf.reshape(context_encode, [-1, self.L, self.D])

        loss = 0.0

        for ind in range(self.n_time_step):
            if ind == 0:
                word_emb = tf.zeros([self.batch_size, self.embedding_size])
            else:
                with tf.variable_scope(tf.get_variable_scope()):
                    tf.get_variable_scope().reuse_variables()
                    with tf.device("/cpu:0"):
                        word_emb = self.word_embedding(self.embedding_matrix, sentence[:, ind-1])

            # shape: [batch_size, 1]
            labels = tf.expand_dims(sentence[:, ind], 1)
            # shape: [batch_size, 1]
            indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
            # shape: [batch_size, 2]
            # we should concat indices vector and the labels vector to use one-hot
            concated = tf.concat([indices, labels], 1)
            # shape: [batch_size, vocab_size]
            onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.vocab_size]), 1.0, 0.0)

            alpha = self._attention(h, context_encode)

            # shape: [batch_size, L, D]
            weighted_context = tf.reduce_sum(context * tf.expand_dims(alpha, 2), 1)

            h, o, _c = self._lstm_function(word_emb, h, weighted_context, c)

            logit_words = self._predict(h)

            current_loss = self.loss(logit_words, sentence, ind)
            loss = loss + current_loss
        return loss, context, sentence

    def train(self, pretrained_model_path=None):
        # get captions and feats
        captions = data_utils.get_some_captions(5000)
        # shape:[5000, 192, 512]
        feats = data_utils.get_features(FEATURES_PATH)

        maxlen = self.n_time_step

        # get word2ix, ixtoword dictionary
        word2ix, ixtoword = data_utils.initialize_vocabulary(VOCAB_PATH)

        learning_rate = self.learning_rate
        n_words = len(word2ix)

        sess = tf.InteractiveSession()

        loss, context, sentence = self.build_model()
        saver = tf.train.Saver(max_to_keep=25)

        train_op = self.optimizer(learning_rate, loss)
        tf.initialize_all_variables().run()

        if pretrained_model_path is not None:
            print("Starting with pretrained model")
            saver.restore(sess, pretrained_model_path)

        for epoch in range(self.epochs):
            for start, end in zip(range(0, len(captions), self.batch_size),
                                  range(self.batch_size, len(captions), self.batch_size)):
                current_feats = feats[start:end]
                current_feats = current_feats.reshape(-1, self.D, self.L).swapaxes(1, 2)

                current_captions = captions[start:end]

                current_captions_ind = []
                for caption in current_captions:
                    caption2id = data_utils.sentence_to_token_ids(caption, word2ix)
                    if len(caption2id) < maxlen:
                        caption2id = [data_utils.GO_ID] + caption2id + [data_utils.EOS_ID]
                        caption2id = caption2id + [data_utils.PAD_ID] * (maxlen - len(caption2id))
                    current_captions_ind.append(caption2id)

                current_captions_ind = np.asarray(current_captions_ind)

                _, loss_value = sess.run([train_op, loss], feed_dict={
                    context: current_feats,
                    sentence: current_captions_ind
                })

                print("Epoch:%d, Current loss:" % epoch, loss_value)
            saver.save(sess, MODEL_PATH, global_step=epoch)

    def build_generator(self):
        context = tf.placeholder("float32", [self.batch_size, self.L, self.D])

        h, c = self.init_LSTM(context)
        context_flat = tf.reshape(context, [-1, self.D])
        context_encode = tf.matmul(tf.squeeze(context_flat), self.image_att_W)
        context_encode = tf.reshape(context_encode, [-1, self.L, self.D])

        word_emb = tf.zeros([1, self.embedding_size])
        generated_words=[]
        logit_list=[]
        for t in range(self.n_time_step):

            #context_encode = context_encode + tf.matmul(h, self.hidden_att_W) + self.pre_att_b
            #context_encode = tf.nn.tanh(context_encode)
            alpha = self._attention(h,context_encode)
            weighted_context = tf.reduce_sum(context * tf.expand_dims(alpha, 2), 1)
            #weighted_context = tf.reduce_sum(tf.squeeze(context_encode) * alpha, 0)
            #weighted_context = tf.expand_dims(weighted_context, 0)
            h, o, c = self._lstm_function(word_emb, h, weighted_context, c)
            logits_word = self._predict(h)
            max_prob_word = tf.argmax(logits_word, 1)

            word_emb = tf.nn.embedding_lookup(self.embedding_matrix, max_prob_word)

            generated_words.append(max_prob_word)
            logit_list.append(logits_word)

        return context, generated_words, logit_list

    def test(self, model_path):
        feats = data_utils.get_features(FEATURES_PATH)
        context, generated_words, logit_list = self.build_generator()
        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        generated_word_index = sess.run(generated_words, feed_dict={context: feats[:100]})
        word2id, id2word=data_utils.initialize_vocabulary(vocabulary_path=VOCAB_PATH)
        generated_captions = []
        for i in range(self.batch_size):
            generated_caption = [id2word[index[i]] for index in generated_word_index]
            generated_captions.append(generated_caption)
        return generated_captions


gen_model = model(Deconfig)
generated_captions = gen_model.test(MODEL_PATH+ '-746')
captions = data_utils.get_some_captions(100)

for i in range(100):
    print(captions[i])
    print(generated_captions[i])

        generated_word_index = sess.run(generated_words, feed_dict={context: feat})
        word2id, id2word=data_utils.initialize_vocabulary(vocabulary_path='')
        generated_words = [id2word[index[0]] for index in generated_word_index]
        return generated_words

"""
gen_model = model(Deconfig)
gen_model.train()
"""
