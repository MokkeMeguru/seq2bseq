""" Seq2betterSeq framework with variational autoencoder"""

from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

import tempfile
import warnings
from .methods import sigmoid, createDeterministicVar, variationalEncoding, outcomePrediction, getDecoding


class VariationalModel(object):
    """
    English:
    Instantiates our model.
    Args:
        sess = tf Session
        max_seq_length = length of longest sequence which will ever be seen.
        vocab = sorted set of vocabulary items. (ex. use vocab.index('a') to get integer-value for 'a'.)
        use_unknown = Use special character when unknown symbol encountered? otherwise will throw error.
        which_outcome_prediction = whether to use 'linear' or 'nonlinear' outcome prediction
        use_sigmoid = if True, then outcome-predictions are constrained to [0, 1].
        outcome_var = variance of outcomes(used for rescaling)
        logdir = None or string specifying where to log output.
    Japanese:
    このモデルに対するクラス
    引数:
    　　sess = Tensorflow の Session
        max_seq_length　= シーケンスの最大の長さ
        vocab = 整列された単語の集合 (ex. vocab.index('a') としたならば、'a' に該当する integer の値を返す)
        use_unknown = 未知の単語に遭遇したときの対応。 False ならば遭遇時にエラーを出す。
        which_outcome_prediction = 出力の予測において線形か非線形のどちらの手法を使うかを指定する。
        use_sigmoid = True ならば、出力の予測値をシグモイド関数を用いることで [0, 1] 区間内に収める。
        outcome_var = 出力の分散(これはスケーリングのために使われます)
        logdir = ログを出力するディレクトリを指定します。(None でも可)
    """

    def __init__(self, max_seq_length, vocab, use_unknown=False, learning_rate=1e-3,
                 embedding_dim=None, memory_dim=100,
                 which_outcome_prediction='nonlinear',
                 use_sigmoid=False, outcome_var=None,
                 logdir=None):
        self.max_seq_length = max_seq_length
        self.vocab = vocab
        self.vocab.append('<PAD>')
        self.PAD_ID = self.vocab.index('<PAD>')
        self.use_unknown = use_unknown
        if use_unknown:
            self.vocab.append('<UNK>')
            self.UNK_ID = self.vocab.index('<UNK>')
        self.vocab_size = len(vocab)
        # Architecture parameters:
        # パラメータの構築
        if embedding_dim is None:
            self.embedding_dim = self.vocab_size - 1
        else:
            self.embedding_dim = embedding_dim
        self.data_type = tf.float32
        self.memory_dim = memory_dim
        if logdir is None:
            self.logdir = tempfile.mktemp()
        else:
            self.logdir = logdir
        print('logdir: {}'.format(self.logdir))
        # To summarize training, run: tensorboard --logdir=/var/folders/Of/.../T/...
        # 学習の要約を確認するためには、 tensorboard --logdir=/var/folders/Of/.../T/...
        # Trainign parameters:
        # 学習のためのパラメータ
        self.learning_rate = learning_rate
        self.outcome_var = 1.0
        # outcome loss is divided by this value to re-scale.
        # Should be variance of outcomes in training data.
        # 出力の損失は、この値に割られることで再スケーリングされます。
        # 訓練データの結果(出力)の分散にする必要があります。
        if outcome_var is not None:
            print("Rescaling prediction MSE loss by outcome variance = {}"
                  .format(outcome_var + 0.1))
            self.outcome_var = outcome_var + 0.1
        # configuration of computing
        # 計算のための設定
        # Run on CPU only:
        # config = tf.ConfigProto(device_count = {'GPU': 0})
        # Run on GPU & CPU
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        ## Create TensorFlow Graph: ##
        # with tf.device('/cpu:0'):
        with tf.device('/gpu:1'):
            # inputs:
            self.enc_inp = [tf.placeholder(tf.int32, shape=(None,), name='inp%i' % t)
                            for t in range(self.max_seq_length)]
            self.labels = [tf.placeholder(tf.int32, shape=(None,), name='labels%i' % t)
                           for t in range(self.max_seq_length)]
            self.weights = [tf.ones_like(labels_t, dtype=tf.float32)
                            for labels_t in self.labels]
            # weight of each sequence position in cross-entropy loss.
            # クロスエントロピー損失における、それぞれのシーケンス位置の重み
            self.outcomes = tf.placeholder(tf.float32, shape=(None, 1), name='outcomes')
            # actual outcome-labels for each sequences
            # 実際の、それぞれのシーケンスの出力ラベル

            # Decoder input: prepend some "GO" token and drop the final token is encoder input
            # 先頭に幾つかの "GO" トークンを入れ、encoder の入力の、
            # 最後の element を取り除きます。
            self.dec_inp = ([tf.zeros_like(self.enc_inp[0], dtype=np.int32, name='GO')] +
                            self.enc_inp[:-1])
            # Setup RNN components:
            # RNN の構成
            rnn_type = 'GRU'
            # can be: 'GRU' or 'DeepGRU'
            # 'GRU' や 'DeepGRU' を使うことが出来ます
            num_rnn_layers = 2
            # only for DeepGRU
            rnn_cell = tf.nn.rnn_cell
            if rnn_type == 'GRU':
                self.cell = rnn_cell.GRUCell(memory_dim)
            else:
                cells = []
                for i in range(num_rnn_layers):
                    cells.append(rnn_cell.GRUCell(memory_dim))
                self.cell = rnn_cell.MultiRNNCell(cells)
            print('RNN type : {} \n cell_state_size : {}'
                  .format(rnn_type, self.cell.state_size))
            self.which_outcome_prediction = which_outcome_prediction
            if self.which_outcome_prediction == 'linear':
                # Linear Outcome prediction
                # 線形の出力予測
                self.weights_pred = \
                    tf.Variable(tf.truncated_normal([self.cell.state_size, 1], dtype=self.data_type),
                                name='coefficients', trainable=True)
                self.biases_pred = tf.Variable(tf.zeros([1], dtype=self.data_type),
                                               name='bias', trainable=True)
                self.use_sigmoid = False
                # no sigmoid used in linear outcome predictions.
                # 線形予測にはシグモイド関数は使いません。
            else:
                # Nonlinear Outcome prediction
                # 非線形の出力予測
                self.weights_pred = dict()
                self.biases_pred = dict()
                self.weights_pred['W1'] = \
                    tf.Variable(tf.truncated_normal([self.cell.state_size, self.cell.state_size],
                                                    dtype=self.data_type),
                                name='W1', trainable=True)
                self.weights_pred['W2'] = \
                    tf.Variable(tf.truncated_normal([self.cell.state_size, 1],
                                                    dtype=self.data_type),
                                name='W2', trainable=True)
                self.biases_pred['B1'] = tf.Variable(tf.zeros([1, self.cell.state_size],
                                                              dtype=self.data_type),
                                                     name='B1', trainable=True)
                self.biases_pred['B2'] = tf.Variable(tf.zeros([1], dtype=self.data_type),
                                                     name='B2', trainable=True)
                self.use_sigmoid = use_sigmoid
                # if True, then outcome-predictions are constrained to [0, 1]
                # True ならば、出力予測は [0, 1] 区間に収められます。
            self.params_pred = (self.weights_pred, self.biases_pred)
            # tuple of outcome-prediction parameters
            # 出力予測のパラメータのタプル
            print('Type of outcome prediction: {} \n All predictions in [0, 1] : {}'
                  .format(self.which_outcome_prediction, self.use_sigmoid))
            # Parameters to produce variational posterior:
            # variational posterior を生成するパラメータ
            self.epsilon_vae = tf.placeholder(tf.float32, shape=(None, self.cell.state_size),
                                              name='epsilon_vae')
            # noise for VAE
            # VAE のノイズ項
            weights_mu = tf.Variable(tf.truncated_normal([self.cell.state_size, self.cell.state_size],
                                                         dtype=self.data_type),
                                     name='weights_mu', trainable=True)
            biases_mu = tf.Variable(tf.zeros([1, self.cell.state_size], dtype=self.data_type),
                                    name='biases_mu', trainable=True)
            initial_sigma_bias = 12.0
            # want very large so variance in posteriors is ~0 at beginning of training.
            # この値は非常に大きくしたいので、訓練開始時の 事後分布 のばらつきは ~ 0　です。
            # (?)
            weights_sigma1 = tf.Variable(tf.truncated_normal([self.cell.state_size, self.cell.state_size],
                                                             dtype=self.data_type),
                                         name='weights_sigma1', trainable=True)
            weights_sigma2 = tf.Variable(tf.truncated_normal([self.cell.state_size, self.cell.state_size],
                                                             dtype=self.data_type),
                                         name='weights_sigma2', trainable=True)
            biases_sigma1 = tf.Variable(tf.zeros([1, self.cell.state_size],
                                                 dtype=self.data_type),
                                        name='biases_sigma1', trainable=True)
            biases_sigma2 = tf.Variable(tf.fill([1, self.cell.state_size], value=initial_sigma_bias),
                                        dtype=self.data_type,
                                        name='biases_sigma2', trainable=True)
            mu_params = dict()
            mu_params['weights'] = weights_mu
            mu_params['biases'] = biases_mu
            sigma_params = dict()
            sigma_params['weights1'] = weights_sigma1
            sigma_params['biases1'] = biases_sigma1
            sigma_params['weights2'] = weights_sigma2
            sigma_params['biases2'] = biases_sigma2
            self.variational_params = (mu_params, sigma_params)
            # Get encoding and outcome prediction:
            # encoding と 出力予測を得る。
            createDeterministicVar(self.enc_inp,
                                   self.cell,
                                   self.vocab_size,
                                   self.embedding_dim)
            self.z0, self.sigma0 = variationalEncoding(self.enc_inp, self.cell,
                                                       self.vocab_size, self.embedding_dim,
                                                       self.variational_params)
            self.outcome0 = outcomePrediction(self.z0,
                                              self.params_pred,
                                              self.which_outcome_prediction,
                                              self.use_sigmoid)
            # Used at train-time:
            self.traindecoderprobs = \
                getDecoding(tf.add(self.z0, tf.multiply(self.sigma0, self.epsilon_vae)),
                            self.dec_inp, self.cell,
                            self.vocab_size, self.embedding_dim,
                            feed_previous=False)
            # Used at test-time:
            self.decodingprobs0 = getDecoding(self.z0, self.dec_inp, self.cell,
                                              self.vocab_size, self.embedding_dim,
                                              feed_previous=True)
            # Invariance portion of graph (decoded inputs fed as enc_inp)
            # グラフの不変部分(enc_inp として供給される decode された入力部)
