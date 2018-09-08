# -*- coding: utf-8 -*-
""" Seq2betterSeq framework with variational autoencoder"""

from __future__ import print_function
from __future__ import division

import math

import tensorflow as tf
import numpy as np

import tempfile
import warnings
from .methods import sigmoid, createDeterministicVar, variationalEncoding, outcomePrediction, getDecoding, levenshtein, \
    mutate_lengthconstrained
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq as s2s


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
            self.invar_target = tf.placeholder_with_default(tf.zeros([1, 1], dtype=self.data_type),
                                                            shape=(None, 1), name='invar_target')
            self.invar_inp = [tf.placeholder_with_default(tf.zeros(1, dtype=tf.int32),
                                                          shape=(None,), name='invar_inp%i' % t)
                              for t in range(self.max_seq_length)]
            # inputs for invariance prediction
            # 不変的な予測のための入力
            self.invar_pred = outcomePrediction(variationalEncoding(self.invar_inp, self.cell,
                                                                    self.vocab_size, self.embedding_dim,
                                                                    self.variational_params)[0],
                                                self.params_pred, self.which_outcome_prediction, self.use_sigmoid)
            # variationalEncoding(...)[0] = mu, vE(...)[1] = sigma
            self.loss_inv = tf.reduce_mean(tf.square(self.invar_target - self.invar_pred))
            tf.summary.scalar('loss_inv', self.loss_inv)
            # invariance loss.
            # 不変的な損失
            self.invar_importance = tf.placeholder('float')
            # bigger = more invariance
            # より大きくなると = より不変的になる
            # Loss Functions:
            # 損失関数
            self.seq2seq_importance = tf.placeholder('float')
            # between 0 and 1
            # [0, 1] 区間の値を取る。
            self.kl_importance = tf.placeholder('float')
            # between 0 and 1
            # [0, 1] 区間の値を取る。
            self.loss_lik = s2s.sequence_loss(self.traindecoderprobs, self.labels,
                                              self.weights, average_across_timesteps=False)
            self.loss_prior = tf.reduce_mean(- (0.5 / self.cell.state_size)
                                             * tf.reduce_sum(1.0 + tf.log(tf.square(self.sigma0))
                                                             - tf.square(self.z0) - tf.square(self.sigma0), 1))
            prior_mean_scaling_factor = 0.1
            self.loss_prior_mean = tf.reduce_mean((0.5 * prior_mean_scaling_factor / self.cell.state_size)
                                                  * tf.reduce_sum(tf.square(self.z0), 1))
            self.loss_pred = tf.reduce_mean(tf.square(self.outcome0 - self.outcomes))
            # MSE loss
            self.loss_joint = (self.seq2seq_importance *
                               (self.loss_lik + (self.loss_prior * self.kl_importance
                                                 + self.loss_prior_mean * (1.0 - self.kl_importance)))
                               + ((1.0 - self.seq2seq_importance) / self.outcome_var) * self.loss_pred
                               + self.invar_importance * self.loss_inv)
            tf.summary.scalar('loss_lik', self.loss_lik)
            # reconstruction loss.
            # 再構成の損失
            tf.summary.scalar('loss_prior', self.loss_prior)
            # variational KL loss
            # 変分 KL 損失
            tf.summary.scalar('loss_pred', self.loss_pred)
            # outcome-prediction loss
            # 出力予測の損失
            tf.summary.scalar('loss_joint', self.loss_joint)
            # Training Optimization:
            # 訓練の最適化
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.gvs_joint = self.optimizer.compute_gradients(self.loss_joint)
            self.capped_joint = [(tf.clip_by_value(grad, -1, 1.), var)
                                 for grad, var in self.gvs_joint if not None]
            self.train_joint = self.optimizer.apply_gradients(self.capped_joint)
            # joint training
            self.train_autoenc = self.optimizer.minimize(self.loss_lik)
            # only train autoencoder
            # autoencoder のみの訓練
            # seq2bseq portion of graph:
            # グラフ中の sequence to better sequence の部分
            self.z = tf.Variable(tf.zeros([1, self.cell.state_size]), name='latent_representation')
            self.inferred_outcome = outcomePrediction(self.z, self.params_pred,
                                                      self.which_outcome_prediction,
                                                      self.use_sigmoid)
            # Barrier functon components
            # バリア関数の部分
            BARRIER_CONST = 1e6
            # higher value = sharper barrier, so solutions are more accurately near constraint.
            # 高い値 = より鋭い障壁を作るため、 solutions はより正確に制約に近づきます。
            self.zfirst = tf.placeholder_with_default(tf.zeros([1, self.cell.state_size], dtype=self.data_type),
                                                      shape=(1, self.cell.state_size), name='zfirst')
            self.Amat = tf.placeholder_with_default(tf.diag(tf.ones(self.cell.state_size, dtype=self.data_type)),
                                                    shape=(self.cell.state_size, self.cell.state_size), name='Amat')
            self.revision_barrier_obj = self.inferred_outcome \
                                        + tf.log(1 - tf.matmul(self.z - self.zfirst,
                                                               tf.matmul(self.Amat, tf.transpose(
                                                                   self.z - self.zfirst)))) / BARRIER_CONST
            self.z_grad = tf.gradients(self.revision_barrier_obj, self.z)[0]
            self.x_hat = getDecoding(self.z, self.dec_inp, self.cell, self.vocab_size, self.embedding_dim)
            # Run TF session
            # tensorflow の session を走らせる。
            self.summary_op = tf.summary.merge_all()
            self.sess = tf.InteractiveSession(config=config)
            # On orava :
            # self.sess = tf.Session(config = tf.ConfigProtp(device_count={'GPU': 0}))
            self.sess.run(tf.global_variables_initializer())
            self.summary_writer = tf.summary.FileWriter(self.logdir, self.sess.graph)
            self.saver = tf.train.Saver()
            # end of Seq2BetterSeq creation

    def train(self, train_seqs, train_outcomes=None,
              test_seqs=None, test_outcomes=None,
              which_model='joint', seq2seq_importanceval=0.5,
              kl_importanceval=None, invar_importanceval=0.0,
              max_epochs=100, batch_size=64,
              invar_batchsize=16,
              save_name=None):
        n = len(train_seqs)
        anneal_kl = False
        anneal_invar = False
        if kl_importanceval is None:
            anneal_kl = True
            kl_importanceval = 0.0
        if invar_importanceval is None:
            anneal_invar = True
            invar_importanceval = 0.0
        if train_outcomes is None:
            if which_model != 'autoencoder':
                warnings.warn('which_model set = autoencoder since no outcomes were provided')
                which_model = 'autoencoder'
        elif len(train_outcomes) != n:
            raise ValueError('train_outcomes and train_seqs do not agree')
        idx = list(np.random.permutation(len(train_seqs)))
        seqs_shuffled = [train_seqs[i] for i in idx]
        # shuffle data.
        # データのシャッフル
        if train_outcomes is not None:
            outcomes_shuffled = [train_outcomes[i] for i in idx]
        # Train in epochs:
        train_losses = []
        if which_model == 'joint':
            lik_losses = []
            pri_losses = []
            pred_losses = []
            inv_losses = []
        if test_seqs is not None:
            test_outcome_losses = []
            test_seq2seq_losses = []
        else:
            test_outcome_losses = None
            test_seq2seq_losses = None
        for epoch in range(max_epochs):
            ind = 0
            avg_loss = 0
            # avg training loss over epoch
            if which_model == 'joint':
                avg_lik_loss = 0
                avg_pri_loss = 0
                avg_pred_loss = 0
                avg_inv_loss = 0
            num_batches = 0
            while ind < n:
                # Train batch
                if train_outcomes is None:
                    batch_outcomes = None
                else:
                    batch_outcomes = outcomes_shuffled[ind:min(n, ind + batch_size)]
                if which_model == 'joint':
                    loss_batch, lik_loss_b, pri_loss_b, pred_loss_b, inv_loss_b, summary_batch = \
                        self.train_batch(seqs_shuffled[ind:min(n, ind + batch_size)],
                                         batch_outcomes, which_model,
                                         seq2seq_importanceval, kl_importanceval, invar_importanceval, invar_batchsize)
                else:
                    loss_batch, summary_batch = self.train_batch(seqs_shuffled[ind:min(n, ind + batch_size)],
                                                                 batch_outcomes, which_model,
                                                                 seq2seq_importanceval, kl_importanceval)
                self.summary_writer.add_summary(summary_batch, epoch + num_batches / np.ceil(n / (batch_size + 1)))
                num_batches += 1.0
                avg_loss = ((num_batches - 1) / num_batches) * avg_loss + loss_batch / num_batches
                if which_model == 'joint':
                    avg_lik_loss = ((num_batches - 1) / num_batches) * avg_lik_loss + lik_loss_b / num_batches
                    avg_pri_loss = ((num_batches - 1) / num_batches) * avg_pri_loss + pri_loss_b / num_batches
                    avg_pred_loss = ((num_batches - 1) / num_batches) * avg_pred_loss + pred_loss_b / num_batches
                    avg_inv_loss = ((num_batches - 1) / num_batches) * avg_inv_loss + inv_loss_b / num_batches
                ind = ind + batch_size
            # evaluate model on test-set:
            if test_seqs is not None:
                outcome_predictions, reconstructions, outcome_error, reconstruction_error, sigmas_pred = \
                    self.predict(test_seqs, test_outcomes)
                test_outcome_losses.append(outcome_error)
                test_seq2seq_losses.append(reconstruction_error)
                train_losses.append(avg_loss)
                print('epoch: {} train_loss={} val_outcome_error={} val_reconstruction_error={}'
                      .format(epoch, avg_loss, outcome_error, reconstruction_error))
                print('  avg_predicted_sigma={}'.format(sigmas_pred))
                if which_model == 'joint':
                    if invar_importanceval > 0:
                        print(' avg_lik_loss={} avg_pri_loss={} avg_pred_loss={} avg_inv_loss={}'
                              .format(avg_lik_loss, avg_pri_loss, avg_pred_loss, avg_inv_loss))
                    else:
                        print(' avg_lik_loss={} avg_pri_loss={} avg_pred_loss={}'
                              .format(avg_lik_loss, avg_pri_loss, avg_pred_loss))
                    lik_losses.append(avg_lik_loss)
                    pri_losses.append(avg_pri_loss)
                    pred_losses.append(avg_pred_loss)
                    inv_losses.append(avg_inv_loss)
            else:
                print('epoch: {} train_loss={}'.format(epoch, avg_loss))
            if save_name is not None:
                # Save model every 10 epochs:
                self.saver.save(self.sess, save_name + '_' + str(epoch))
            if anneal_kl:
                print('kl_importance= {}'.format(kl_importanceval))
                if epoch < max_epochs / 5:
                    kl_importanceval = 0.0
                elif epoch > max_epochs / 5:
                    kl_importanceval = 1.0
                else:
                    kl_importanceval = sigmoid(-10.0 + 20.0 * epoch / max_epochs)
                    # anneal from sigmoid (-6) to sigmoid 6
                    # sigmoid (-6) から sigmoid 6 へアニール(焼きなまし)する。
            if anneal_invar:
                print('invar_importance= {}'.format(invar_importanceval))
                if epoch < max_epochs / 5:
                    invar_importanceval = 0.0
                elif epoch > max_epochs / 5:
                    invar_importanceval = 1.0
                else:
                    invar_importanceval = sigmoid(-15.0 + 30.0 * epoch / max_epochs)
                    # anneal from sigmoid(-9) to sigmoid 9
            self.summary_writer.flush()
        # end for over epochs.
        if save_name is not None:
            self.saver.save(self.sess, save_name + '_FINAL')
        return train_losses, test_outcome_losses, test_seq2seq_losses

    def train_batch(self, batch_seqs, batch_outcomes=None, which_model='joint',
                    seq2seq_importanceval=0.5,
                    kl_importancevval=0.0, invar_importanceval=0.0, invar_batch_size=16):
        """
        English:
        Train variational model using provided mini-batch.
        Args:
            batch_seqs: List of Length batch_size where ith element is sequence (list of strings).
            batch_outcomes: List of Length batch_size corresponding to outcomes for each sequence.
                            Can be omitted if there are no labels for these sequences.
            which_model: Determines the model to train, one of: 'joint' or '' (to train both models).
            seq2seq_importanceval: The relative weighting of seq2seq model's loss in the training,
                                   only used if which_model = 'joint'.
        Japanese:
        提供されたミニバッチを使用して変分モデルを訓練します。
        引数:
            batch_seqs: batch_size 分のリスト (i番目の要素はシーケンス (つまりstringのリスト) )
            batch_outcomes: それぞれのシーケンスの出力に対応している batch_size 文のリスト
            which_model: 訓練するモデルの指定: 'joint' を指定することで、いずれのモデルも訓練できます
            seq2seq_importanceval: 訓練する際の、seq2seq のモデルの損失に関する重み付け。
                                   'joint' を which_model に指定したときに用いられます。
        """
        batch_size = len(batch_seqs)
        epsilons = np.random.normal(size=[batch_size, self.cell.state_size])
        # noise variable
        if batch_outcomes is None:
            if which_model != 'autoencoder':
                warnings.warn('which_model set = autoencoder since no outcomes were provided')
                which_model = 'autoencoder'
            batch_outcomes = np.zeros((batch_size,))
        else:
            if len(batch_outcomes) != batch_size:
                raise ValueError('batch_seqs and batch_outcomes do not agree')
            batch_outcomes = np.array(batch_outcomes)
        batch_outcomes = batch_outcomes.reshape(len(batch_outcomes), 1)
        index_seqs = self.convertToIndexSeq(batch_seqs)
        # reformatted seq in terms of indices in vocab
        # 語彙のインデックスの形式で再フォーマットされたシーケンス
        # Dimshuffle to seq_len * batch_size
        # 次元の入れ替え batch_size * seq_len -> seq_len * batch_size
        index_array2 = np.array(index_seqs[:]).T
        # converted inputs (may be reversed)
        index_array = np.array(self.convertToEncoderSeq(batch_seqs)).T
        # converted inputs (may be reversed)
        feed_dict = {self.enc_inp[t]: index_array[t] for t in range(self.max_seq_length)}
        feed_dict.update({self.labels[t]: index_array2[t] for t in range(self.max_seq_length)})
        feed_dict.update({self.outcomes: batch_outcomes})
        feed_dict.update({self.seq2seq_importance: seq2seq_importanceval})
        feed_dict.update({self.kl_importance: kl_importancevval})
        feed_dict.update({self.epsilon_vae: epsilons})
        feed_dict.update({self.invar_importance: invar_importanceval})
        if invar_importanceval > 0:
            # Train E, F w.r.t. invariance loss:
            inv_target_vals, inv_inputs = self.get_invar_inputs(invar_batch_size)
            feed_dict.update({self.invar_inp[t]: inv_inputs[t] for t in range(self.max_seq_length)})
            feed_dict.update({self.invar_target: inv_target_vals})
        if which_model == 'joint':
            if invar_importanceval > 0:
                _, train_loss, lik_loss, pri_loss, pred_loss, inv_loss, summary = \
                    self.sess.run([self.train_joint, self.loss_joint,
                                   self.loss_lik, self.loss_prior, self.loss_pred, self.loss_inv,
                                   self.summary_op],
                                  feed_dict)
            else:
                _, train_loss, lik_loss, pri_loss, pred_loss, summary = \
                    self.sess.run([self.train_joint, self.loss_joint,
                                   self.loss_lik, self.loss_prior, self.loss_pred,
                                   self.summary_op],
                                  feed_dict)
                inv_loss = 0
            return train_loss, lik_loss, pri_loss, pred_loss, inv_loss, summary
        elif which_model == 'autoencoder':
            # train autoencoder only
            _, loss_t, summary = self.sess.run([self.train_autoenc, self.loss_lik,
                                                self.summary_op],
                                               feed_dict)
        elif which_model == 'prediction':
            # train outcome prediction only, keep autoencoder params fixed
            # 出力予測のみを訓練する, 既に計算された autoencoder のパラメータは維持します
            _, loss_t, summary = self.sess.run([self.train_pred, self.loss_pred,
                                                self.summary_op],
                                               feed_dict)
        return loss_t, summary

    def predict(self, test_seqs, test_outcomes=None, batch_size=64):
        """
        English:
        Use model for prediction (done is batches).
        Args:
            test_outcomes: the actual labels (optional).
        Returns:
            A tuple of form (predicted_outcomes, reconstructed_seqs,
                            avg_outcome_error, avg_reconstruction_error)
            where avg_outcome_error is only computed if test_outcomes is provided.
        Japanese:
        予測のためのモデル (バッチで済まされる)
        引数:
            test_outcomes: 実際のラベル (optional)
        返り値:
            (predicted_outcomes, reconstructed_seqs,
            avg_outcome_error, avg_reconstruction_error) のタプル
            avg_outcome_error は test_outcomes が与えられたときにのみ与えられます
        """
        n = len(test_seqs)
        if (test_outcomes is not None) and (len(test_outcomes) != n):
            raise ValueError('test_outcomes and test_seqs do not agree')
        predicted_outcomes = []
        reconstructed_seqs = []
        outcome_errors = []
        reconstructed_errors = []
        sigmas_pred = []
        ind = 0
        while ind < n:
            # train batch:
            if test_outcomes is None:
                batch_outcomes = None
            else:
                batch_outcomes = test_outcomes[ind:min(n, ind + batch_size)]
            b_predicted_outcomes, b_reconstructed_seqs, b_outcome_errors, b_reconstruction_errors, b_sigmas_pred = \
                self.predict_batch(test_seqs[ind:min(n, ind + batch_size)], batch_outcomes)
            predicted_outcomes += b_predicted_outcomes
            reconstructed_seqs += b_reconstructed_seqs
            reconstructed_errors += b_reconstruction_errors
            if test_outcomes is not None:
                outcome_errors += b_outcome_errors
            sigmas_pred += [b_sigmas_pred]
            ind = ind + batch_size
        return (predicted_outcomes, reconstructed_seqs,
                np.mean(np.array(outcome_errors)),
                np.mean(np.array(reconstructed_errors)),
                np.mean(np.array(sigmas_pred)))

    def predict_batch(self, batch_seqs, batch_outcomes=None):
        index_array_og = np.array(self.convertToIndexSeq(batch_seqs)).T
        # reformatted seq in terms of indices in vocab, NOT reversed, Dimshuffle to seq_len *  batch_size
        # vocab のインデックスの形で再構成されたシーケンス。(シーケンスの順序は反転されない)
        # 次元の入れ替え batch_size * seq_len -> seq_len * batch_size
        index_array = np.array(self.convertToEncoderSeq(batch_seqs)).T
        # converted inputs, could be reversed
        # input を変換するが、こちらはシーケンスの順序が反転される
        feed_dict = {self.enc_inp[t]: index_array[t] for t in range(self.max_seq_length)}
        decoding_scores, outcome_predictions, pred_sigmas =\
            self.sess.run([self.decodingprobs0, self.outcome0, self.sigma0], feed_dict)
        decodings = np.array([logits_t.argmax(axis=1) for logits_t in decoding_scores])
        if batch_outcomes is not None:
            batch_outcomes = np.array(batch_outcomes)
            batch_outcomes = batch_outcomes.reshape(len(batch_outcomes), 1)
            outcome_errors = np.square(outcome_predictions - batch_outcomes)
            outcome_errors = [outcome_errors[i][0] for i in range(len(batch_seqs))]
        else:
            outcome_errors = None
        reconstruction_errors = list(np.sum(index_array_og != decodings, axis=0))
        # count of number of errors in each sentence
        # それぞれの単語の誤りの数
        predicted_outcomes = [outcome_predictions[i][0] for i in range(len(batch_seqs))]
        reconstructed_seqs = [[self.vocab[vocab_index] for vocab_index in list(decodings[:, i])]
                              for i in range(batch_seqs)]
        return (predicted_outcomes, reconstructed_seqs, outcome_errors,
                reconstruction_errors, np.mean(pred_sigmas))

    def get_invar_inputs(self, invar_batch_size):
        """
        English:
        Produces formatted inputs for invariance part of TF graph
        Japanese:
        Tensorflow のグラフの変分の部分のための変換された入力を生成する
        """
        z_externals = np.random.normal(size=[invar_batch_size, self.cell.state_size])
        decoded_seqs = []
        index_array = np.array(self.convertToEncoderSeq([[self.vocab[0], self.vocab[0]]])).T
        # converted random input
        # ランダムな入力を変換する
        feed_dict = {self.enc_inp[t]: index_array[t] for t in range(self.max_seq_length)}
        for b in range(invar_batch_size):
            z_external = z_externals[b, :].reshape([1, self.cell.state_size])
            decoded_seq = self.decode(z_external, feed_dict=feed_dict)
            decoded_seqs.append(decoded_seq)
        inv_inputs = np.array(self.convertToEncoderSeq(decoded_seqs)).T
        external_outcomes = []
        for b in range(invar_batch_size):
            z_external = z_externals[b, :].reshape([1, self.cell.state_size])
            feed_dict.update({self.z: z_external})
            outcome_external = self.sess.run(self.inferred_outcome, feed_dict)
            external_outcomes.append(outcome_external)
        inv_target_vals = np.array(external_outcomes).reshape([invar_batch_size, 1])
        return inv_target_vals, inv_inputs

    def train_invar_batch(self, batch_size):
        """
        English:
        Trains E, F such that F becomes invariant of E, D variation
        Japanese:
        F が E, D の変化に対する不変量となるように E, F を訓練する
        """
        z_externals = np.random.normal(size=[batch_size, self.cell.state_size])
        # samples from prior
        decoded_seqs = []
        index_array = np.array(self.convertToEncoderSeq([[self.vocab[0], self.vocab[0]]])).T
        # converted inputs
        feed_dict = {self.enc_inp[t]: index_array[t] for t in range(self.max_seq_length)}
        for b in range(batch_size):
            z_external = z_externals[b, :].reshape([1, self.cell.state_size])
            decoded_seq = self.decode(z_external, feed_dict=feed_dict)
            decoded_seqs.append(decoded_seq)
        index_array = np.array(self.convertToEncoderSeq(decoded_seqs)).T
        # converted decoded_seqs
        # Tragets = outcomes inferred directly from z_externals
        # Targets = z_externals から直接推論される結果
        external_outcomes = []
        for b in range(batch_size):
            z_external = z_externals[b, :].reshape([1, self.cell.state_size])
            feed_dict.update({self.z: z_external})
            outcome_external = self.sess.run(self.inferred_outcome, feed_dict)
            external_outcomes.append(outcome_external)
        external_outcomes = np.array(external_outcomes).reshape([batch_size, 1])
        feed_dict = {self.enc_inp[t]: index_array[t] for t in range(self.max_seq_length)}
        feed_dict.update({self.invar_target: external_outcomes})
        _, loss_invar = self.sess.run([self.train_inv, self.loss_inv], feed_dict)
        return loss_invar

    def fixedGradientRevise(self, input_seq, outcome_opt_learn_rate=0.05,
                            max_outcome_opt_iter=1000, print_iter=None):
        if self.which_outcome_prediction == 'linear':
            warnings.warn('Linear outcome prediction used with gradientRevise')
        if print_iter is None:
            print_iter = math.ceil(max_outcome_opt_iter / 10.0)
        index_seq = self.convertToIndexSeq(input_seq)
        input_trans = np.array(index_seq).T
        feed_dict = {self.enc_inp[t]: input_trans[t] for t in range(len(index_seq[0]))}
        outcome_init, z_val, sigmas_init = self.sess.run([self.outcome0, self.z0, self.sigma0],
                                                         feed_dict)
        sigmas = sigmas_init[0]
        avg_sigma_init = np.mean(sigmas)
        z_init = np.copy(z_val)
        reconstruct_init = self.decode(z_init, feed_dict)
        for i in range(max_outcome_opt_iter):
            g, obj = self.sess.run([self.z_grad, self.inferred_outcome], feed_dict={self.z: z_val})
            z_val += g * outcome_opt_learn_rate
            if (print_iter > 0) and (i % print_iter == 0):
                print('iter={} obj={}'.format(i, obj - outcome_init))
        z_star = np.copy(z_val)
        x_star = self.decode(z_star, feed_dict)
        edit_dist = levenshtein(input_seq[0], x_star)
        obj = np.ravel(obj)[0]
        outcome_init = np.ravel(outcome_init)[0]
        return x_star, z_star, obj - outcome_init, obj, \
               reconstruct_init, z_init, outcome_init, avg_sigma_init, edit_dist

    def barrierGradientRevise(self, input_seq, log_alpha, outcome_opt_learn_rate=0.05,
                              max_outcome_opt_iter=1000, print_iter=None, use_adaptive=False):
        """
        English:
        Use barrier method + gradient ascent to produce revision
        of the given input sequence within constraint-set specified by posterior.
        Args:
            input_seq: list of length 1 containing the sequence to be revised (must be list!!!)
            outcome_opt_learn_rate: the learning rate used in the outcome-optimization
            max_outcome_opt_iter: the number of gradient steps to take in outcome-optimization
            print_iter: print objective val every so often (set = 0 or False to turn off printing)
        Returns:
            x_star: the optimal sequence
            z_star: the optimal latent latent_representation
            improvement: the amount of predicted improvent achieved
            objective: the predicted outcome for z_star
            z_init: the initial latent representation
            outcome_init: the predicted outcome for z_init
        Japanese:
        バリア関数を使用する
        + posterior で指定された制約された集合の中で、
          与えられた入力シーケンスの変形したそれを作るための勾配上昇
        引数:
            input_seq: 変形する配列を含む長さ 1 のリスト
                        (つまり、[text] = [[[sent0][sent1][sent2]...]] ?)
            outcome_opt_learn_rate: 出力の最適化に関する学習率
            max_outcome_opt_iter: 出力の最適化に関する勾配の深さ(勾配ステップの数)
            print_iter: 値を印刷する頻度
        """
        if self.which_outcome_prediction == 'linear':
            warnings.warn('Linear oucome prediction used with barrierGradientRevise')
        if print_iter is None:
            print_iter = math.ceil(max_outcome_opt_iter / 10.0)
        index_seq = self.convertToIndexSeq(input_seq)
        input_trans = np.array(index_seq).T
        feed_dict1 = {self.enc_inp[t]: input_seq[t] for t in range(len(index_seq[0]))}
        outcome_init, z_val, sigmas_init = self.sess.run([self.outcome0, self.z0, self.sigma0],
                                                         feed_dict1)
        sigmas = sigmas_init[0]
        avg_sigma_init = np.mean(sigmas)
        z_init = np.copy(z_val)
        reconstruct_init = self.decode(z_init, feed_dict1)
        # Get constraint-ellipse
        # 制約された楕円を得る
        min_sigma_threshold = 1e-2
        for i in range(len(sigmas)):
            if sigmas[i] < min_sigma_threshold:
                sigmas[i] = min_sigma_threshold
        log_alpha = log_alpha - (self.cell.state_size / 2) * np.log(2 * np.pi)
        sigmas_sq = np.square(sigmas)
        Covar = np.diag(sigmas_sq)
        max_log_alpha = -0.5 * np.sum(np.log(2 * np.pi * sigmas_sq))
        if log_alpha > max_log_alpha:
            warnings.warn('log_alpha ={0:f} is too learge (max ={0:f} will return no revision.'
                          .format(log_alpha, max_log_alpha))
            return input_seq[0], z_init, 0.0, np.ravel(outcome_init)[0], \
                   reconstruct_init, z_init, np.ravel(outcome_init)[0], avg_sigma_init, 0.0
        K = -2 * (np.log(np.power(2 * np.pi, self.cell.state_size / 2))
                  + 0.5 * np.sum(np.log(sigmas_sq)) + log_alpha)
        A = np.linalg.prinv(Covar) / K
        # A is matrix s.t. z^T A z < 1 corresponds to alpha-counter of N(0, diag(sigmas))
        convergence_thresh = 1e-8
        last_obj = -1e6
        feed_dict = {self.zfirst: z_init}
        feed_dict.update({self.Amat: A})
        for i in range(max_outcome_opt_iter):
            feed_dict.update({self.z: z_val})
            g, obj, inferred_outcome_val = self.sess.run([self.z_grad, self.revision_barrier_obj,
                                                          self.inferred_outcome],
                                                         feed_dict)
            step_size = outcome_opt_learn_rate * 1000 / (1000 + np.sqrt(i))
            violation = True
            while violation and (step_size >= convergence_thresh / 100.0):
                z_proposal = z_val + g * step_size
                shift = z_proposal - z_init
                if np.dot(np.dot(shift, A), shift.transpose()) < 1:
                    # we are inside constraint-set
                    # 既に制約がされた集合の中にいる。
                    violation = False
                else:
                    step_size /= 2.0
                    # keep dividing by 2 until we remain within constraint
                    # 制約内に入るまで2で割り続ける
            if step_size < convergence_thresh / 100.0:
                break
                # break out of for loop.
            z_val = z_proposal
            if (print_iter > 0) and (i % print_iter == 0):
                print('iter={} obj={}'.format(i, obj - outcome_init))
            if np.abs(obj - last_obj) < convergence_thresh:
                break
            last_obj = obj
        z_star = np.copy(z_val)
        if use_adaptive:
            print_beta = True
            x_star = self.adaptiveDecode(z_star, input_seq, print_beta=print_beta)
        else:
            x_star = self.decode(z_star, feed_dict1)
        edit_dist = levenshtein(z_star, feed_dict1)
        inferred_outcome_val = np.ravel(inferred_outcome_val)[0]
        outcome_init = np.ravel(outcome_init)[0]
        if print_iter > 0:
            shift = z_star - z_init
            print('Elliptical constraint value:{}'.format(np.dot(np.dot(shift, A), shift.transpose())))
            # should be < 1
        return x_star, z_star, inferred_outcome_val - outcome_init, inferred_outcome_val,\
               reconstruct_init, z_init, outcome_init, avg_sigma_init, edit_dist

    def randomRevise(self, input_seq, num_candidates=100, num_edits=3, print_iter=None, length_range=(10, 20)):
        if print_iter is None:
            print_iter = math.ceil(num_candidates / 10.0)
        index_seq = self.convertToIndexSeq(input_seq)
        input_trans = np.array(index_seq).T
        feed_dict = {self.enc_inp[t]: input_trans[t] for t in range(self.max_seq_length)}
        outcome_init = np.ravel(self.sess.run([self.outcome0], feed_dict))[0]
        best_revision_score = outcome_init
        revised_seq = input_seq[0]
        edit_dist = 0
        for i in range(num_candidates):
            new_seq, new_edit_dist = mutate_lengthconstrained(input_seq[0], num_edits, self.vocab,
                                                              length_range=length_range)
            new_index_seq = self.convertToIndexSeq([new_seq])
            new_trans = np.array(new_index_seq).T
            feed_dict = {self.enc_inp[t]: new_trans[t] for t in range(len(new_index_seq[0]))}
            new_outcome = np.ravel(self.sess.run([self.outcome0], feed_dict))[0]
            if new_outcome > best_revision_score:
                best_revision_score = new_outcome
                revised_seq = new_seq
                edit_dist = new_edit_dist
            if (print_iter > 0) and (i % print_iter == 0):
                print('iter={} obj={}'.format(i, best_revision_score - outcome_init))
        return revised_seq, best_revision_score - outcome_init, best_revision_score, \
               outcome_init, edit_dist

    def decode(self, z_encoding, feed_dict):
        """
        English:
        Performs decoding from a given latent state Z.
        feed_dict should contain the original sequence.
        Returns sequence in original vocabulary with padding removed.
        Japanese:
        与えられた潜在域 Z から decoding する。
        feed_dict は元のシーケンスを含んでいなければならない
        返り値は padding が省かれた元の語彙のシーケンス(つまり語彙を数値に変換した数値列ではない)
        """
        feed_dict.update({self.z: z_encoding})
        decoded_probs = self.sess.run(self.x_hat, feed_dict)
        decoded_num_seq = np.array([logits_t.argmax(axis=1) for logits_t in decoded_probs])
        decoded_vocab_seq = [self.vocab[vocab_index] for vocab_index in list(decoded_num_seq[:, 0])]
        if '<PAD>' in decoded_vocab_seq:
            # remove everything after PAD char
            # PAD 以降のすべてを削除する
            first_pad = decoded_vocab_seq.index('<PAD>')
            decoded_vocab_seq = decoded_vocab_seq[:first_pad]
        return decoded_vocab_seq

    def adaptiveDecode(self, z_encoding, x0, print_beta=False):
        """
        English:
        Decoding from a given latent Z which is biased toward x0 (must be a list of seq)
        feed_dict should contain the original sequence
        Returns sequence in original vocabulary with padding removed.
        x0: input sequence (a list of lists of length=1).
        Japanese:
        x0 (シーケンスのリスト) へ向かうように、バイアスをかけられた潜在域 Z から decoding を行う。
        feed_dict は元のシーケンスを含んでいなければならない。
        返り値は padding を省いた元の語彙のシーケンス
        x0: 入力のシーケンス (長さ1のリストのリスト)
        """
        betas = self.computeBetas(x0, print_beta)
        og_index_seq = self.convertToIndexSeq(x0)
        og_input_trans = np.apply(og_index_seq).T
        for position in range(self.max_seq_length):
            # where we are decoding currently:
            # 現在 decoding している位置
            feed_dict = {self.enc_inp[t]: og_input_trans[t] if t >= position else new_input_trans[t]
                         for t in range(self.max_seq_length)}
            feed_dict.update({self.z: z_encoding})
            decoded_probs = self.sess.run(self.x_hat, feed_dict)
            # Bias decoded_probs using betas:
            decoded_probs = self.applyBetas(betas, decoded_probs, x0)
            decoded_num_seq = np.array([logits_t.argmax(axis=1) for logits_t in decoded_probs])
            decoded_vocab_seq = [self.vocab[vocab_index]
                                 for vocab_index in list(decoded_num_seq[:, 0])]
            new_index_seq = self.convertToIndexSeq([decoded_vocab_seq])
            new_input_trans = np.array(new_index_seq).T
        if '<PAD>' in decoded_vocab_seq.index('<PAD>'):
            # remove everything after PAD char
            first_pad = decoded_vocab_seq.index('<PAD>')
            decoded_vocab_seq = decoded_vocab_seq[:first_pad]
        return decoded_vocab_seq

    def computeBetas(self, x0, print_beta=False):
        """
        English:
        Identifies betas needed for adaptive decoding
        Japanese:
        適応型の decoding のための betas を識別する
        """
        index_seqs = self.convertToIndexSeq(x0)
        # reformatted seq in terms of indices in vocab
        # vocab のインデックスの形で再構成されたシーケンス。
        index_arrays2 = np.array(index_seqs[:]).T
        # NON-converted inputs
        index_array = np.array(self.convertToEncoderSeq(x0)).T
        # converted inputs (may be reversed)
        # 変換された入力 (反転されている)
        feed_dict = {self.enc_inp[t]: index_array[t] for t in range(self.max_seq_length)}
        feed_dict.update({self.labels[t]: index_arrays2[t] for t in range(self.max_seq_length)})
        decoding_scores = self.sess.run(self.decodingprobs0, feed_dict)
        betas = np.array([0.0] * self.max_seq_length)
        max_decodings = [logits_t.max(axis=1) for logits_t in decoding_scores]
        pad_index = self.vocab.index('<PAD>')
        for t in range(self.max_seq_length):
            if t < len(x0[0]):
                x0_index_t = self.vocab.index(x0[0][t])
                # index of the t-th symbol in x0
                # x0 の　t 番目のシンボルのインデックス
            else:
                x0_index_t = pad_index
            if decoding_scores[t][0][x0_index_t] < max_decodings[t][0]:
                betas[t] = max_decodings[t] - decoding_scores[t][0][x0_index_t] + 1e-3
        if print_beta:
            print('betas:{}'.format(betas))
        return betas

    def applyBetas(self, betas, decoded_probs, x0):
        new_probs = decoded_probs[:]
        pad_index = self.vocab.index('<PAD>')
        for t in range(self.max_seq_length):
            if t < len(x0[0]):
                x0_index_t = self.vocab.index(x0[0][t])
                # index of the t-th symbol in x0
            else:
                x0_index_t = pad_index
            new_probs[t][0][x0_index_t] = decoded_probs[t][0][x0_index_t] + betas[t]
        return new_probs

    def convertToIndexSeq(self, seqs):
        """
        English:
        Reformat sequence in terms of vocab-indices for embedding RNNs
        Japanese:
        RNNへの埋め込みのために、単語シーケンスから、単語 index のシーケンスを作成する
        """
        return [np.array(self.pad_sequence(self.index_sequence(seq))) for seq in seqs]

    def convertToEncoderSeq(self, seqs, reverse_seq=False):
        """
        English:
        Reformat sequence in terms of REVERSED vocab-indices for encoder RNN
        Japanese:
        encoder RNN への入力のために、単語シーケンスから、逆順にした単語 index のシーケンスを作成する。
        """
        if reverse_seq:
            return [np.array(self.pad_sequence(self.index_sequence(seq))[::-1]) for seq in seqs]
        else:
            return [np.array(self.pad_sequence(self.index_sequence(seq))) for seq in seqs]

    def index_sequence(self, seq):
        if len(seq) > self.max_seq_length:
            raise ValueError('batch contains sequence of length > max_seq_length')
        elif self.use_unknown:
            index_seq = [self.vocab.index(s) if s in self.vocab else self.UNK_ID for s in seq]
        else:
            index_seq = []
            for s in seq:
                if s in self.vocab:
                    index_seq.append(self.vocab.index(s))
                else:
                    raise ValueError('Unknown symbol: ' + str(s))
        return index_seq

    def pad_sequence(self, index_seq):
        """
        English:
        Pads Index sequence to max_seq_length (padding added to end of sequence)
        Japanese:
        max_seq_length になるまで末尾に PAD の index を追加する
        """
        while len(index_seq) < self.max_seq_length:
            index_seq.append(self.PAD_ID)
        return index_seq

    def save(self, save_name):
        """
        English:
        Saves model to file
        Japanese:
        モデルをファイルに保存する
        """
        self.saver.save(self.sess, save_name)
        print('Model saved as: {}'.format(save_name))

    def restore(self, save_name):
        """
        English:
        Restores model from file
        Japanese:
        ファイルからモデルを取り出す
        """
        self.saver.restore(self.sess, save_name)
        print('Model restored from: {}'.format(save_name))

    def setOutcomeVar(self, variance_val):
        """
        English:
        Calculate using variance in outcomes from training data
        Japanse:
        訓練データから出力の分散を計算する。
        """
        self.outcome_var = variance_val
        print('outcome_var set to: {}'.format(self.outcome_var))

    def resetLearnRate(self, new_learn_rate):
        """
        English:
        Set learning rate for parameter-training to given value
        Japanese:
        学習率を変更する
        """
        self.learning_rate = new_learn_rate
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        print('learning rate changed to: {}'.format(new_learn_rate))
