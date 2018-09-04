""" Helper functions for VariationalModel class """

from __future__ import print_function
from __future__ import division

import math
import random

import tensorflow as tf
import tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq as s2s


def linearOutcomePrediction(zs, params_pred, scope=None):
    """
    English:
    Model for predictions outcomes from latent representations Z,
    zs = batch of z-vectors (encoder-states, matrix)
    Japanese:
    このモデルにおける、潜在表現Zから得られる出力の予測です。
    zs = ベクトル z のバッチ(袋)です。 (encoder の状態であり、行列です)
    (恐らく、[z_0, z_1, z_2, ...] というような意味)
    """
    with s2s.variable_scope.variable_scope(scope or "outcomepred", reuse=True):
        coefficients, bias = params_pred
        outcome_preds = tf.add(tf.matmul(zs, coefficients), bias)
    return outcome_preds


def flexibleOutcomePrediction(zs, params_pred, use_sigmoid=False, scope=None):
    """
    English:
    Model for nonlinearly predicting outcomes from latent representations Z.
    Uses a single hidden layer of pre-specified size, by default = d (the size of the RNN hidden-state)
    zs = batch of z-vectors (encoder-states, matrix)
    use_sigmoid = if True, then outcome-predictions are constrained to [0, 1]
    Japanese:
    このモデルにおける、潜在表現Zから得られる非線形な出力の予測です。
    事前にサイズ (標準では d 、つまりRNNの隠れ層の数) が指定されている、一つの隠れ層を用います。
    zs = ベクトル z のバッチ(袋)です。(encoder の状態であり、行列です。)
    use_sigmoid = これが True であるならば、出力はシグモイド関数によって [0, 1] 区間に抑えられます。
    (d は encoder のための連なったRNNの最後の隠れ層を示している考えられます。)
    """
    with s2s.variable_scope.variable_scope(scope or "outcomepred", reuse=True):
        weights_pred = params_pred[0]
        biases_pred = params_pred
        hidden1 = tf.nn.tanh(tf.add(tf.matmul(zs, weights_pred['W1']), biases_pred['B1']))
        outcome_preds = tf.add(tf.matmul(hidden1, weights_pred['W2']), biases_pred['B2'])
        if use_sigmoid:
            outcome_preds = tf.sigmoid(outcome_preds)
    return outcome_preds


def outcomePrediction(zs, params_pred, which_outcomeprediction, use_sigmoid=False, scope=None):
    if which_outcomeprediction == 'linear':
        return linearOutcomePrediction(zs, params_pred, scope=scope)
    else:
        return flexibleOutcomePrediction(zs, params_pred, scope=scope)


def getEncoding(inputs, cell, num_symbols, embedding_size, dtype=s2s.dtypes.float32, scope=None):
    """
    English:
    Model for produce encoding z from x
    zs = batch of z-vectors (encoding-states, matrix)
    Japanese:
    このモデルにおける、入力 x から潜在表現 z の生成です。
    zs =  ベクトル z のバッチ(袋)です。(encoder の状態であり、行列です。)
    """
    with s2s.variable_scope.variable_scope(scope or 'seq2seq', reuse=True):
        encoder_cell = s2s.core_rnn_cell.EmbeddingWrapper(
            cell, embedding_classes=num_symbols,
            embedding_size=embedding_size
        )
        _, encoder_state = s2s.rnn.static_rnn(encoder_cell, inputs, dtype=dtype)
        # batch_size x cell.state_size
        # batch_size だけ、cell が含まれていると考えると良いでしょう。
        return encoder_state


def variationalEncoding(inputs, cell, num_symbols, embedding_size,
                        variational_params, dtypes=s2s.dtypes.float32, scope=None):
    """
    English:
    Model for produce encoding z from x.
    zs =  batch of z-vectors (encoding-stats, matrix).
    sigmas: posterior standard devs for each dimension,
               produced using 2-layer neural net with Relu units.
    Japanese:
    このモデルにおける、入力 x から潜在表現 z の生成です。
    zs =  ベクトル z のバッチ(袋)です。(encoder の状態であり、行列です。)
    sigmas = それぞれの次元における、事後標準偏差(devs = deviations)であり、
               Relu ユニットから成る2つのレイヤーを用いて生成されます。
    variational_params = VAE 内で生成される \mu と \sigma を持っています。
    """
    min_sigma = 1e-6
    # the smallest allowable sigma value
    # 許容できる最小の偏差です。
    h_T = getEncoding(inputs, cell, num_symbols, embedding_size, dtype=dtypes, scope=scope)
    with s2s.variable_scope.variable_scope(scope or 'variational', reuse=True):
        mu_params, sigma_params = variational_params
        mu = tf.add(tf.matmul(h_T, mu_params['weights']), mu_params['biases'])
        hidden_layer_sigma = tf.nn.relu(tf.add(tf.matmul(h_T, sigma_params['weights1']),
                                               sigma_params['biases1']))
        # Relu layer of same size as h_T
        # h_T と同じサイズの Relu レイヤーです。
        sigma = tf.clip_by_value(
            tf.exp(- tf.abs(tf.add(tf.matmul(hidden_layer_sigma, sigma_params['weights2']),
                                   sigma_params['biases2']))), min_sigma, 1.0)
        return mu, sigma


def getDecoding(encoder_state, inputs, cell,
                num_symbols, embedding_size,
                feed_previous=True, output_prejection=None,
                dtype=s2s.dtypes.float32, scope=None):
    """
    English:
    Model for producing probabilities over x from z
    Japanese:
    このモデルにおける、z から x へ向かう確率を計算します。
    """
    with s2s.variable_scope.variable_scope(scope or 'seq2seq', reuse=True):
        if output_prejection is None:
            cell = s2s.core_rnn_cell.OutputProjectionWrapper(cell, num_symbols, dtype=dtype)
        decode_probs, _ = s2s.embedding_rnn_decoder(
            inputs, encoder_state, cell, num_symbols,
            embedding_size, output_projection=output_prejection,
            feed_previous=feed_previous, dtype=dtype)
    return decode_probs


def cretaeVariationalVar(inputs, cell, num_symbols, embedding_size,
                         feed_previous=False, output_projection=None,
                         dtype=s2s.dtypes.float32, scope=None):
    """
    English:
    Creates Tensorflow variables which can reused.
    Japanese:
    再利用可能な Tensorflow の変数を作ります。
    """
    with s2s.variable_scope.variable_scope(scope or 'seq2seq'):
        encoder_cell = s2s.core_rnn_cell.EmbeddingWrapper(
            cell, embedding_classes=num_symbols, embedding_size=embedding_size)
        _, encoder_state = s2s.rnn.static_rnn(encoder_cell, inputs, dtype=dtype)
        # batch_size x cell.state_size
        if output_projection is None:
            cell = s2s.core_rnn_cell.OutputProjectionWrapper(cell, num_symbols)
        decode_probs, _ = s2s.embedding_rnn_decoder(
            inputs, encoder_state, cell, num_symbols,
            embedding_size, output_projection=output_projection,
            feed_previous=feed_previous)
    return None


def createDeterministicVar(inputs, cell, num_symbols, embedding_size,
                           feed_previous=False, output_projection=None,
                           dtype=s2s.dtypes.float32, scope=None):
    """
    English:
    Creates Tensorflow variables which can be reused.
    Japanese:
    再利用可能な Tensorflow の変数を作ります。
    """
    with s2s.variable_scope.variable_scope(scope or 'seq2seq'):
        encoder_cell = s2s.core_rnn_cell.EmbeddingWrapper(
            cell, embedding_classes=num_symbols,
            embedding_size=embedding_size)
        _, encoder_state = s2s.rnn.static_rnn(encoder_cell, inputs, dtype=dtype)
        # batch_size x cell.state_size
        if output_projection is None:
            cell = s2s.core_rnn_cell.OutputProjectionWrapper(cell, num_symbols)
        decode_probs, _ = s2s.embedding_rnn_decoder(
            inputs, encoder_state, cell, num_symbols,
            embedding_size, output_projection=output_projection,
            feed_previous=feed_previous)
    return None


def levenshtein(seq1, seq2):
    """
    English:
    Computes edit distance between two (possibly padded) sequences:
    Japanese:
    2つのシーケンスにおける独自のレーベンシュタイン距離を計算する。
    (padding である '<PAD>'が加えられている可能性を考慮しています)
    (ここにおけるレーベンシュタイン距離は、
     恐らく単語ごとに分割した場合のレーベンシュタイン距離(一般には文字ごと))
    """
    s1 = [value for value in seq1 if value != '<PAD>']
    s2 = [value for value in seq2 if value != '<PAD>']
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 +
                                  min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


""" 
Info
for i1, c1 in enumerate(['a', 'b', 'c']):
    print('{} : {}'.format(i1, c1))
=> 
 0 : a
 1 : b
 2 : c
"""

def mutate_lengthconstrained(init_seq, num_edits, vocab, length_range=(10, 20)):
    """
    English:
    Preforms random edits of sequences, respecting min/max sequence-length constraints.
    At each edit, possible operations (equally likely) are:
    (1) Do nothing (2) Substitution (3) Deletion (4) Insertion
    Each operation is uniform over possible symbols and possible positions
    Japanese:
    最小/最大のシーケンスの長さに制約をかけながら、シーケンスのランダムな編集を行います。
    編集時に可能な操作は以下の4つです。
    (1) 何もしない (2) 置換 (3) 削除 (4) 挿入
    それぞれの編集は、可能なシンボル(単語など)や位置に対して均一に(偏りなく)行われます。
    """
    min_seq_length,  max_seq_length = length_range
    new_seq = init_seq[:]
    for i in range(num_edits):
        operation = random.randint(1, 4)
        # 1 = Do nothing, 2 = Substitution, 3 = Deletion, 4 = Insertion
        # 1 = 何もしない  2 = 置換  3 = 削除 4 = 挿入
        if operation > 1:
            char = '<PAD>'
            # potential character, cannot be PAD.
            # 潜在的な element であり、 <PAD> になることはない。
            # (つまり <PAD> 以外の任意の element(単語) になる)
            while char == '<PAD>':
                char = vocab[random.randint(0, len(vocab) - 1)]
            position = random.randint(0, len(new_seq) - 1)
            if (operation == 4) and (len(new_seq) < max_seq_length):
                position = random.randint(0, len(new_seq))
                new_seq.insert(position, char)
            elif (operation == 3) and (len(new_seq) > min_seq_length):
                _ = new_seq.pop(position)
            elif operation == 2:
                new_seq[position] = char
    edit_dist = levenshtein(new_seq, init_seq)
    if edit_dist > num_edits:
        raise ValueError('edit distance invalid')
    return new_seq, edit_dist


def mutate(init_seq, num_edits, vocab):
    new_seq = init_seq[:]
    for i in range(num_edits):
        operation = random.randint(1, 4)
        # 1 = Do nothing, 2 = Substitution, 3 = Deletion, 4 = Insertion
        # 1 = 何もしない  2 = 置換  3 = 削除 4 = 挿入
        if operation > 1:
            char = '<PAD>'
            # potential character, cannot be PAD.
            # 潜在的な element であり、 <PAD> になることはない。
            while char == '<PAD>':
                char = vocab[random.randint(0, len(vocab) - 1)]
            position = random.randint(0, len(new_seq) - 1)
            if operation == 4:
                position = random.randint(0, len(new_seq))
                new_seq.insert(position, char)
            elif (operation == 3) and len(new_seq) > 1:
                _ = new_seq.pop(position)
            elif operation == 2:
                new_seq[position] = char
    edit_dist = levenshtein(new_seq, init_seq)
    if edit_dist > num_edits:
        raise ValueError("edit distance invalid")
    return new_seq, edit_dist


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def smoothedsigmoid(x, b=1):
    """
    English:
    b controls smoothness, lower = smoother
    Japanese:
    b は緩やかさを調整します。b が小さいほど緩やかに(変化が小さく)なります。
    """
    return 1 / (1 + math.exp(- b * x))