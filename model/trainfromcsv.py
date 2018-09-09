# -*- coding: utf-8 -*-
"""
English:
Example for two text files.
One of files contains based sentences.
Another one contains styled sentences.
* Notice: The meanings of the sentences of the two files
            DOES NOT necessarily match.
Japanese:
2つのテキストファイルを用いた例
一つは基本となる短文を格納しているファイル
もう一つはスタイルが付与された短文が格納されているファイル
* 注意: 2つのファイルに含まれる短文は意味が一致している必要はありません。
(つまりスタイルが違う短文集を2つのファイルに分割してあればよいということです)
"""

from __future__ import print_function
from __future__ import division
from model.variationalmodel import VariationalModel
import random

import MeCab as mecab
import re
from operator import or_
from functools import reduce
import numpy as np

""" 日本語の解析のためのユーティリティ """
ws = re.compile(' ')
wakati_tagger = mecab.Tagger('-Owakati')
hiragana_tagger = mecab.Tagger('-Oyomi')


def get_word_list(sentence):
    """
    Notice: returned list will not contain '、'.
    ['今日', 'は', '良い', '天気', 'です', 'ね', '。']
    :param sentence: simple text
    :return: list of word
    """
    return [x for x in ws.split(wakati_tagger.parse(sentence)) if x != '、'][:-1]


def get_hiragana_list(sentence):
    """
    Notice: returned list will not contain '、'.
    get_word_yomi_list('今日は、良い天気ですね。')
    ['キ', 'ョ', 'ウ', 'ハ', 'ヨ', 'イ', 'テ', 'ン', 'キ', 'デ', 'ス', 'ネ', '。']
    :param sentence: simple text
    :return: list of hiragana
    """
    return [x for x in hiragana_tagger.parse(sentence) if x != '、'][:-1]


def get_word_yomi_list(sentence):
    """
    Notice: returned list will not contain '、'.
    Example: get_word_yomi_list('今日は、良い天気ですね。')
    ['キョウ', 'ハ', 'ヨイ', 'テンキ', 'デス', 'ネ', '。']
    :param sentence: simple text
    :return: list of HIRAGANA word
    """
    return [hiragana_tagger.parse(x).rstrip()
            for x in ws.split(wakati_tagger.parse(sentence)) if x != '、'][:-1]


""" utilities for dealing with text files """


def read_file(path, func):
    """
    :param path: the file's path
    :param func: sentence parse function
    :return: list of parsed sentences (list of (word, character etc. ) list)
    """
    with open(path, 'rt', encoding='utf-8') as file:
        return [func(sentence.rstrip())
                for sentence in file.readlines() if sentence != '\n']


def add_vocab(s, sentences):
    """
    add vocabulary into the set
    :param s: set (this is vocabulary set)
    :param sentences: parsed sentence's list
    :return: updated vocabulary set
    """
    return s.union(reduce(or_, [set(sentence) for sentence in sentences]))


def add_vocab_from_files(s, paths, func):
    for path in paths:
        s = add_vocab(s, read_file(path, func))
    return s


if __name__ == '__main__':
    """ Resources """
    train_val = [0.1, 0.9]
    train_path = ['../st-data/base.csv', '../st-data/styled.csv']
    parse_func = get_word_yomi_list

    train_data = list(zip(*(reduce(lambda x, y: x + y,
                                   map(lambda val_path:
                                       [[sent, val_path[0]]
                                        for sent in read_file(val_path[1], parse_func)],
                                       zip(train_val, train_path))))))
    train_data_idx = random.sample(range(len(train_data[0])), len(train_data[0]))
    train_data = list(map(lambda row: list(map(lambda idx: train_data[row][idx],
                                               train_data_idx)),
                          [0, 1]))
    test_data_idx = random.sample(range(len(train_data[0])), len(train_data[0]) // 10)
    test_data = list(map(lambda row: list(map(lambda idx: train_data[row][idx],
                                              test_data_idx)),
                         [0, 1]))

    wyl = sorted(add_vocab_from_files(set(),
                                      ['../st-data/styled.csv', '../st-data/base.csv'],
                                      parse_func))

    """ Settings """
    max_seq_length = 30
    learning_rate = 1e-3
    embedding_dim = 256
    memory_dim = 512
    log_dir = '../output/logs/'

    """ Information """
    print('sentences size: {}'
          .format(len(train_data[0])))
    print('vocaburaly size: {}'.format(len(wyl)))
    for wy in wyl[:3]:
        print('vocab id: {} element: {}'.format(wyl.index(wy), wy))
    print('example:')
    for sentence, val in zip(train_data[0][:3], train_data[1][:3]):
        print('sentence: {} val: {}'.format(sentence, val))

    """ create model """
    model = VariationalModel(max_seq_length=max_seq_length, vocab=wyl, use_unknown=True,
                             learning_rate=learning_rate, embedding_dim=embedding_dim,
                             memory_dim=memory_dim,
                             use_sigmoid=True,
                             outcome_var=np.var(np.array(train_data[1])),
                             logdir=log_dir)

    """ tensorboard --logdir=output/logs/ """

    """ 
    English:
    In practice, need to train model as follows:

    1)  First train with kl_importanceval = 0, invar_importanceval = 0
        until val_reconstruction_error + val_outcome_error plateau.

    2)  Find a good setting of seq2seq_importanceval which
        leads to low val_reconstruction_error and val_outcome_error

    3)  Carefully begin increasing kl_importanceval from 0 to 1 and continue training.
        Make sure to save your model every few training epochs.
        Anytime val_reconstruction/outcome_error suddenly becomes worse, need to: 
        - halt training and reload last model where val_reconstruction/outcome_error was still good
        - slightly lower kl_importanceval and resume training

    4)  Carefully begin increasing invar_importanceval.
        Make sure to save your model every few training epochs.
        Anytime val_reconstruction/outcome_error suddenly becomes worse, need to: 
        - halt training and reload last model where val_reconstruction/outcome_error was still good
        - slightly lower invar_importanceval and resume training
        Keep increasing invar_importanceval until val_reconstruction/outcome_error always begins to worsen. 
    Japanese:
    実際には、次のようにモデルを訓練する必要があります。
    1)  まず kl_importance = 0, invar_importance = 0 として、 
        val_reconstruction_error + val_outcome_error が大きい間訓練します。
    
    2)  val_reconstruction_error と val_outcome_error が低くなるような、
        seq2seq_importanceval の良い値を求めます。
    
    3)  kl_importanceval を 徐々に 0 から 1 へ増加させます。
        モデルを数 epoch ごとにセーブすることを忘れないで下さい。
        突然 val_reconstruction/outcome_error が悪くなることがありますが、そのときは以下のようにして下さい。
        - 訓練を中止して以前のモデルをリロードして下さい。
        - ほんの少し kl_importanceval を下げて訓練を再開して下さい。
    
    4)  invar_importanceval を徐々に 0 から 1 へ増加させます。
        モデルを数 epoch ごとにセーブすることを忘れないで下さい。
        突然 val_reconstruction/outcome_error が悪くなることがありますが、その時は以下のようにして下さい。
        - 訓練を中止して以前のモデルをリロードして下さい。
        - ほんの少し invar_importanceval を下げて訓練を再開して下さい。
        val_reconstruction/outcome_error が常に悪化する用になるまで invar_importanceval を増やして下さい。
    """

    # Train model:
    """
    example: [first]
    train_losses, val_outcome_errors, val_seq2seq_errors = model.train(
        train_seqs=train_data[0], train_outcomes=train_data[1],
        test_seqs=test_data[0], test_outcomes=test_data[1],
        which_model='joint', seq2seq_importanceval=0.95,
        kl_importanceval=0.0, invar_importanceval=0.0,
        max_epochs=50, invar_batchsize=32
    )
    """
    """
    # Revise this single sequence
        
    # model.restore('../output/0908-4th095.bin')
    search_id = 1
    def test(search_id, model):
        init_seq = train_data[0][search_id: search_id + 1]
        init_outcome = train_data[1][search_id: search_id + 1]
        log_alpha = -1000

        revision_results = \
            model.barrierGradientRevise(init_seq, log_alpha=log_alpha,
                                        outcome_opt_learn_rate=1.0,
                                        max_outcome_opt_iter=1000,
                                        use_adaptive=False)
        x_star, z_star, inferred_improvement, \
        outcome_star, reconstruct_init, z_init, \
        outcome_init, avg_sigma_init, edit_dist = revision_results

        if x_star != init_seq[0]:
            print('{} -> {}'.format(init_seq[0], x_star[1]))
            
    def test_sentence(sentence, model):
        log_alpha = -1000
        init_seq = [get_word_yomi_list(sentence)]
        revision_results = \
            model.barrierGradientRevise(init_seq, log_alpha=log_alpha,
                                        outcome_opt_learn_rate=1.0,
                                        max_outcome_opt_iter=1000,
                                        use_adaptive=False)
        x_star, z_star, inferred_improvement, \
        outcome_star, reconstruct_init, z_init, \
        outcome_init, avg_sigma_init, edit_dist = revision_results
        print('{} -> {}'.format(init_seq[0], x_star))
    """
    """
    ex.
    SUCCESS:
    test_sentence('早く寝たい。', model)
    iter=0 obj=[[0.]]
    iter=100 obj=[[0.7482172]]
    iter=200 obj=[[0.71754885]]
    iter=300 obj=[[0.85257566]]
    iter=400 obj=[[1.1158444]]
    iter=500 obj=[[1.121194]]
    iter=600 obj=[[1.3258313]]
    iter=700 obj=[[0.90995586]]
    Elliptical constraint value:[[0.99999994]]
    ['ハヤク', 'ネ', 'タイ', '。'] -> ['ハヤク', 'ネ', 'タ', 'ホウ', 'ガ', 'ヨイ', 'ネ', '。']
    test_sentence('それは良い。', model)
    iter=0 obj=[[0.]]
    iter=100 obj=[[0.31412524]]
    Elliptical constraint value:[[0.9980601]]
    ['ソレ', 'ハ', 'ヨイ', '。'] -> ['ソレ', 'ハ', 'ヨイ', 'ネ', '。']
    test_sentence('応援する。', model)
    iter=0 obj=[[0.]]
    iter=100 obj=[[0.65189254]]
    Elliptical constraint value:[[0.9958178]]
    ['オウエン', 'スル', '。'] -> ['オウエン', 'シ', 'テル', '。']
    test_sentence('鳥肌がたった。', model)
    iter=0 obj=[[0.]]
    iter=100 obj=[[0.37873054]]
    Elliptical constraint value:[[0.99807376]]
    ['トリハダ', 'ガ', 'タッ', 'タ', '。'] -> ['トリハダ', 'ガ', 'タッ', 'タ', 'ネ', '。']

    Question:
    test_sentence('今日は寒かった。', model)
    iter=0 obj=[[0.]]
    Elliptical constraint value:[[0.9977083]]
    ['キョウ', 'ハ', 'サムカッ', 'タ', '。'] -> ['キョウ', 'ハ', 'サムカッ', 'タ', '。']
    test_sentence('今日は寒かった', model)
    iter=0 obj=[[0.]]
    iter=100 obj=[[0.7380588]]
    Elliptical constraint value:[[0.9816849]]
    ['キョウ', 'ハ', 'サムカッ', 'タ'] -> ['キョウ', 'ハ', 'サムカッ', 'タ', 'ネ', '。']
    test_sentence('夕飯は？', model)
    iter=0 obj=[[0.]]
    iter=100 obj=[[0.36785245]]
    Elliptical constraint value:[[0.9974266]]
    ['ユウハン', 'ハ', '？'] -> ['ユウハン', 'ハ', 'ドウ', 'シヨ', 'ウ', 'カ', '？']
    
    Failed:
    test_sentence('何か不安だなぁ。', model)
    iter=0 obj=[[0.]]
    iter=100 obj=[[0.33364055]]
    Elliptical constraint value:[[0.99713546]]
    ['ナニ', 'カ', 'フアン', 'ダ', 'ナァ', '。'] -> ['ナニ', 'カ', 'ノム', '？']
    """