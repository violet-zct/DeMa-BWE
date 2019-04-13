# Copyright (c) 2017-present, Facebook, Inc, 2019-present, Chunting Zhou.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from copy import deepcopy
import numpy as np
from torch.autograd import Variable
from torch import Tensor as torch_tensor

from . import get_wordsim_scores, get_crosslingual_wordsim_scores, get_wordanalogy_scores
from . import get_word_translation_accuracy, get_word_translation_accuracy_small_vocab
from . import load_europarl_data, get_sent_translation_accuracy
from tools.dico_builder import get_candidates, build_dictionary
from tools.utils import get_idf
import torch
import os

logger = getLogger()


class Evaluator(object):

    def __init__(self, model, src_emb, tgt_emb):
        """
        Initialize evaluator.
        """
        self.pre_src_emb = src_emb
        self.pre_tgt_emb = tgt_emb
        self.model = model

        self.src_dict = model.src_dict
        self.tgt_dict = model.tgt_dict
        self.src_flow = model.src_flow
        self.tgt_flow = model.tgt_flow
        self.args = model.args

    def monolingual_wordsim(self, to_log):
        """
        Evaluation on monolingual word similarity.
        """
        src_ws_scores = get_wordsim_scores(
            self.src_dict.lang, self.src_dict.word2id,
            self.src_emb.numpy()
        )
        tgt_ws_scores = get_wordsim_scores(
            self.tgt_dict.lang, self.tgt_dict.word2id,
            self.tgt_emb.numpy()
        )
        if src_ws_scores is not None:
            src_ws_monolingual_scores = np.mean(list(src_ws_scores.values()))
            print("Monolingual source word similarity score average: %.5f" % src_ws_monolingual_scores)
            to_log['src_ws_monolingual_scores'] = src_ws_monolingual_scores
            to_log.update({'src_' + k: v for k, v in src_ws_scores.items()})
        if tgt_ws_scores is not None:
            tgt_ws_monolingual_scores = np.mean(list(tgt_ws_scores.values()))
            print("Monolingual target word similarity score average: %.5f" % tgt_ws_monolingual_scores)
            to_log['tgt_ws_monolingual_scores'] = tgt_ws_monolingual_scores
            to_log.update({'tgt_' + k: v for k, v in tgt_ws_scores.items()})
        if src_ws_scores is not None and tgt_ws_scores is not None:
            ws_monolingual_scores = (src_ws_monolingual_scores + tgt_ws_monolingual_scores) / 2
            print("Monolingual word similarity score average: %.5f" % ws_monolingual_scores)
            to_log['ws_monolingual_scores'] = ws_monolingual_scores

    def monolingual_wordanalogy(self, to_log):
        """
        Evaluation on monolingual word analogy.
        """
        src_analogy_scores = get_wordanalogy_scores(
            self.src_dict.lang, self.src_dict.word2id,
            self.src_emb.numpy()
        )
        tgt_analogy_scores = get_wordanalogy_scores(
            self.tgt_dict.lang, self.tgt_dict.word2id,
            self.tgt_emb.numpy())

        if src_analogy_scores is not None:
            src_analogy_monolingual_scores = np.mean(list(src_analogy_scores.values()))
            print("Monolingual source word analogy score average: %.5f" % src_analogy_monolingual_scores)
            to_log['src_analogy_monolingual_scores'] = src_analogy_monolingual_scores
            to_log.update({'src_' + k: v for k, v in src_analogy_scores.items()})
        if tgt_analogy_scores is not None:
            tgt_analogy_monolingual_scores = np.mean(list(tgt_analogy_scores.values()))
            print("Monolingual target word analogy score average: %.5f" % tgt_analogy_monolingual_scores)
            to_log['tgt_analogy_monolingual_scores'] = tgt_analogy_monolingual_scores
            to_log.update({'tgt_' + k: v for k, v in tgt_analogy_scores.items()})

    def crosslingual_wordsim(self, to_log, s2t=True):
        """
        Evaluation on cross-lingual word similarity.
        """
        src_emb = self.src_emb.numpy()
        tgt_emb = self.tgt_emb.numpy()
        # cross-lingual wordsim evaluation
        if s2t:
            src_tgt_ws_scores = get_crosslingual_wordsim_scores(
                self.src_dict.lang, self.src_dict.word2id, src_emb,
                self.tgt_dict.lang, self.tgt_dict.word2id, tgt_emb,
            )
            prefix = self.src_dict.lang + "-" + self.tgt_dict.lang
        else:
            src_tgt_ws_scores = get_crosslingual_wordsim_scores(
                self.tgt_dict.lang, self.tgt_dict.word2id, tgt_emb,
                self.src_dict.lang, self.src_dict.word2id, src_emb,
            )
            prefix = self.tgt_dict.lang + "-" + self.src_dict.lang

        if src_tgt_ws_scores is None:
            return
        ws_crosslingual_scores = np.mean(list(src_tgt_ws_scores.values()))
        print(prefix + " --- Cross-lingual word similarity score average: %.5f" % ws_crosslingual_scores)
        to_log[prefix + '_ws_crosslingual_scores'] = ws_crosslingual_scores
        to_log.update({prefix + "_" + k: v for k, v in src_tgt_ws_scores.items()})

    def word_translation(self, to_log):
        """
        Evaluation on word translation.
        """
        # mapped word embeddings
        src_emb = self.src_emb
        tgt_emb = self.tgt_emb

        for eval_path in [self.args.dico_eval]:#, self.args.sup_dict_path]:
            for method in ['nn', 'csls_knn_10']:
                results = get_word_translation_accuracy(
                    self.src_dict.lang, self.src_dict.word2id, src_emb,
                    self.tgt_dict.lang, self.tgt_dict.word2id, tgt_emb,
                    method=method,
                    dico_eval=eval_path
                    # dico_eval=self.args.dico_eval
                )
                to_log.update([('%s-%s' % (k, method), v) for k, v in results])

    def sent_translation(self, to_log):
        """
        Evaluation on sentence translation.
        Only available on Europarl, for en - {de, es, fr, it} language pairs.
        """
        lg1 = self.src_dict.lang
        lg2 = self.tgt_dict.lang

        # parameters
        n_keys = 200000
        n_queries = 2000
        n_idf = 300000

        # load europarl data
        if not hasattr(self, 'europarl_data'):
            self.europarl_data = load_europarl_data(
                lg1, lg2, n_max=(n_keys + 2 * n_idf)
            )

        # if no Europarl data for this language pair
        if not self.europarl_data:
            return

        # mapped word embeddings
        src_emb = self.src_emb
        tgt_emb = self.tgt_emb

        # get idf weights
        idf = get_idf(self.europarl_data, lg1, lg2, n_idf=n_idf)

        for method in ['nn', 'csls_knn_10']:

            # source <- target sentence translation
            results = get_sent_translation_accuracy(
                self.europarl_data,
                self.src_dict.lang, self.src_dict.word2id, src_emb,
                self.tgt_dict.lang, self.tgt_dict.word2id, tgt_emb,
                n_keys=n_keys, n_queries=n_queries,
                method=method, idf=idf
            )
            to_log.update([('tgt_to_src_%s-%s' % (k, method), v) for k, v in results])

            # target <- source sentence translation
            results = get_sent_translation_accuracy(
                self.europarl_data,
                self.tgt_dict.lang, self.tgt_dict.word2id, tgt_emb,
                self.src_dict.lang, self.src_dict.word2id, src_emb,
                n_keys=n_keys, n_queries=n_queries,
                method=method, idf=idf
            )
            to_log.update([('src_to_tgt_%s-%s' % (k, method), v) for k, v in results])

    def dist_mean_cosine(self, to_log, s2t=True):
        """
        Mean-cosine model selection criterion.
        """
        if s2t:
            prefix = self.src_dict.lang + "-" + self.tgt_dict.lang
        else:
            prefix = self.tgt_dict.lang + "-" + self.src_dict.lang

        # build dictionary
        for dico_method in ['csls_knn_10']:
            dico_build = 'S2T'
            dico_max_size = 10000
            # temp params / dictionary generation
            _params = deepcopy(self.args)
            _params.dico_method = dico_method
            _params.dico_build = dico_build
            _params.dico_threshold = 0
            _params.dico_max_rank = 10000
            _params.dico_min_size = 0
            _params.dico_max_size = dico_max_size
            s2t_candidates = get_candidates(self.src_emb, self.tgt_emb, _params)
            t2s_candidates = get_candidates(self.tgt_emb, self.src_emb, _params)

            if s2t:
                dico = build_dictionary(self.src_emb, self.tgt_emb, _params, s2t_candidates, t2s_candidates)
            else:
                dico = build_dictionary(self.tgt_emb, self.src_emb, _params, t2s_candidates, s2t_candidates)

            # mean cosine
            if dico is None:
                mean_cosine = -1e9
            else:
                if s2t:
                    mean_cosine = (self.src_emb[dico[:dico_max_size, 0]] * self.tgt_emb[dico[:dico_max_size, 1]]).sum(1).mean()
                else:
                    mean_cosine = (self.src_emb[dico[:dico_max_size, 1]] * self.tgt_emb[dico[:dico_max_size, 0]]).sum(1).mean()

            mean_cosine = mean_cosine.item() if isinstance(mean_cosine, torch_tensor) else mean_cosine
            print("%s: Mean cosine (%s method, %s build, %i max size): %.5f"
                        % (prefix, dico_method, _params.dico_build, dico_max_size, mean_cosine))
            to_log['%s-mean_cosine-%s-%s-%i' % (prefix, dico_method, _params.dico_build, dico_max_size)] = mean_cosine

    def word_translation_bidirect(self, to_log, eval_path, src_to_tgt=True, valid_dico=None):
        """
        Evaluation on word translation.
        """
        # mapped word embeddings
        src_emb = self.src_emb
        tgt_emb = self.tgt_emb

        if valid_dico is not None:
            prefix = "valid-"
        else:
            prefix = ""

        for eval_path in [eval_path]:#, self.args.sup_dict_path]:
            for method in ['csls_knn_10', 'density']:
                if src_to_tgt:
                    results = get_word_translation_accuracy(
                        self.src_dict.lang, self.src_dict.word2id, src_emb,  # query
                        self.tgt_dict.lang, self.tgt_dict.word2id, tgt_emb,
                        method=method,
                        dico_eval=eval_path,
                        var = self.args.s2t_t_var,
                        valid_dico=valid_dico
                    )
                    to_log.update(
                        [(prefix + self.args.src_lang + "-" + self.args.tgt_lang + '-%s-%s' % (k, method), v) for k, v in
                         results])
                else:

                    results = get_word_translation_accuracy(
                        self.tgt_dict.lang, self.tgt_dict.word2id, tgt_emb,  # query
                        self.src_dict.lang, self.src_dict.word2id, src_emb,
                        method=method,
                        dico_eval=eval_path,
                        var = self.args.t2s_s_var,
                        valid_dico=valid_dico
                    )
                    to_log.update(
                        [(prefix + self.args.tgt_lang + "-" + self.args.src_lang + '-%s-%s' % (k, method), v) for k, v in
                         results])

    def all_eval(self, to_log, train=False, s2t=True, t2s=True, unsup_eval=False):
        """
        Run all evaluations.
        """
        src_to_tgt_emb, tgt_to_src_emb = self.model.map_embs(self.pre_src_emb, self.pre_tgt_emb, s2t, t2s)

        if s2t:
            print("<%s> TO <%s> Evaluation: " % (self.src_dict.lang, self.tgt_dict.lang))
            self.src_emb = src_to_tgt_emb
            self.tgt_emb = self.pre_tgt_emb.cpu()
            # self.monolingual_wordsim(to_log)
            # self.crosslingual_wordsim(to_log, True)
            self.word_translation_bidirect(to_log, self.args.dico_eval, src_to_tgt=True)
            if train and unsup_eval:
                self.dist_mean_cosine(to_log, True)
            if train and self.model.s2t_valid_dico is not None:
                print("-" * 20 + "Validation" + "-" * 20)
                self.word_translation_bidirect(to_log, self.args.dico_eval, src_to_tgt=True, valid_dico=self.model.s2t_valid_dico)

        if t2s:
            print("<%s> TO <%s> Evaluation: " % (self.tgt_dict.lang, self.src_dict.lang))
            self.src_emb = self.pre_src_emb.cpu()
            self.tgt_emb = tgt_to_src_emb
            # self.monolingual_wordsim(to_log)
            # self.crosslingual_wordsim(to_log, False)
            tgt_to_src_path = os.path.join(os.path.dirname(self.args.dico_eval),
                                           self.args.tgt_lang + "-" + self.args.src_lang + ".5000-6500.txt")
            self.word_translation_bidirect(to_log, tgt_to_src_path, src_to_tgt=False)
            if train and unsup_eval:
                self.dist_mean_cosine(to_log, False)
            if train and self.model.t2s_valid_dico is not None:
                print("-" * 20 + "Validation" + "-" * 20)
                self.word_translation_bidirect(to_log, tgt_to_src_path, src_to_tgt=False, valid_dico=self.model.t2s_valid_dico)
