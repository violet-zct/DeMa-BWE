# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import io
from logging import getLogger
import numpy as np
import torch
import math

from tools.utils import get_nn_avg_dist, get_nn_avg_dist_mog


DIC_EVAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'crosslingual', 'dictionaries')


logger = getLogger()


def load_identical_char_dico(word2id1, word2id2):
    """
    Build a dictionary of identical character strings.
    """
    pairs = [(w1, w1) for w1 in word2id1.keys() if w1 in word2id2]
    if len(pairs) == 0:
        raise Exception("No identical character strings were found. "
                        "Please specify a dictionary.")

    logger.info("Found %i pairs of identical character strings." % len(pairs))

    # sort the dictionary by source word frequencies
    pairs = sorted(pairs, key=lambda x: word2id1[x[0]])
    dico = torch.LongTensor(len(pairs), 2)
    for i, (word1, word2) in enumerate(pairs):
        dico[i, 0] = word2id1[word1]
        dico[i, 1] = word2id2[word2]

    pairs = [(w1, w1) for w1 in word2id2.keys() if w1 in word2id1]
    if len(pairs) == 0:
        raise Exception("No identical character strings were found. "
                        "Please specify a dictionary.")

    logger.info("Found %i pairs of identical character strings for tgt to src." % len(pairs))

    # sort the dictionary by target word frequencies
    rev_pairs = sorted(pairs, key=lambda x: word2id2[x[0]])
    rev_dico = torch.LongTensor(len(rev_pairs), 2)
    for i, (word1, word2) in enumerate(rev_pairs):
        rev_dico[i, 0] = word2id1[word1]
        rev_dico[i, 1] = word2id2[word2]

    return dico, rev_dico


def load_dictionary(path, word2id1, word2id2, reverse=False):
    """
    Return a torch tensor of size (n, 2) where n is the size of the
    loader dictionary, and sort it by source word frequency.
    """
    assert os.path.isfile(path)

    pairs = []
    not_found = 0
    not_found1 = 0
    not_found2 = 0

    with io.open(path, 'r', encoding='utf-8') as f:
        for _, line in enumerate(f):
            assert line == line.lower()
            word1, word2 = line.rstrip().split()

            if word1 in word2id1 and word2 in word2id2:
                pairs.append((word1, word2))
            else:
                not_found += 1
                not_found1 += int(word1 not in word2id1)
                not_found2 += int(word2 not in word2id2)

    logger.info("Found %i pairs of words in the dictionary (%i unique). "
                "%i other pairs contained at least one unknown word "
                "(%i in lang1, %i in lang2)"
                % (len(pairs), len(set([x for x, _ in pairs])),
                   not_found, not_found1, not_found2))

    # sort the dictionary by source word frequencies
    pairs = sorted(pairs, key=lambda x: word2id1[x[0]])
    dico = torch.LongTensor(len(pairs), 2)
    for i, (word1, word2) in enumerate(pairs):
        if reverse == True:
            # 1 is tgt, 0 is src
            dico[i, 1] = word2id1[word1]
            dico[i, 0] = word2id2[word2]
        else:
            dico[i, 0] = word2id1[word1]
            dico[i, 1] = word2id2[word2]
    return dico


def get_word_translation_accuracy(lang1, word2id1, emb1, lang2, word2id2, emb2, method, dico_eval, var=0.01, get_scores=False, valid_dico=None):
    """
    Given source and target word embeddings, and a dictionary,
    evaluate the translation accuracy using the precision@k.

    1 is query, 2 is database
    """
    if dico_eval == 'default':
        path = os.path.join(DIC_EVAL_PATH, '%s-%s.5000-6500.txt' % (lang1, lang2))
    else:
        path = dico_eval

    if valid_dico is not None:
        dico = valid_dico
    else:
        dico = load_dictionary(path, word2id1, word2id2)

    assert dico[:, 0].max() < emb1.size(0)
    assert dico[:, 1].max() < emb2.size(0)

    def _l2_distance(x, y):
        # x: N, d, y: M, d
        # return N, M
        return torch.pow(x, 2).sum(1).unsqueeze(1) - 2 * torch.mm(x, y.t()) + torch.pow(y, 2).sum(1)

    if method == "density":
        pre_norm_emb1 = emb1

    # # normalize word embeddings
    emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
    emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)

    # nearest neighbors
    if method == 'nn':
        query = emb1[dico[:, 0]]
        scores = query.mm(emb2.transpose(0, 1))

    # inverted softmax
    elif method.startswith('invsm_beta_'):
        beta = float(method[len('invsm_beta_'):])
        bs = 128
        word_scores = []
        for i in range(0, emb2.size(0), bs):
            scores = emb1.mm(emb2[i:i + bs].transpose(0, 1))
            scores.mul_(beta).exp_()
            scores.div_(scores.sum(0, keepdim=True).expand_as(scores))
            word_scores.append(scores.index_select(0, dico[:, 0]))
        scores = torch.cat(word_scores, 1)

    # contextual dissimilarity measure
    elif method.startswith('csls_knn_'):
        # average distances to k nearest neighbors
        knn = method[len('csls_knn_'):]
        assert knn.isdigit()
        knn = int(knn)
        average_dist1 = get_nn_avg_dist(emb2, emb1, knn)
        average_dist2 = get_nn_avg_dist(emb1, emb2, knn)
        average_dist1 = torch.from_numpy(average_dist1).type_as(emb1)
        average_dist2 = torch.from_numpy(average_dist2).type_as(emb2)
        # queries / scores
        query = emb1[dico[:, 0]]
        scores = query.mm(emb2.transpose(0, 1))
        scores.mul_(2)
        scores.sub_(average_dist1[dico[:, 0]][:, None])
        scores.sub_(average_dist2[None, :])

    elif method == "density":
        var = var
        # TODO: use pre_norm_emb1 or emb1
        pre_norm_emb1 = pre_norm_emb1 / (math.sqrt(2 * var))

        use_emb1 = pre_norm_emb1
        # use_emb1 = emb1
        emb2 = emb2 / (math.sqrt(2 * var))

        query = use_emb1[dico[:, 0]]
        query_idx = dico[:, 0].unsqueeze(1).float().type_as(emb2)
        db_idx = torch.arange(emb2.size(0)).type_as(emb2).unsqueeze(0)
        
        temp = 2.
        rank_diff = torch.abs(torch.log(query_idx) - torch.log(db_idx)) / temp # q x N
        scores = -_l2_distance(query, emb2) - rank_diff
        #scores = -_l2_distance(query, emb2)

        knn =10
        average_dist1 = get_nn_avg_dist_mog(emb2, use_emb1, knn)
        average_dist2 = get_nn_avg_dist_mog(use_emb1, emb2, knn)
        average_dist1 = torch.from_numpy(average_dist1).type_as(emb1)
        average_dist2 = torch.from_numpy(average_dist2).type_as(emb2)

        scores.mul_(2)
        scores.add_(average_dist1[dico[:, 0]][:, None])
        scores.add_(average_dist2[None, :])
    else:
        raise Exception('Unknown method: "%s"' % method)

    results = []
    top_matches = scores.topk(10, 1, True)[1]

    print("Evaluation on:", path)
    for k in [1, 5, 10]:
        top_k_matches = top_matches[:, :k]
        _matching = (top_k_matches == dico[:, 1][:, None].expand_as(top_k_matches)).sum(1)
        # allow for multiple possible translations
        matching = {}
        for i, src_id in enumerate(dico[:, 0].cpu().numpy()):
            matching[src_id] = min(matching.get(src_id, 0) + _matching[i], 1)
        # evaluate precision@k
        precision_at_k = 100 * np.mean(list(matching.values()))
        print("%i source words - %s - Precision at k = %i: %f" %
                    (len(matching), method, k, precision_at_k))
        results.append(('precision_at_%i' % k, precision_at_k))

    if get_scores:
        return dico, top_matches[:, :10]
    return results


def get_word_translation_accuracy_small_vocab(lang1, word2id1, emb1, lang2, word2id2, emb2, method, dico_eval, s_voc, t_voc, get_scores=False):
    """
    Given source and target word embeddings, and a dictionary,
    evaluate the translation accuracy using the precision@k.
    """
    if dico_eval == 'default':
        path = os.path.join(DIC_EVAL_PATH, '%s-%s.5000-6500.txt' % (lang1, lang2))
    else:
        path = dico_eval
    dico = load_dictionary(path, word2id1, word2id2)

    assert dico[:, 0].max() < emb1.size(0)
    assert dico[:, 1].max() < emb2.size(0)

    count = 0
    oovs = []
    csls_dict = []
    print("Dict size: ", dico.size(0))
    for i in dico[:, 0]:
        if i.item() > s_voc:
            oovs.append(dico[count].unsqueeze(0))
        else:
            csls_dict.append(dico[count].unsqueeze(0))
        count += 1
    csls_dict = torch.cat(csls_dict, dim=0)
    if len(oovs) > 0:
        oov_dict = torch.cat(oovs, dim=0)
    else:
        print("Full dict, csls dict, oov dict: ", count, csls_dict.size(0), len(oovs))

    # # normalize word embeddings
    emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
    emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)
    small_src_emb = emb1[:s_voc]
    small_tgt_emb = emb2[:t_voc]

    assert csls_dict[:, 0].max() < s_voc

    # nearest neighbors
    if method == 'nn':
        query = emb1[dico[:, 0]]
        scores = query.mm(emb2.transpose(0, 1))
        top_matches = scores.topk(10, 1, True)[1]
    # inverted softmax
    elif method.startswith('invsm_beta_'):
        beta = float(method[len('invsm_beta_'):])
        bs = 128
        word_scores = []
        for i in range(0, emb2.size(0), bs):
            scores = emb1.mm(emb2[i:i + bs].transpose(0, 1))
            scores.mul_(beta).exp_()
            scores.div_(scores.sum(0, keepdim=True).expand_as(scores))
            word_scores.append(scores.index_select(0, dico[:, 0]))
        scores = torch.cat(word_scores, 1)
        top_matches = scores.topk(10, 1, True)[1]
    # contextual dissimilarity measure
    elif method.startswith('csls_knn_'):
        # average distances to k nearest neighbors
        knn = method[len('csls_knn_'):]
        assert knn.isdigit()
        knn = int(knn)
        average_dist1 = get_nn_avg_dist(small_tgt_emb, small_src_emb, knn)
        average_dist2 = get_nn_avg_dist(small_src_emb, small_tgt_emb, knn)
        average_dist1 = torch.from_numpy(average_dist1).type_as(emb1)
        average_dist2 = torch.from_numpy(average_dist2).type_as(emb2)
        # queries / scores
        # csls_dict contain pairs that the src word must be in the small vocab
        query = small_src_emb[csls_dict[:, 0]]
        scores = query.mm(small_tgt_emb.transpose(0, 1))
        scores.mul_(2)
        scores.sub_(average_dist1[csls_dict[:, 0]][:, None])
        scores.sub_(average_dist2[None, :])

        if len(oovs) > 0:
            oov_query = emb1[oov_dict[: 0]]
            oov_scores = oov_query.mm(emb2)
            dico = torch.cat([csls_dict, oov_dict], dim=0)

            top_matches = torch.cat([scores.topk(10, 1, True)[1], oov_scores.topk(10, 1, True)[1]], dim=0)
        else:
            top_matches = scores.topk(10, 1, True)[1]
            dico = csls_dict
    else:
        raise Exception('Unknown method: "%s"' % method)

    results = []

    print("Evaluation on:", path)
    for k in [1, 5, 10]:
        top_k_matches = top_matches[:, :k]
        _matching = (top_k_matches == dico[:, 1][:, None].expand_as(top_k_matches)).sum(1)
        # allow for multiple possible translations
        matching = {}
        for i, src_id in enumerate(dico[:, 0].cpu().numpy()):
            matching[src_id] = min(matching.get(src_id, 0) + _matching[i], 1)
        # evaluate precision@k
        precision_at_k = 100 * np.mean(list(matching.values()))
        print("%i source words - %s - Precision at k = %i: %f" %
                    (len(matching), method, k, precision_at_k))
        results.append(('precision_at_%i' % k, precision_at_k))

    if get_scores:
        return dico, top_matches[:, :10]

    return results
