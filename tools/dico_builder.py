# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import torch

from .utils import get_nn_avg_dist


logger = getLogger()


def get_candidates(emb1, emb2, args):
    """
    Get best translation pairs candidates.
    """
    # emb1 and emb2 are torch.tensor on cpus
    bs = 128

    all_scores = []
    all_targets = []

    # number of source words to consider
    n_src = emb1.size(0)
    if args.dico_max_rank > 0 and not args.dico_method.startswith('invsm_beta_'):
        n_src = args.dico_max_rank

    # nearest neighbors
    if args.dico_method == 'nn':

        # for every source word
        for i in range(0, n_src, bs):

            # compute target words scores
            # src_bs, tgt_all
            scores = emb2.mm(emb1[i:min(n_src, i + bs)].transpose(0, 1)).transpose(0, 1)
            best_scores, best_targets = scores.topk(2, dim=1, largest=True, sorted=True)

            # update scores / potential targets
            all_scores.append(best_scores.cpu())
            all_targets.append(best_targets.cpu())

        all_scores = torch.cat(all_scores, 0)
        all_targets = torch.cat(all_targets, 0)

    # inverted softmax
    elif args.dico_method.startswith('invsm_beta_'):

        beta = float(args.dico_method[len('invsm_beta_'):])

        # for every target word
        for i in range(0, emb2.size(0), bs):

            # compute source words scores
            scores = emb1.mm(emb2[i:i + bs].transpose(0, 1))
            scores.mul_(beta).exp_()
            scores.div_(scores.sum(0, keepdim=True).expand_as(scores))

            best_scores, best_targets = scores.topk(2, dim=1, largest=True, sorted=True)

            # update scores / potential targets
            all_scores.append(best_scores.cpu())
            all_targets.append((best_targets + i).cpu())

        all_scores = torch.cat(all_scores, 1)
        all_targets = torch.cat(all_targets, 1)

        all_scores, best_targets = all_scores.topk(2, dim=1, largest=True, sorted=True)
        all_targets = all_targets.gather(1, best_targets)

    # contextual dissimilarity measure
    elif args.dico_method.startswith('csls_knn_'):

        knn = args.dico_method[len('csls_knn_'):]
        assert knn.isdigit()
        knn = int(knn)

        # average distances to k nearest neighbors
        average_dist1 = torch.from_numpy(get_nn_avg_dist(emb2, emb1, knn))
        average_dist2 = torch.from_numpy(get_nn_avg_dist(emb1, emb2, knn))
        average_dist1 = average_dist1.type_as(emb1)
        average_dist2 = average_dist2.type_as(emb2)

        # for every source word
        for i in range(0, n_src, bs):

            # compute target words scores
            scores = emb2.mm(emb1[i:min(n_src, i + bs)].transpose(0, 1)).transpose(0, 1)
            scores.mul_(2)
            scores.sub_(average_dist1[i:min(n_src, i + bs)][:, None] + average_dist2[None, :])
            best_scores, best_targets = scores.topk(2, dim=1, largest=True, sorted=True)

            # update scores / potential targets
            all_scores.append(best_scores.cpu())
            all_targets.append(best_targets.cpu())

        all_scores = torch.cat(all_scores, 0)
        all_targets = torch.cat(all_targets, 0)

    all_pairs = torch.cat([
        torch.arange(0, all_targets.size(0)).long().unsqueeze(1),
        all_targets[:, 0].unsqueeze(1)
    ], 1)

    # sanity check
    assert all_scores.size() == all_pairs.size() == (n_src, 2)

    # sort pairs by score confidence
    diff = all_scores[:, 0] - all_scores[:, 1]
    reordered = diff.sort(0, descending=True)[1]
    all_scores = all_scores[reordered]
    all_pairs = all_pairs[reordered]

    # max dico words rank
    if args.dico_max_rank > 0:
        selected = all_pairs.max(1)[0] <= args.dico_max_rank
        mask = selected.unsqueeze(1).expand_as(all_scores).clone()
        all_scores = all_scores.masked_select(mask).view(-1, 2)
        all_pairs = all_pairs.masked_select(mask).view(-1, 2)

    # max dico size
    if args.dico_max_size > 0:
        all_scores = all_scores[:args.dico_max_size]
        all_pairs = all_pairs[:args.dico_max_size]

    # min dico size
    diff = all_scores[:, 0] - all_scores[:, 1]
    if args.dico_min_size > 0:
        diff[:args.dico_min_size] = 1e9
    
    # print("min diff: ", diff.min().item())
    # confidence threshold
    if args.dico_threshold > 0:
        mask = diff > args.dico_threshold
        print("Selected %i / %i pairs above the confidence threshold." % (mask.sum(), diff.size(0)))
        mask = mask.unsqueeze(1).expand_as(all_pairs).clone()
        all_pairs = all_pairs.masked_select(mask).view(-1, 2)

    return all_pairs


def build_dictionary(src_emb, tgt_emb, args, s2t_candidates=None, t2s_candidates=None):
    """
    Build a training dictionary given current embeddings / mapping.
    """
    print("Building the train dictionary ...")
    s2t = 'S2T' in args.dico_build
    t2s = 'T2S' in args.dico_build
    assert s2t or t2s

    if s2t:
        if s2t_candidates is None:
            s2t_candidates = get_candidates(src_emb, tgt_emb, args)
    if t2s:
        if t2s_candidates is None:
            t2s_candidates = get_candidates(tgt_emb, src_emb, args)
        t2s_candidates = torch.cat([t2s_candidates[:, 1:], t2s_candidates[:, :1]], 1)

    if args.dico_build == 'S2T':
        dico = s2t_candidates
    elif args.dico_build == 'T2S':
        dico = t2s_candidates
    else:
        s2t_candidates = set([(a, b) for a, b in s2t_candidates.numpy()])
        t2s_candidates = set([(a, b) for a, b in t2s_candidates.numpy()])
        if args.dico_build == 'S2T|T2S':
            final_pairs = s2t_candidates | t2s_candidates
        else:
            assert args.dico_build == 'S2T&T2S'
            final_pairs = s2t_candidates & t2s_candidates
            if len(final_pairs) == 0:
                print("Empty intersection ...")
                return None
        dico = torch.LongTensor(list([[int(a), int(b)] for (a, b) in final_pairs]))

    print('New train dictionary of %i pairs.' % dico.size(0))
    return dico
