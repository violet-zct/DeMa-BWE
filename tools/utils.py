import os
import io
import re
import sys
import numpy as np
from tools.dictionary import Dictionary
from torch import optim
import math
from torch import nn
import torch


# load Faiss if available (dramatically accelerates the nearest neighbor search)
try:
    import faiss
    FAISS_AVAILABLE = True
    if not hasattr(faiss, 'StandardGpuResources'):
        sys.stderr.write("Impossible to import Faiss-GPU. "
                         "Switching to FAISS-CPU, "
                         "this will be slower.\n\n")

except ImportError:
    sys.stderr.write("Impossible to import Faiss library!! "
                     "Switching to standard nearest neighbors search implementation, "
                     "this will be significantly slower.\n\n")
    FAISS_AVAILABLE = False


def get_batches(n, batch_size):
    tot = math.ceil(n*1.0/batch_size)
    batches = []
    for i in range(tot):
        batches.append((i*batch_size, min((i+1)*batch_size, n)))
    return batches


def get_exp_path(path):
    if os.path.exists(path):
        # tt = strftime("%Y-%m-%d-%H:%M:%S", gmtime())
        # path = path + "-" + tt
        # os.system(f"mkdir -p {path}")
        rm_path = path + "*.bin"
        os.system(f"rm -rf {rm_path}")
    else:
        os.system(f"mkdir -p {path}")
    return path


def select_subset(word_list, max_vocab, freqs, lang="en"):
    """
    Select a subset of words to consider, to deal with words having embeddings
    available in different casings. In particular, we select the embeddings of
    the most frequent words, that are usually of better quality.
    """
    # word2id is consistent with the full embedding with the size of ``max_vocab''
    word2id = {}
    indexes = []
    select_freqs = []
    for i, (word, f) in enumerate(zip(word_list, freqs)):
        word = word.lower()
        if word not in word2id:
            word2id[word] = len(word2id)
            indexes.append(i)
            select_freqs.append(f)

        if max_vocab > 0 and len(word2id) >= max_vocab:
            break
    assert len(word2id) == len(indexes)
    return word2id, indexes, select_freqs


def load_fasttext_model(path):
    """
    Load a binarized fastText model.
    """
    try:
        import fastText
    except ImportError:
        raise Exception("Unable to import fastText. Please install fastText for Python: "
                        "https://github.com/facebookresearch/fastText")
    return fastText.load_model(path)


def cal_empiral_freqs(occurrences, smooth_c):
    # This empirical distribution is calculated based on the subsampling procedure of how skip-gram is trained
    freq = occurrences / sum(occurrences)
    dist = (freq + smooth_c) / (sum(freq + smooth_c))
    return dist


def load_txt_var(args, source: bool, word2id):
    # reload variance file
    path = args.src_var_path if source else args.tgt_var_path

    var = np.ones(len(word2id)) * -1
    with io.open(path, "r", encoding="utf-8") as fin:
        i = 0
        for line in fin:
            if i == 0:
                i += 1
                continue
            tokens = line.strip().split()
            if len(tokens) != 2:
                print("Invalid input line!")
                continue
            word, v = tokens[0], tokens[1]
            if word in word2id:
                var[word2id[word]] = float(v)
    t = sum(var == -1)
    print("Total %d words are not assigned variance!" % t)
    var = var.astype(dtype=np.float32)
    return var


def load_bin_embeddings(args, source: bool):
    """
    Reload pretrained embeddings from a fastText binary file.
    """
    # reload fastText binary file
    lang = args.src_lang if source else args.tgt_lang
    # remove stop words out of these top words
    mf = args.src_train_most_frequent if source else args.tgt_train_most_frequent
    max_vocab = args.max_vocab

    model = load_fasttext_model(args.src_emb_path if source else args.tgt_emb_path)
    words, freqs = model.get_labels(include_freq=True)
    assert model.get_dimension() == args.emb_dim
    print("Loaded binary model. Generating embeddings ...")
    embeddings = np.concatenate([model.get_word_vector(w)[None] for w in words], 0)
    print("Generated embeddings for %i words." % len(words))
    assert embeddings.shape == (len(words), args.emb_dim)

    # select a subset of word embeddings (to deal with casing)
    # stop words might have been removed from freqs and train_indexes
    word2id, indexes, freqs = select_subset(words, max_vocab, freqs, lang=lang)
    # smooth the frequency
    word_dist = cal_empiral_freqs(np.array(freqs), args.smooth_c)
    embeddings = embeddings[indexes]

    id2word = {i: w for w, i in word2id.items()}

    if mf > 0:
        word_dist = word_dist[:mf] / word_dist[:mf].sum()

    dico = Dictionary(id2word, word2id, lang, word_dist)

    assert embeddings.shape == (len(dico), args.emb_dim)
    print(f"Number of words in {lang} = {len(dico)}", len(word_dist))
    print("Max frequency = %.7f, min frequency = %.7f" % (max(word_dist), min(word_dist)))

    return dico, embeddings, word_dist


def read_txt_embeddings(args, source, full_vocab):
    """
    Reload pretrained embeddings from a text file.
    """
    word2id = {}
    vectors = []

    # load pretrained embeddings
    lang = args.src_lang if source else args.tgt_lang
    emb_path = args.src_emb_path if source else args.tgt_emb_path
    _emb_dim_file = args.emb_dim
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):
            if i == 0:
                split = line.split()
                assert len(split) == 2
                assert _emb_dim_file == int(split[1])
            else:
                word, vect = line.rstrip().split(' ', 1)
                if not full_vocab:
                    word = word.lower()
                vect = np.fromstring(vect, sep=' ')
                if np.linalg.norm(vect) == 0:  # avoid to have null embeddings
                    vect[0] = 0.01
                if word in word2id:
                    if full_vocab:
                        print("Word '%s' found twice in %s embedding file"
                                       % (word, 'source' if source else 'target'))
                else:
                    if not vect.shape == (_emb_dim_file,):
                        print("Invalid dimension (%i) for %s word '%s' in line %i."
                                       % (vect.shape[0], 'source' if source else 'target', word, i))
                        continue
                    assert vect.shape == (_emb_dim_file,), i
                    word2id[word] = len(word2id)
                    vectors.append(vect[None])
            if args.max_vocab > 0 and len(word2id) >= args.max_vocab and not full_vocab:
                break

    assert len(word2id) == len(vectors)
    print("Loaded %i pre-trained word embeddings." % len(vectors))

    # compute new vocabulary / embeddings
    id2word = {v: k for k, v in word2id.items()}
    dico = Dictionary(id2word, word2id, lang)
    embeddings = np.concatenate(vectors, 0)
    # embeddings = torch.from_numpy(embeddings).float()

    assert embeddings.shape == (len(dico), args.emb_dim)
    return dico, embeddings


def normalize_embeddings(emb, types, mean=None):
    """
    Normalize embeddings by their norms / recenter them.
    """
    for t in types.split(','):
        if t == '':
            continue
        if t == 'center':
            if mean is None:
                mean = emb.mean(0, keepdim=True)
            emb.sub_(mean.expand_as(emb))
        elif t == 'renorm':
            emb.div_(emb.norm(2, 1, keepdim=True).expand_as(emb))
        elif t == 'double':
            emb.div_(emb.norm(2, 1, keepdim=True).expand_as(emb))
            mean = emb.mean(0, keepdim=True)
            emb.sub_(mean.expand_as(emb))
            emb.div_(emb.norm(2, 1, keepdim=True).expand_as(emb))
        elif t == 'rescale':
            emb.div_(emb.abs().max())
        else:
            raise Exception('Unknown normalization type: "%s"' % t)


def normalize_vec(vec):
    vec = vec / torch.sum(vec)
    return vec


def clip_parameters(model, clip):
    """
    Clip model weights.
    """
    if clip > 0:
        with torch.no_grad():
            for x in model.parameters():
                x.clamp_(-clip, clip)


def bow_idf(sentences, word_vec, idf_dict=None):
    """
    Get sentence representations using weigthed IDF bag-of-words.
    """
    embeddings = []
    for sent in sentences:
        sent = set(sent)
        list_words = [w for w in sent if w in word_vec and w in idf_dict]
        if len(list_words) > 0:
            sentvec = [word_vec[w] * idf_dict[w] for w in list_words]
            sentvec = sentvec / np.sum([idf_dict[w] for w in list_words])
        else:
            sentvec = [word_vec[list(word_vec.keys())[0]]]
        embeddings.append(np.sum(sentvec, axis=0))
    return np.vstack(embeddings)


def get_idf(europarl, src_lg, tgt_lg, n_idf):
    """
    Compute IDF values.
    """
    idf = {src_lg: {}, tgt_lg: {}}
    k = 0
    for lg in idf:
        start_idx = 200000 + k * n_idf
        end_idx = 200000 + (k + 1) * n_idf
        for sent in europarl[lg][start_idx:end_idx]:
            for word in set(sent):
                idf[lg][word] = idf[lg].get(word, 0) + 1
        n_doc = len(europarl[lg][start_idx:end_idx])
        for word in idf[lg]:
            idf[lg][word] = max(1, np.log10(n_doc / (idf[lg][word])))
        k += 1
    return idf


def get_nn_avg_dist(emb, query, knn):
    """
    Compute the average distance of the `knn` nearest neighbors
    for a given set of embeddings and queries.
    Use Faiss if available.
    """
    if FAISS_AVAILABLE:
        emb = emb.cpu().numpy()
        query = query.cpu().numpy()
        if hasattr(faiss, 'StandardGpuResources'):
            # gpu mode
            res = faiss.StandardGpuResources()
            config = faiss.GpuIndexFlatConfig()
            config.device = 0
            index = faiss.GpuIndexFlatIP(res, emb.shape[1], config)
        else:
            # cpu mode
            index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        distances, _ = index.search(query, knn)
        return distances.mean(1)
    else:
        bs = 1024
        all_distances = []
        emb = emb.transpose(0, 1).contiguous()
        for i in range(0, query.shape[0], bs):
            distances = query[i:i + bs].mm(emb)
            best_distances, _ = distances.topk(knn, dim=1, largest=True, sorted=True)
            all_distances.append(best_distances.mean(1).cpu())
        all_distances = torch.cat(all_distances)
        return all_distances.numpy()


def get_nn_avg_dist_mog(emb, query, knn):
    """
    Compute the average distance of the `knn` nearest neighbors
    for a given set of embeddings and queries.
    Use Faiss if available.

    emb has divided sqrt(2) * var
    """
    if FAISS_AVAILABLE:
        emb = emb.cpu().numpy()
        query = query.cpu().numpy()
        if hasattr(faiss, 'StandardGpuResources'):
            # gpu mode
            res = faiss.StandardGpuResources()
            config = faiss.GpuIndexFlatConfig()
            config.device = 0
            index = faiss.GpuIndexFlatL2(res, emb.shape[1], config)
        else:
            # cpu mode
            index = faiss.IndexFlatL2(emb.shape[1])
        index.add(emb)
        # Ad-hoc implementation
        topK = 1000
        temp = 2.
        topK = 10
        distances, idxes = index.search(query, topK)
        return distances.mean(1)
        #query_idx = np.tile(np.arange(query.shape[0]) + 1, (topK, 1)).transpose()
        #rank_diff = abs(np.log(idxes + 1) - np.log(query_idx)) / temp
        #mog_distances_sorted = np.sort(distances + rank_diff)[:, :knn]
        # return: qN, knn
        #return mog_distances_sorted.mean(1)
    else:
        bs = 1024
        all_distances = []
        emb = emb.transpose(0, 1).contiguous()
        for i in range(0, query.shape[0], bs):
            distances = query[i:i + bs].mm(emb)
            best_distances, _ = distances.topk(knn, dim=1, largest=True, sorted=True)
            all_distances.append(best_distances.mean(1).cpu())
        all_distances = torch.cat(all_distances)
        return all_distances.numpy()


def init_linear_layer(args, nn_linear):
    # The commented are normal setting, here we want to make the weights very small to bound the log var
    init_method = args.init_linear
    if init_method == "uniform":
        nn.init.uniform_(nn_linear.weight, -0.1, 0.1)
    elif init_method == "xavier_normal":
        nn.init.xavier_normal_(nn_linear.weight, 0.1)
    else:
        print("No initialization specified!")
        return
    if nn_linear.bias is not None:
        nn_linear.bias.data.zero_()


def get_optimizer(s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    return optim_fn, optim_params
