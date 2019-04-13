import sys
sys.path.append("../")
import argparse
from tools.dictionary import Dictionary
from tools.utils import normalize_embeddings
import torch
import os
from tools.lazy_reader import *
import math
import io


def select_subset(word_list, max_vocab, freqs):
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


def load_bin_embeddings(args, source: bool):
    """
    Reload pretrained embeddings from a fastText binary file.
    """
    # load fastText binary file
    lang = args.src_lang if source else args.tgt_lang
    # remove stop words out of these top words
    max_vocab = args.vocab_size

    model = load_fasttext_model(args.src_emb_path if source else args.tgt_emb_path)
    words, freqs = model.get_labels(include_freq=True)
    assert model.get_dimension() == args.emb_dim
    print("Loaded binary model. Generating embeddings ...")
    embeddings = np.concatenate([model.get_word_vector(w)[None] for w in words], 0)
    print("Generated embeddings for %i words." % len(words))
    assert embeddings.shape == (len(words), args.emb_dim)

    # select a subset of word embeddings (to deal with casing)
    # stop words might have been removed from freqs and train_indexes
    word2id, indexes, freqs = select_subset(words, max_vocab, freqs)
    embeddings = embeddings[indexes]

    id2word = {i: w for w, i in word2id.items()}
    dico = Dictionary(id2word, word2id, lang)

    assert embeddings.shape == (len(dico), args.emb_dim)
    print(f"Number of words in {lang} = {len(dico)}")

    return embeddings, dico

parser = argparse.ArgumentParser()
parser.add_argument("--src_lang", type=str)
parser.add_argument("--tgt_lang", type=str)
parser.add_argument("--src_emb_path", type=str)
parser.add_argument("--tgt_emb_path", type=str)
parser.add_argument("--s2t_map_path", type=str, default=None)
parser.add_argument("--t2s_map_path", type=str, default=None)
parser.add_argument("--emb_dim", type=int, default=300)
parser.add_argument("--vocab_size", type=int, default=50000, help="number of most frequent embeddings to map")
parser.add_argument("--normalize_embeddings", type=str, default="double", choices=['',  'double', 'renorm', 'center', 'rescale'])

args = parser.parse_args()

save_path = "../eval/"
if not os.path.exists(save_path):
    os.system("mkdir -p %s" % save_path)

assert args.src_emb_path is not None and args.tgt_emb_path is not None
assert args.tgt_emb_path.endswith("bin") and args.src_emb_path.endswith("bin")

args.cuda = torch.cuda.is_available()
device = torch.device('cuda') if args.cuda else torch.device("cpu")

np_src_emb, src_dict = load_bin_embeddings(args, True)
np_tgt_emb, tgt_dict = load_bin_embeddings(args, False)

gb_size = 1073741824
print("Size of the src and tgt embedding in Gigabytes: %f, %f" %
      ((np_src_emb.size * np_src_emb.itemsize / gb_size, np_tgt_emb.size * np_tgt_emb.itemsize / gb_size)))

# prepare embeddings
src_emb = torch.from_numpy(np_src_emb).float().to(device)
tgt_emb = torch.from_numpy(np_tgt_emb).float().to(device)

normalize_embeddings(src_emb, args.normalize_embeddings)
normalize_embeddings(tgt_emb, args.normalize_embeddings)

W_s2t = torch.from_numpy(torch.load(args.s2t_map_path)).to(device)
W_t2s = torch.from_numpy(torch.load(args.t2s_map_path)).to(device)

s2t = t2s = False
if args.s2t_map_path is not None:
    W_s2t = torch.from_numpy(torch.load(args.s2t_map_path)).to(device)
    s2t = True
if args.t2s_map_path is not None:
    W_t2s = torch.from_numpy(torch.load(args.t2s_map_path)).to(device)
    t2s = True
if not s2t and not t2s:
    exit(0)


def get_batches(n, batch_size):
    tot = math.ceil(n * 1.0 / batch_size)
    batches = []
    for i in range(tot):
        batches.append((i * batch_size, min((i + 1) * batch_size, n)))
    return batches


def map_embs(src_emb, tgt_emb, s2t=True, t2s=True):
    src2tgt_emb = tgt2src_emb = None

    if s2t:
        src_to_tgt_list = []
        for i, j in get_batches(src_emb.size(0), 512):
            src_emb_batch = src_emb[i:j, :]
            src_to_tgt = src_emb_batch.mm(W_s2t)
            src_to_tgt_list.append(src_to_tgt.cpu())
        src2tgt_emb = torch.cat(src_to_tgt_list, dim=0)
    if t2s:
        tgt_to_src_list = []
        for i, j in get_batches(tgt_emb.size(0), 512):
            tgt_emb_batch = tgt_emb[i:j, :]
            tgt_to_src = tgt_emb_batch.mm(W_t2s)
            tgt_to_src_list.append(tgt_to_src.cpu())
        tgt2src_emb = torch.cat(tgt_to_src_list, dim=0)

    return src2tgt_emb, tgt2src_emb


def export_embeddings(src_emb, tgt_emb, exp_path, src_dict, tgt_dict, s2t=True, t2s=True):
    mapped_src_emb, mapped_tgt_emb = map_embs(src_emb, tgt_emb, s2t=s2t, t2s=t2s)
    src_path = exp_path + src_dict.lang + "2" + tgt_dict.lang + ".vec"
    tgt_path = exp_path + tgt_dict.lang + "2" + src_dict.lang + ".vec"
    pre_src_path = exp_path + src_dict.lang + ".vec"
    pre_tgt_path = exp_path + tgt_dict.lang + ".vec"

    if s2t:
        mapped_src_emb = mapped_src_emb.numpy()
        print(f'Writing mapped source to target embeddings to {src_path}')
        with io.open(src_path, 'w', encoding='utf-8') as f:
            f.write(u"%i %i\n" % mapped_src_emb.shape)
            for i in range(len(src_dict)):
                f.write(u"%s %s\n" % (src_dict[i], " ".join('%.5f' % x for x in mapped_src_emb[i])))
        print(f'Writing corresponding target embeddings to {pre_tgt_path}')
        with io.open(pre_tgt_path, 'w', encoding='utf-8') as f:
            f.write(u"%i %i\n" % tgt_emb.shape)
            for i in range(len(tgt_dict)):
                f.write(u"%s %s\n" % (tgt_dict[i], " ".join('%.5f' % x for x in tgt_emb[i])))

    if t2s:
        mapped_tgt_emb = mapped_tgt_emb.numpy()
        print(f'Writing mapped target to source embeddings to {tgt_path}')
        with io.open(tgt_path, 'w', encoding='utf-8') as f:
            f.write(u"%i %i\n" % mapped_tgt_emb.shape)
            for i in range(len(tgt_dict)):
                f.write(u"%s %s\n" % (tgt_dict[i], " ".join('%.5f' % x for x in mapped_tgt_emb[i])))
        print(f'Writing corresponding source embeddings to {pre_src_path}')
        with io.open(pre_src_path, 'w', encoding='utf-8') as f:
            f.write(u"%i %i\n" % src_emb.shape)
            for i in range(len(src_dict)):
                f.write(u"%s %s\n" % (src_dict[i], " ".join('%.5f' % x for x in src_emb[i])))

export_embeddings(src_emb, tgt_emb, save_path, src_dict, tgt_dict, s2t, t2s)