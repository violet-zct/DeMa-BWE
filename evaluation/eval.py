import sys
sys.path.append("../")
import argparse
from tools.args import *
from tools.utils import *
import torch
import random
from modules.emb2emb import E2E
from collections import OrderedDict
from evaluation import Evaluator

from tools.lazy_reader import *


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


def reload_bin_embeddings(args, source: bool, vocab_size):
    """
    Reload pretrained embeddings from a fastText binary file.
    """
    # reload fastText binary file
    lang = args.src_lang if source else args.tgt_lang
    max_vocab = vocab_size

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


parser = argparse.ArgumentParser(description="embedding to noise")
add_e2e_args(parser)
args = parser.parse_args()

print(args)
print("Path s2t: ", args.load_from_pretrain_s2t)
print("Path t2s: ", args.load_from_pretrain_s2t)

save_path = "../eval/"
if not os.path.exists(save_path):
    os.system("mkdir -p %s" % save_path)

assert args.src_emb_path is not None and args.tgt_emb_path is not None
args.src_lang = args.src_emb_path.split("/")[-1].split(".")[1]
args.tgt_lang = args.tgt_emb_path.split("/")[-1].split(".")[1]

VALIDATION_METRIC_SUP_s2t = args.src_lang + "-" + args.tgt_lang + '-precision_at_1-csls_knn_10'
VALIDATION_METRIC_SUP_t2s = args.tgt_lang + "-" + args.src_lang + '-precision_at_1-csls_knn_10'
DENSITY_METRIC_SUP_s2t = args.src_lang + "-" + args.tgt_lang + '-precision_at_1-density'
DENSITY_METRIC_SUP_t2s = args.tgt_lang + "-" + args.src_lang + '-precision_at_1-density'

best_valid_t2s_metric = 1e-12
best_valid_s2t_metric = 1e-12

best_valid_density_s2t = best_valid_density_t2s = 1e-12

args.cuda = torch.cuda.is_available()
random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
if args.cuda:
    torch.cuda.manual_seed(args.random_seed)
device = torch.device('cuda') if args.cuda else torch.device("cpu")

assert args.tgt_emb_path.endswith("bin")

src_dict, np_src_emb, np_src_freqs = load_bin_embeddings(args, True)
tgt_dict, np_tgt_emb, np_tgt_freqs = load_bin_embeddings(args, False)

gb_size = 1073741824
print("Size of the src and tgt embedding in Gigabytes: %f, %f" %
      ((np_src_emb.size * np_src_emb.itemsize / gb_size, np_tgt_emb.size * np_tgt_emb.itemsize / gb_size)))

# prepare embeddings
src_emb = torch.from_numpy(np_src_emb).float().to(device)
tgt_emb = torch.from_numpy(np_tgt_emb).float().to(device)

normalize_embeddings(src_emb, args.normalize_embeddings)
normalize_embeddings(tgt_emb, args.normalize_embeddings)

# prepare model and evaluator
src_emb_for_mog = src_emb[:10000]
tgt_emb_for_mog = tgt_emb[:10000]

src_dev = torch.std(src_emb_for_mog, dim=0)
tgt_dev = torch.std(tgt_emb_for_mog, dim=0)

print(f"{src_dict.lang}: max std={torch.max(src_dev).item()}, min std={torch.min(src_dev).item()}, mean std={torch.mean(src_dev).item()}")
print(f"{tgt_dict.lang}: max std={torch.max(tgt_dev).item()}, min std={torch.min(tgt_dev).item()}, mean std={torch.mean(tgt_dev).item()}")

s2t = t2s = False
if args.load_from_pretrain_s2t is not None:
    s2t = True
if args.load_from_pretrain_t2s is not None:
    t2s = True
if not s2t and not t2s:
    exit(0)

model = E2E(args, src_dict, tgt_dict, src_emb_for_mog, tgt_emb_for_mog, device).to(device)

def export_embeddings(mm, src_emb, tgt_emb, exp_path, src_dict, tgt_dict, s2t=True, t2s=True):
    if s2t:
        mm.load_best_s2t()
    if t2s:
        mm.load_best_t2s()

    mapped_src_emb, mapped_tgt_emb = mm.map_embs(src_emb, tgt_emb, s2t=s2t, t2s=t2s)
    src_path = exp_path + src_dict.lang + "2" + tgt_dict.lang + "_emb.vec"
    tgt_path = exp_path + tgt_dict.lang + "2" + src_dict.lang + "_emb.vec"
    pre_src_path = exp_path + src_dict.lang + ".vec"
    pre_tgt_path = exp_path + tgt_dict.lang + ".vec"

    if s2t:
        mapped_src_emb = mapped_src_emb.cpu().numpy()
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
        mapped_tgt_emb = mapped_tgt_emb.cpu().numpy()
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

save_s2t = False
save_t2s = False

args.dico_method = "csls_knn_10"
evaluator = Evaluator(model, src_emb, tgt_emb)
to_log = OrderedDict()
print("--------------------------------- Before refinement ---------------------------------" )
evaluator.all_eval(to_log, s2t=s2t, t2s=t2s)

if args.n_refinement > 0:
    print("--------------------------------- Starting Procrustes Refinement ---------------------------------")
    for n_iter in range(args.n_refinement):
        print("Refinement iteration %d" % (n_iter+1))

        model.build_dictionary(src_emb, tgt_emb, s2t=s2t, t2s=t2s)
        model.procrustes(src_emb, tgt_emb, s2t=s2t, t2s=t2s)

        to_log["iters"] = n_iter
        evaluator.all_eval(to_log, s2t=s2t, t2s=t2s)

        if s2t and to_log[VALIDATION_METRIC_SUP_s2t] > best_valid_s2t_metric:
            model.set_save_s2t_path(save_path + "best_" + args.src_lang + "2" + args.tgt_lang + "_params.bin")
            model.save_best_s2t()
            best_valid_s2t_metric = to_log[VALIDATION_METRIC_SUP_s2t]
            save_s2t = True
        if t2s and to_log[VALIDATION_METRIC_SUP_t2s] > best_valid_t2s_metric:
            model.set_save_t2s_path(save_path + "best_" + args.tgt_lang + "2" + args.src_lang + "_params.bin")
            model.save_best_t2s()
            best_valid_t2s_metric = to_log[VALIDATION_METRIC_SUP_t2s]
            save_t2s = True

        if s2t and to_log[DENSITY_METRIC_SUP_s2t] > best_valid_density_s2t:
            best_valid_density_s2t = to_log[DENSITY_METRIC_SUP_s2t]
        if t2s and to_log[DENSITY_METRIC_SUP_t2s] > best_valid_t2s_metric:
            best_valid_density_t2s = to_log[DENSITY_METRIC_SUP_t2s]
        print(f"-----------------  best s2t = {best_valid_s2t_metric}, best t2s = {best_valid_t2s_metric} --------------")
        print(f"-----------------  best s2t density = {best_valid_density_s2t}, best t2s density = {best_valid_density_t2s} --------------")
    
    if args.export_emb:
        np_src_emb, src_dict = reload_bin_embeddings(args, True, args.max_vocab_src)
        np_tgt_emb, tgt_dict = reload_bin_embeddings(args, False, args.max_vocab_tgt)
        # prepare embeddings
        src_emb = torch.from_numpy(np_src_emb).float().to(device)
        tgt_emb = torch.from_numpy(np_tgt_emb).float().to(device)
    
        normalize_embeddings(src_emb, args.normalize_embeddings)
        normalize_embeddings(tgt_emb, args.normalize_embeddings)
        export_embeddings(model, src_emb, tgt_emb, save_path, src_dict, tgt_dict, s2t, t2s)
