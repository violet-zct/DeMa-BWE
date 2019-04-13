import argparse
from tools.args import *
from tools.utils import *
import torch
import random
import time
from modules.emb2emb import E2E
from collections import OrderedDict
from evaluation import Evaluator
import json
import gc

from tools.lazy_reader import *

parser = argparse.ArgumentParser(description="embedding to embedding")
add_e2e_args(parser)
args = parser.parse_args()

print('\n'.join('%s: %s' % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
exp_path = get_exp_path("../saved_exps/" + args.model_name + "/")
log_path = exp_path + "exp.log"

assert args.src_emb_path is not None and args.tgt_emb_path is not None
args.src_lang = args.src_emb_path.split("/")[-1].split(".")[1]
args.tgt_lang = args.tgt_emb_path.split("/")[-1].split(".")[1]

if args.valid_option == "unsup":
    # unsupervised validation metric
    VALIDATION_METRIC_s2t = args.src_lang + "-" + args.tgt_lang + '-mean_cosine-csls_knn_10-S2T-10000'
    VALIDATION_METRIC_t2s = args.tgt_lang + "-" + args.src_lang + '-mean_cosine-csls_knn_10-S2T-10000'
elif args.valid_option == "train":
    # validation metric on the sampled training set
    VALIDATION_METRIC_s2t = "valid-" + args.src_lang + "-" + args.tgt_lang + '-precision_at_1-csls_knn_10'
    VALIDATION_METRIC_t2s = "valid-" + args.tgt_lang + "-" + args.src_lang + '-precision_at_1-csls_knn_10'
else:
    raise NotImplementedError

# results on the test set with csls
VALIDATION_METRIC_SUP_s2t = args.src_lang + "-" + args.tgt_lang + '-precision_at_1-csls_knn_10'
VALIDATION_METRIC_SUP_t2s = args.tgt_lang + "-" + args.src_lang + '-precision_at_1-csls_knn_10'

# results on the test set with csls-d
DENSITY_METRIC_SUP_s2t = args.src_lang + "-" + args.tgt_lang + '-precision_at_1-density'
DENSITY_METRIC_SUP_t2s = args.tgt_lang + "-" + args.src_lang + '-precision_at_1-density'

best_valid_s2t_metric = 1e-12
best_valid_t2s_metric = 1e-12

# best csls results selected by validation
best_valid_density_s2t_metric = 1e-12
best_valid_density_t2s_metric = 1e-12

# best csls-d results selected by validation
best_valid_density_s2t_train_metric = 1e-12
best_valid_density_t2s_train_metric = 1e-12

# true best on test
best_csls_s2t = 1e-12
best_csls_t2s = 1e-12
best_density_s2t = 1e-12
best_density_t2s = 1e-12

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
src_emb_for_mog = src_emb[:args.src_train_most_frequent]
tgt_emb_for_mog = tgt_emb[:args.tgt_train_most_frequent]

if args.t2s_s_var == 0:
    args.t2s_s_var = args.s_var
    args.s2t_t_var = args.t_var

if args.init_var:
    src_var = load_txt_var(args, True, src_dict.word2id)
    tgt_var = load_txt_var(args, False, tgt_dict.word2id)
    src_var = torch.from_numpy(src_var).float().to(device)
    tgt_var = torch.from_numpy(tgt_var).float().to(device)

    src_dict.var = src_var
    tgt_dict.var = tgt_var


model = E2E(args, src_dict, tgt_dict, src_emb_for_mog, tgt_emb_for_mog, device).to(device)
evaluator = Evaluator(model, src_emb, tgt_emb)

# prepare supervised batches
batch_size = args.batch_size

if args.supervise_id:
    src_in_dict = src_emb[model.dict[:, 0], :].to(device)
    tgt_in_dict = tgt_emb[model.dict[:, 1], :].to(device)
    tot_sup_batches = math.ceil(src_in_dict.size(0) / args.batch_size)
    sup_batch_inds = np.random.permutation(range(src_in_dict.size(0)))


def get_sup_batches(batch_inds, batch_id):
    if batch_size > src_in_dict.size(0):
        return src_in_dict, tgt_in_dict
    src_batch = src_in_dict[batch_inds[batch_id * batch_size: min((batch_id + 1) * batch_size, src_in_dict.size(0))]]
    tgt_batch = tgt_in_dict[batch_inds[batch_id * batch_size: min((batch_id + 1) * batch_size, tgt_in_dict.size(0))]]
    return src_batch, tgt_batch


def check_dict(dict):
    for i in range(10):
        print(src_dict.id2word[dict[i][0].item()], tgt_dict.id2word[dict[i][1].item()])

train_step = 1
n_words_proc = 0
tic = time.time()

training_stats = {"disc_loss": [], "S2T_nll": [], "T2S_nll": [], "adv_loss": [], "sup_loss": [],
                  "Sup_S2T": [], "Sup_T2S": [],
                  "Cos_S": [], "Cos_T": [], "flow_tot_loss": [], "Diag_S": [], "Diag_T": [],
                  "BT_S2T": [], "BT_T2S": [], "step": train_step}



src_sampler = words_sampler_iterator(probs=np_src_freqs, buffer_size=args.src_base_batch_size,
                                    batch_size=args.batch_size, uniform_sample=args.uniform_sample)
tgt_sampler = words_sampler_iterator(probs=np_tgt_freqs, buffer_size=args.tgt_base_batch_size,
                                    batch_size=args.batch_size, uniform_sample=args.uniform_sample)

base_src_idx = src_sampler.retrieve_cache()
base_tgt_idx = tgt_sampler.retrieve_cache()

for src_idx, tgt_idx in zip(src_sampler, tgt_sampler):
    if train_step > 0 and train_step % args.display_steps == 0:
        ss = 'Step=%i, %i samples/s' % (train_step, int(n_words_proc / (time.time() - tic)))
        flow_lr = model.flow_scheduler.get_lr()[0]
        ss += ", flow lr=%.6f" % flow_lr
        for k, v in training_stats.items():
            if type(v) != list or len(v) == 0:
                continue
            ss += ", %s=%.4f" % (k, np.mean(v))
        print(ss)
        for k, v in training_stats.items():
            if type(training_stats[k]) == list:
                del training_stats[k][:]

        if train_step % 1000 == 0:
            n_words_proc = 0
            tic = time.time()

    if args.supervise_id:
        sup_src_batch, sup_tgt_batch = src_in_dict, tgt_in_dict
    else:
        sup_src_batch = sup_tgt_batch = None

    model.flow_step(base_src_idx, base_tgt_idx, src_idx, tgt_idx, training_stats, sup_src_batch, sup_tgt_batch)
    n_words_proc += len(src_idx) * 2

    if train_step > 0 and train_step % args.valid_steps == 0:
        gc.collect()
        to_log = OrderedDict({'train_iters': train_step, 'exp_path': exp_path})
        evaluator.all_eval(to_log, train=True, unsup_eval=args.valid_option=="unsup")

        if to_log[VALIDATION_METRIC_s2t] > best_valid_s2t_metric:
            model.set_save_s2t_path(exp_path + "best_s2t_params.bin")
            model.save_best_s2t()
            best_valid_s2t_metric = to_log[VALIDATION_METRIC_s2t]
            best_valid_csls_s2t_metric = to_log[VALIDATION_METRIC_SUP_s2t]
            best_valid_density_s2t_metric = to_log[DENSITY_METRIC_SUP_s2t]

        if to_log[VALIDATION_METRIC_t2s] > best_valid_t2s_metric:
            model.set_save_t2s_path(exp_path + "best_t2s_params.bin")
            model.save_best_t2s()
            best_valid_t2s_metric = to_log[VALIDATION_METRIC_t2s]
            best_valid_csls_t2s_metric = to_log[VALIDATION_METRIC_SUP_t2s]
            best_valid_density_t2s_metric = to_log[DENSITY_METRIC_SUP_t2s]

        if to_log[VALIDATION_METRIC_SUP_s2t] > best_csls_s2t:
            best_csls_s2t = to_log[VALIDATION_METRIC_SUP_s2t]

        if to_log[VALIDATION_METRIC_SUP_t2s] > best_csls_t2s:
            best_csls_t2s = to_log[VALIDATION_METRIC_SUP_t2s]

        if to_log[DENSITY_METRIC_SUP_s2t] > best_density_s2t:
            best_density_s2t = to_log[DENSITY_METRIC_SUP_s2t]

        if to_log[DENSITY_METRIC_SUP_t2s] > best_density_t2s:
            best_density_t2s = to_log[DENSITY_METRIC_SUP_t2s]

        print("Selected via metric with %s!" % args.valid_option)
        print(f"-----------------  best valid csls s2t = {best_valid_csls_s2t_metric}, "
              f"best valid csls t2s = {best_valid_csls_t2s_metric} --------------")

        print(f"-----------------  best valid density s2t = {best_valid_density_s2t_metric}, "
              f"best valid density t2s = {best_valid_density_t2s_metric} --------------")

        print("Evaluation on the test set!")
        print(f"-----------------  best test csls s2t = {best_csls_s2t}, best test csls t2s = {best_csls_t2s} --------------")
        print(f"-----------------  best test density s2t = {best_density_s2t}, best test density t2s = {best_density_t2s} --------------")

        print(json.dumps(to_log))

        if model.flow_scheduler.get_lr()[0] < args.min_lr:
            print('Learning rate < 1e-6. BREAK.')
            break

    train_step += 1
    if train_step > args.n_steps:
        print("Reach maximum training step. BREAK.")
        break

    training_stats["step"] = train_step

if args.export_emb:
    model.export_embeddings(src_emb, tgt_emb, exp_path)
