import torch.nn as nn
from modules.model import Model
from modules.flows.mog_flow import MogFlow_batch
import torch
from tools.utils import *
from tools.dico_builder import build_dictionary
import torch.nn.functional as F
from evaluation.word_translation import *
from torch.nn import CosineEmbeddingLoss
import codecs
import scipy

class E2E(Model):
    def __init__(self, args, src_dict, tgt_dict, src_embedding, tgt_embedding, device):
        super(E2E, self).__init__(args)

        self.args = args
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

        # src_flow: assume tgt embeddings are transformed from the src mog space
        self.register_buffer('src_embedding', src_embedding)
        self.register_buffer('tgt_embedding', tgt_embedding)

        if args.init_var:
            # initialize with gaussian variance
            self.register_buffer("s2t_s_var", src_dict.var)
            self.register_buffer("s2t_t_var", tgt_dict.var)
            self.register_buffer("t2s_s_var", src_dict.var)
            self.register_buffer("t2s_t_var", tgt_dict.var)
        else:
            self.s2t_s_var = args.s_var
            self.s2t_t_var = args.s2t_t_var
            self.t2s_t_var = args.t_var
            self.t2s_s_var = args.t2s_s_var

        self.register_buffer('src_freqs', torch.tensor(src_dict.freqs, dtype=torch.float))
        self.register_buffer('tgt_freqs', torch.tensor(tgt_dict.freqs, dtype=torch.float))

        # backward: t2s
        self.src_flow = MogFlow_batch(args, self.t2s_s_var)
        # backward: s2t
        self.tgt_flow = MogFlow_batch(args, self.s2t_t_var)
        
        self.s2t_valid_dico = None
        self.t2s_valid_dico = None

        self.device = device
        # use dict pairs from train data (supervise) or identical words (supervise_id) as supervisions
        self.supervise = args.supervise_id
        if self.supervise:
            self.load_training_dico()
            if args.sup_obj == 'mse':
                self.sup_loss_func = nn.MSELoss()
            elif args.sup_obj == 'cosine':
                self.sup_loss_func = CosineEmbeddingLoss()

        optim_fn, optim_params= get_optimizer(args.flow_opt_params)
        self.flow_optimizer = optim_fn(list(self.src_flow.parameters()) + list(self.tgt_flow.parameters()), **optim_params)
        self.flow_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.flow_optimizer, gamma=args.lr_decay)

        self.best_valid_metric = 1e-12

        self.sup_sw = args.sup_s_weight
        self.sup_tw = args.sup_t_weight

        self.mse_loss = nn.MSELoss()
        self.cos_loss = CosineEmbeddingLoss()

        # Evaluation on trained model
        if args.load_from_pretrain_s2t != "" or args.load_from_pretrain_t2s != "":
            self.load_from_pretrain()

    def orthogonalize(self):
        """
        Orthogonalize the mapping.
        """
        W1 = self.src_flow.W
        W2 = self.tgt_flow.W
        beta = 0.01

        with torch.no_grad():
            for _ in range(self.args.ortho_steps):
                W1.copy_((1 + beta) * W1 - beta * W1.mm(W1.transpose(0, 1).mm(W1)))
                W2.copy_((1 + beta) * W2 - beta * W2.mm(W2.transpose(0, 1).mm(W2)))

    def sup_step(self, src_emb, tgt_emb):
        src_to_tgt, tgt_to_src, _, _ = self.run_flow(src_emb, tgt_emb, 'both', False)
        if self.args.sup_obj == "mse":
            s2t_sim = (src_to_tgt * tgt_emb).sum(dim=1)
            s2t_sup_loss = self.sup_loss_func(s2t_sim, torch.ones_like(s2t_sim))
            t2s_sim = (tgt_to_src * src_emb).sum(dim=1)
            t2s_sup_loss = self.sup_loss_func(t2s_sim, torch.ones_like(t2s_sim))
            loss = s2t_sup_loss + t2s_sup_loss
        elif self.args.sup_obj == "cosine":
            target = torch.ones(src_emb.size(0)).to(self.device)
            s2t_sup_loss = self.sup_loss_func(src_to_tgt, tgt_emb, target)
            t2s_sup_loss = self.sup_loss_func(tgt_to_src, src_emb, target)
            loss = s2t_sup_loss + t2s_sup_loss
        else:
            raise NotImplementedError
        # check NaN
        if (loss != loss).data.any():
            print("NaN detected (supervised loss)")
            exit()

        return s2t_sup_loss, t2s_sup_loss, loss

    def flow_step(self, base_src_ids, base_tgt_ids, src_ids, tgt_ids, training_stats, src_emb_in_dict=None, tgt_emb_in_dict=None):
        src_emb = self.src_embedding[src_ids]
        tgt_emb = self.tgt_embedding[tgt_ids]
        base_src_emb = self.src_embedding[base_src_ids]
        base_tgt_emb = self.tgt_embedding[base_tgt_ids]

        base_src_var = base_tgt_var = None
        if self.args.init_var:
            train_src_var = self.s2t_s_var[src_ids]
            base_src_var = self.t2s_s_var[base_src_ids]
            train_tgt_var = self.t2s_t_var[tgt_ids]
            base_tgt_var = self.s2t_t_var[base_tgt_ids]
            src_std = torch.sqrt(train_src_var).unsqueeze(1)
            tgt_std = torch.sqrt(train_tgt_var).unsqueeze(1)
        else:
            src_std = math.sqrt(self.s2t_s_var)
            tgt_std = math.sqrt(self.t2s_t_var)

        src_emb = src_emb + torch.randn_like(src_emb) * src_std
        tgt_emb = tgt_emb + torch.randn_like(tgt_emb) * tgt_std

        if self.args.cofreq:
            # ids of words are their frequency ranks
            train_src_freq = src_emb.new_tensor(src_ids) + 1.
            train_tgt_freq = tgt_emb.new_tensor(tgt_ids) + 1.
            base_src_freq = src_emb.new_tensor(base_src_ids) + 1.
            base_tgt_freq = tgt_emb.new_tensor(base_tgt_ids) + 1.
        else:
            train_src_freq = train_tgt_freq = None
            src_freq_normalized = self.src_freqs[base_src_ids]
            src_freq_normalized = src_freq_normalized / src_freq_normalized.sum()
            tgt_freq_normalized = self.tgt_freqs[base_tgt_ids]
            tgt_freq_normalized = tgt_freq_normalized / tgt_freq_normalized.sum()
            base_src_freq = torch.log(src_freq_normalized)
            base_tgt_freq = torch.log(tgt_freq_normalized)

        src_to_tgt, src_ll = self.tgt_flow.backward(src_emb, x=base_tgt_emb, x_freqs=base_tgt_freq,
                                                    require_log_probs=True, var=base_tgt_var, y_freqs=train_src_freq)
        tgt_to_src, tgt_ll = self.src_flow.backward(tgt_emb, x=base_src_emb, x_freqs=base_src_freq,
                                                    require_log_probs=True, var=base_src_var, y_freqs=train_tgt_freq)
        # the log density of observing src embeddings (transformm to target space)
        src_nll, tgt_nll = -src_ll.mean(), -tgt_ll.mean()
        loss = src_nll + tgt_nll

        if self.args.back_translate_src_w > 0 and self.args.back_translate_tgt_w > 0:
            target = torch.ones(src_emb.size(0)).to(self.device)

            tgt_to_src_to_tgt, src_to_tgt_to_src,  _, _ = self.run_flow(tgt_to_src, src_to_tgt, 'both', False)

            src_bt_loss = self.cos_loss(src_emb, src_to_tgt_to_src, target)
            tgt_bt_loss = self.cos_loss(tgt_emb, tgt_to_src_to_tgt, target)

            bt_w_src = self.args.back_translate_src_w
            bt_w_tgt = self.args.back_translate_src_w
            loss = loss + bt_w_src * src_bt_loss + bt_w_tgt * tgt_bt_loss
            training_stats["BT_S2T"].append(src_bt_loss.item())
            training_stats["BT_T2S"].append(tgt_bt_loss.item())

        if self.supervise:
            assert src_emb_in_dict is not None, tgt_emb_in_dict is not None
            s2t_sup_loss, t2s_sup_loss, sup_loss = self.sup_step(src_emb_in_dict, tgt_emb_in_dict)
            loss = loss + self.sup_sw * s2t_sup_loss + self.sup_tw * t2s_sup_loss
            training_stats["Sup_S2T"].append(s2t_sup_loss.item())
            training_stats["Sup_T2S"].append(t2s_sup_loss.item())
        else:
            sup_loss = torch.tensor(0.0)

        loss.backward()

        self.flow_optimizer.step()
        self.flow_scheduler.step()
        self.flow_optimizer.zero_grad()

        loss, src_nll, tgt_nll, sup_loss = loss.item(), src_nll.item(), tgt_nll.item(), sup_loss.item()

        if self.args.cuda:
            torch.cuda.empty_cache()
        training_stats["S2T_nll"].append(src_nll)
        training_stats["T2S_nll"].append(tgt_nll)

    def load_training_dico(self):
        """
        Load training dictionary.
        """
        word2id1 = self.src_dict.word2id
        word2id2 = self.tgt_dict.word2id
        valid_dico_size = 1000
        if self.args.supervise_id > 0:
            id_dict_1, id_dict_2 = load_identical_char_dico(word2id1, word2id2)
            print("Idenditical dictionary pairs = %d, %d" % (id_dict_1.size(0), id_dict_2.size(0)))
            dict = id_dict_1[:self.args.supervise_id, :]
        else:
            dict = torch.tensor(0)

        if self.args.valid_option == "train":
            dict_s2t = load_dictionary(self.args.sup_dict_path, word2id1, word2id2)
            t2s_dict_path = os.path.join(os.path.dirname(self.args.sup_dict_path),  self.tgt_dict.lang + "-" + self.src_dict.lang + ".0-5000.txt")
            dict_t2s = load_dictionary(t2s_dict_path, word2id2, word2id1, reverse=True)

            ids_s2t = list(np.random.permutation(range(dict_s2t.size(0))))
            ids_t2s = list(np.random.permutation(range(dict_t2s.size(0))))

            self.s2t_valid_dico = dict_s2t[ids_s2t[0: valid_dico_size], :]
            self.t2s_valid_dico = dict_t2s[ids_t2s[0: valid_dico_size], :]
            self.t2s_valid_dico = torch.cat([dict_t2s[:, 1].unsqueeze(1), dict_t2s[:, 0].unsqueeze(1)], dim=1)
            print("Loading validation dictionary: %d %d" % (self.s2t_valid_dico.size(0), self.t2s_valid_dico.size(0)))

            for w1, w2 in self.s2t_valid_dico[:100]:
                print(self.src_dict.id2word[w1.item()], self.tgt_dict.id2word[w2.item()])
            print("-" * 30)
            for w1, w2 in self.t2s_valid_dico[:100]:
                print(self.tgt_dict.id2word[w1.item()], self.src_dict.id2word[w2.item()])

        print("Pruning dictionary pairs = %d" % dict.size(0))

        # toch.LongTensor: [len(pairs), 2]
        self.dict = dict

    def run_flow(self, src_emb=None, tgt_emb=None, side="both", require_logll=True):
        if side == "src":
            # from src to tgt
            assert src_emb is not None
            src_to_tgt, src_log_ll = self.tgt_flow.backward(src_emb, require_log_probs=require_logll)
            return src_to_tgt, src_log_ll
        elif side == "tgt":
            assert tgt_emb is not None
            tgt_to_src, tgt_log_ll = self.src_flow.backward(tgt_emb, require_log_probs=require_logll)
            return tgt_to_src, tgt_log_ll
        elif side == "both":
            assert tgt_emb is not None and src_emb is not None
            src_to_tgt, src_log_ll = self.tgt_flow.backward(src_emb, require_log_probs=require_logll)
            tgt_to_src, tgt_log_ll = self.src_flow.backward(tgt_emb, require_log_probs=require_logll)
            return src_to_tgt, tgt_to_src, src_log_ll, tgt_log_ll

    def map_embs(self, src_emb, tgt_emb, s2t=True, t2s=True):
        src2tgt_emb = tgt2src_emb = None
        with torch.no_grad():
            if s2t:
                src_to_tgt_list = []
                for i, j in get_batches(src_emb.size(0), self.args.dico_batch_size):
                    src_emb_batch = src_emb[i:j, :]#.to(self.device)
                    src_to_tgt, _ = self.run_flow(src_emb=src_emb_batch, side="src", require_logll=False)
                    src_to_tgt_list.append(src_to_tgt.cpu())
                # reside on cpu
                src2tgt_emb = torch.cat(src_to_tgt_list, dim=0)
            if t2s:
                tgt_to_src_list = []
                for i, j in get_batches(tgt_emb.size(0), self.args.dico_batch_size):
                    tgt_emb_batch = tgt_emb[i:j, :]#.to(self.device)
                    tgt_to_src, _ = self.run_flow(tgt_emb=tgt_emb_batch, side="tgt", require_logll=False)
                    tgt_to_src_list.append(tgt_to_src.cpu())
                tgt2src_emb = torch.cat(tgt_to_src_list, dim=0)

        return src2tgt_emb, tgt2src_emb

    def build_dictionary(self, src_emb, tgt_emb, s2t=True, t2s=True):
        # Build dictionary with current trained mappings to augment the original dictionary
        src_to_tgt_emb, tgt_to_src_emb = self.map_embs(src_emb, tgt_emb, s2t=s2t, t2s=t2s)
        # torch.longTensor
        topk = 50000
        if s2t:
            self.build_s2t_dict = torch.cat([self.dict_s2t, build_dictionary(src_to_tgt_emb.cuda()[:topk],
                                                                             tgt_emb[:topk], self.args)], dim=0)
            s2t = self.build_s2t_dict
            for i in range(300, 320):
                print(self.src_dict.id2word[s2t[i, 0].item()], self.tgt_dict.id2word[s2t[i, 1].item()])

        if t2s:
            self.build_t2s_dict = torch.cat([self.dict_t2s, build_dictionary(tgt_to_src_emb.cuda()[:topk], src_emb[:topk], self.args)], dim=0)
            t2s = self.build_t2s_dict

            print("---" * 20)
            for i in range(300, 320):
                print(self.src_dict.id2word[t2s[i, 1].item()], self.tgt_dict.id2word[t2s[i, 0].item()])

    def procrustes(self, src_emb, tgt_emb, s2t=True, t2s=True):
        """
        Find the best orthogonal matrix mapping using the Orthogonal Procrustes problem
        https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        """
        if s2t:
            A = src_emb[self.build_s2t_dict[:, 0]]
            B = tgt_emb[self.build_s2t_dict[:, 1]]
            W = self.tgt_flow.W
            M = B.transpose(0, 1).mm(A).cpu().numpy()
            U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
            with torch.no_grad():
                W.copy_(torch.from_numpy((U.dot(V_t)).transpose()).type_as(W))

        if t2s:
            A = tgt_emb[self.build_t2s_dict[:, 0]]
            B = src_emb[self.build_t2s_dict[:, 1]]
            W2 = self.src_flow.W
            M = B.transpose(0, 1).mm(A).cpu().numpy()
            U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
            with torch.no_grad():
                W2.copy_(torch.from_numpy((U.dot(V_t)).transpose()).type_as(W2))

    def load_best_from_both_sides(self):
        self.load_best_s2t()
        self.load_best_t2s()

    def load_best_s2t(self):
        print("Load src to tgt mapping to %s" % self.s2t_save_to)
        to_reload = torch.from_numpy(torch.load(self.s2t_save_to))
        with torch.no_grad():
            W1 = self.tgt_flow.W
            W1.copy_(to_reload.type_as(W1))

    def load_best_t2s(self):
        print("Load src to tgt mapping to %s" % self.t2s_save_to)
        to_reload = torch.from_numpy(torch.load(self.t2s_save_to))
        with torch.no_grad():
            W1 = self.src_flow.W
            W1.copy_(to_reload.type_as(W1))

    def save_best_s2t(self):
        print("Save src to tgt mapping to %s" % self.s2t_save_to)
        with torch.no_grad():
            torch.save(self.tgt_flow.W.cpu().numpy(), self.s2t_save_to)

    def save_best_t2s(self):
        print("Save tgt to src mapping to %s" % self.t2s_save_to)
        with torch.no_grad():
            torch.save(self.src_flow.W.cpu().numpy(), self.t2s_save_to)

    def export_embeddings(self, src_emb, tgt_emb, exp_path):
        self.load_best_from_both_sides()
        mapped_src_emb, mapped_tgt_emb = self.map_embs(src_emb, tgt_emb)
        src_path = exp_path + self.src_dict.lang + "2" + self.tgt_dict.lang + "_emb.vec"
        tgt_path = exp_path + self.tgt_dict.lang + "2" + self.src_dict.lang + "_emb.vec"

        mapped_src_emb = mapped_src_emb.cpu().numpy()
        mapped_tgt_emb = mapped_tgt_emb.cpu().numpy()

        print(f'Writing source embeddings to {src_path}')
        with io.open(src_path, 'w', encoding='utf-8') as f:
            f.write(u"%i %i\n" % mapped_src_emb.shape)
            for i in range(len(self.src_dict)):
                f.write(u"%s %s\n" % (self.src_dict[i], " ".join('%.5f' % x for x in mapped_src_emb[i])))
        print(f'Writing target embeddings to {tgt_path}')
        with io.open(tgt_path, 'w', encoding='utf-8') as f:
            f.write(u"%i %i\n" % mapped_tgt_emb.shape)
            for i in range(len(self.tgt_dict)):
                f.write(u"%s %s\n" % (self.tgt_dict[i], " ".join('%.5f' % x for x in mapped_tgt_emb[i])))

    def load_from_pretrain(self):
        # load src to tgt W for tgt flow
        if self.args.load_from_pretrain_s2t is not None:
            print("Loading from pretrained model %s!" % self.args.load_from_pretrain_s2t)
            with torch.no_grad():
                s2t = torch.from_numpy(torch.load(self.args.load_from_pretrain_s2t))
                W1 = self.tgt_flow.W
                W1.copy_(s2t.type_as(W1))

        if self.args.load_from_pretrain_t2s is not None:
            print("Loading from pretrained model %s!" % self.args.load_from_pretrain_t2s)
            with torch.no_grad():
                t2s = torch.from_numpy(torch.load(self.args.load_from_pretrain_t2s))
                W2 = self.src_flow.W
                W2.copy_(t2s.type_as(W2))

    def write_topK(self, dico, topk, fname, id2word_1, id2word_2):
        dico = dico.cpu().numpy()
        topk = topk.cpu().numpy()

        assert dico.shape[0] == topk.shape[0]
        with codecs.open("../analysis/" + fname, "w", "utf-8") as fout:
            d = dict()
            for t, (w1, w2) in enumerate(dico):
                word_1 = id2word_1[w1]
                top_10 = [id2word_2[i] for i in topk[t, :]]
                if word_1 not in d:
                    d[word_1] = []
                    d[word_1].append(top_10)

                if id2word_2[w2] in top_10:
                    score = top_10.index(id2word_2[w2])
                else:
                    score = -1
                d[word_1].append((id2word_2[w2], score))

            for kword, ll in d.items():
                best_score = -1
                fout.write(kword + ": " + " ".join(["Top 10:"] + ll[0]) + "\n")
                groud_words = []
                for word_2, s in ll[1:]:
                    if s > best_score:
                        best_score = s
                    groud_words.append(word_2)
                fout.write("Ground Truth words: " + " ".join(groud_words) + "\n")
                fout.write("Best match: " + str(best_score) + "\n")
                fout.write("-" * 50 + "\n")

    def check_word_translation(self, full_src_emb, full_tgt_emb, topK=True, density=False):
        src_to_tgt_emb, tgt_to_src_emb = self.map_embs(full_src_emb, full_tgt_emb)
        s2t_path = self.src_dict.lang + "-" + self.tgt_dict.lang + ".topK"
        t2s_path = self.tgt_dict.lang + "-" + self.src_dict.lang + ".topK"

        if density:
            print("<%s> TO <%s> Evaluation!" % (self.src_dict.lang, self.tgt_dict.lang))
            for method in ['density']:
                s2t_dico, s2t_top_k = get_word_translation_accuracy(
                    self.src_dict.lang, self.src_dict.word2id, src_to_tgt_emb,  # query
                    self.tgt_dict.lang, self.tgt_dict.word2id, full_tgt_emb.cpu(),
                    method=method,
                    dico_eval=self.args.dico_eval,
                    get_scores=topK,
                    var=self.args.s2t_t_var
                )
            self.write_topK(s2t_dico, s2t_top_k, s2t_path, self.src_dict.id2word, self.tgt_dict.id2word)

            print("<%s> TO <%s> Evaluation!" % (self.tgt_dict.lang, self.src_dict.lang))
            tgt_to_src_path = os.path.join(os.path.dirname(self.args.dico_eval),
                                           self.args.tgt_lang + "-" + self.args.src_lang + ".5000-6500.txt")
            for method in ['density']:
                t2s_dico, t2s_top_k = get_word_translation_accuracy(
                    self.tgt_dict.lang, self.tgt_dict.word2id, tgt_to_src_emb,  # query
                    self.src_dict.lang, self.src_dict.word2id, full_src_emb.cpu(),
                    method=method,
                    dico_eval=tgt_to_src_path,
                    get_scores=topK,
                    var=self.args.t2s_s_var
                )
            self.write_topK(t2s_dico, t2s_top_k, t2s_path, self.tgt_dict.id2word, self.src_dict.id2word)
            return

        if topK:
            print("<%s> TO <%s> Evaluation!" % (self.src_dict.lang, self.tgt_dict.lang))
            for method in ['nn', 'csls_knn_10']:
                s2t_dico, s2t_top_k = get_word_translation_accuracy(
                    self.src_dict.lang, self.src_dict.word2id, src_to_tgt_emb,  # query
                    self.tgt_dict.lang, self.tgt_dict.word2id, full_tgt_emb.cpu(),
                    method=method,
                    dico_eval=self.args.dico_eval,
                    get_scores=topK
                )
            self.write_topK(s2t_dico, s2t_top_k, s2t_path, self.src_dict.id2word, self.tgt_dict.id2word)

            print("<%s> TO <%s> Evaluation!" % (self.tgt_dict.lang, self.src_dict.lang))
            tgt_to_src_path = os.path.join(os.path.dirname(self.args.dico_eval),
                                           self.args.tgt_lang + "-" + self.args.src_lang + ".5000-6500.txt")
            for method in ['nn', 'csls_knn_10']:
                t2s_dico, t2s_top_k = get_word_translation_accuracy(
                    self.tgt_dict.lang, self.tgt_dict.word2id, tgt_to_src_emb,  # query
                    self.src_dict.lang, self.src_dict.word2id, full_src_emb.cpu(),
                    method=method,
                    dico_eval=tgt_to_src_path,
                    get_scores=topK
                )
            self.write_topK(t2s_dico, t2s_top_k, t2s_path, self.tgt_dict.id2word, self.src_dict.id2word)
