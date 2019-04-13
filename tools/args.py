def add_e2e_args(parser):
    parser.add_argument("--cuda", default=False, action="store_true")
    parser.add_argument("--random_seed", default=6700417, type=int)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--load_from_pretrain_s2t", type=str)
    parser.add_argument("--load_from_pretrain_t2s", type=str)
    parser.add_argument("--emb_dim", default=300, type=int)
    parser.add_argument("--export_emb", type=int, default=0)
    parser.add_argument("--valid_option", type=str, choices=["train", "unsup"], default="unsup")

    parser.add_argument("--sup_s_weight", default=10., type=float, help="s2t")
    parser.add_argument("--sup_t_weight", default=10., type=float, help="ts2")

    parser.add_argument("--src_emb_path", type=str, default=None, help="wiki.xx.bin, in binary format")
    parser.add_argument("--tgt_emb_path", type=str, default=None, help="wiki.xx.bin, in binary format")
    parser.add_argument("--init_var", type=int, default=0, help="if True, set var path")
    parser.add_argument("--src_var_path", type=str, default=None, help="if pretrain variance for each src word is provided")
    parser.add_argument("--tgt_var_path", type=str, default=None, help="if pretrain variance for each tgt word is provided")

    parser.add_argument("--src_lang", type=str)
    parser.add_argument("--tgt_lang", type=str)
    parser.add_argument("--normalize_embeddings", type=str, default="double", choices=['',  'double', 'renorm', 'center', 'rescale'],
                        help="Normalize embeddings before training")

    # obj
    parser.add_argument("--supervise_id", type=int, default=0, help="if add supervise identical pairs")
    parser.add_argument("--sup_obj", type=str, default="cosine", choices=["mse", "cosine"])

    parser.add_argument("--n_steps", type=int, default=5, help="Maximum number of steps to be trained")
    parser.add_argument("--batch_size", type=int, default=512, help="training Batch size")

    parser.add_argument("--src_base_batch_size", type=int, default=2048, help="base distribution batch size")
    parser.add_argument("--tgt_base_batch_size", type=int, default=2048)

    parser.add_argument("--valid_steps", type=int, default=5000)
    parser.add_argument("--display_steps", type=int, default=500)
    parser.add_argument("--flow_opt_params", type=str, default="adam", help="flow optimizer")
    parser.add_argument("--lr_decay", type=float, default=0.999999, help="Learning rate decay (SGD only)")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate (SGD only)")

    # dictionary params
    parser.add_argument("--max_vocab", type=int, default=200000)
    parser.add_argument("--max_vocab_src", type=int, default=50000, help="Maximum vocabulary size (-1 to disable)")
    parser.add_argument("--max_vocab_tgt", type=int, default=50000, help="Maximum vocabulary size (-1 to disable)")

    # Flow / generator
    parser.add_argument("--smooth_c", type=float, default=0.01, help="smooth the empirical dist, the larger, the smoother")

    parser.add_argument("--uniform_sample", default=1, type=int)
    parser.add_argument("--src_train_most_frequent", default=10000, type=int)
    parser.add_argument("--tgt_train_most_frequent", default=10000, type=int)

    parser.add_argument("--cofreq", type=int, default=1, help="use the normalized freq/rank similarities as weights")
    parser.add_argument("--temp", type=float, default=2.)

    # mixture of gaussians args
    parser.add_argument("--s_var", default=0.01, type=float, help="s2t_s_var, noise")
    parser.add_argument("--t_var", default=0.015, type=float, help="t2s_t_var, noise")
    parser.add_argument("--s2t_t_var", default=0.015, type=float, help="base var")
    parser.add_argument("--t2s_s_var", default=0.01, type=float, help="base var")
    parser.add_argument("--back_translate_src_w", default=0.5, type=float)
    parser.add_argument("--back_translate_tgt_w", default=0.5, type=float)

    # training refinement
    parser.add_argument("--n_refinement", type=int, default=0,
                        help="Number of refinement iterations (0 to disable the refinement procedure)")
    # dictionary creation parameters (for refinement)
    parser.add_argument("--dico_batch_size", type=int, default=512, help="for creating newly mapped embeddings")
    parser.add_argument("--sup_dict_path", type=str, default="default",
                        help="Path to training dictionary (default: use identical character strings)")
    parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")#, required=True)
    parser.add_argument("--dico_method", type=str, default='csls_knn_10',
                        help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10/density)")
    parser.add_argument("--dico_build", type=str, default='S2T', help="S2T,T2S,S2T|T2S,S2T&T2S")
    parser.add_argument("--dico_threshold", type=float, default=0,
                        help="Threshold confidence for dictionary generation")
    parser.add_argument("--dico_max_rank", type=int, default=15000, help="Maximum dictionary words rank (0 to disable)")
    parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
    parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")