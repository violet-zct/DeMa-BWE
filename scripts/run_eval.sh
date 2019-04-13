#!/bin/bash

source activate dema

src_lang="en"
tgt_lang="de"

MODEL_NAME=eval_${src_lang}_${tgt_lang}
load_path="../saved_exps/${tgt_lang}"
data_path="../data"

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 python -u ../evaluation/eval.py \
    --load_from_pretrain_s2t ${load_path}/best_s2t_params.bin \
    --load_from_pretrain_t2s ${load_path}/best_t2s_params.bin \
    --model_name ${MODEL_NAME} \
    --export_emb 1 \
    --n_refinement 0 \
    --src_emb_path $data_path/fasttext/wiki.${src_lang}.bin \
    --tgt_emb_path $data_path/fasttext/wiki.${tgt_lang}.bin \
    --s_var 0.01 \
    --s2t_t_var 0.015 \
    --t_var 0.015 \
    --t2s_s_var 0.01 \
    --max_vocab_src 50000 \
    --max_vocab_tgt 50000 \
    --max_vocab 200000 \
    --dico_max_rank 25000 \
    --sup_dict_path $data_path/crosslingual/dictionaries/${src_lang}-${tgt_lang}.0-5000.txt \
    --dico_eval $data_path/crosslingual/dictionaries/${src_lang}-${tgt_lang}.5000-6500.txt 2>&1 | tee eval_${tgt_lang}.log
