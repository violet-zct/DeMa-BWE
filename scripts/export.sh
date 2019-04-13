#!/bin/bash

source activate dema

src_lang="en"
tgt_lang="de"
load_path="../saved_exps/${tgt_lang}"
data_path="../data"

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 python -u ../evaluation/export_embs.py \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --s2t_map_path ${load_path}/best_s2t_params.bin \
    --t2s_map_path ${load_path}/best_t2s_params.bin \
    --src_emb_path $data_path/fasttext/wiki.${src_lang}.bin \
    --tgt_emb_path $data_path/fasttext/wiki.${tgt_lang}.bin \
    --vocab_size 50000
