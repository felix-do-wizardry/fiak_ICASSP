# FiAK - Language Modeling on WikiText-103

## Installation
Please refer to [get_started.md](docs/get_started.md) for installation and dataset preparation.


## Experiments

### Training FiAKformer
```bash
python train.py --cuda --data ../data/wikitext-103/ \
--dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 \
--dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 150000 --attn_type 202 \
--tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 \
--multi_gpu --project_name 'mgk' --seed 1111 --job_name 8head-gmm-gd-seed-1111-matrix \
--work_dir checkpoints/8head-gmm-gd-seed-1111-matrix --update_mode 'not_vector' --use_wandb
```

### Attention score pruning via FiAK
```bash
python prune_finetune.py --cuda --data ../data/wikitext-103/ \
--dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 \
--dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 150000 --attn_type 202 \
--tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --amount 0.4 --key_amount 0.1 \
--multi_gpu --project_name 'mgk' --seed 1111 --job_name 8head-gmm-gd-seed-1111-matrix_0.4_mixed_0.1 \
--work_dir checkpoints/8head-gmm-gd-seed-1111-matrix --update_mode 'not_vector' --load_model --use_wandb
```

