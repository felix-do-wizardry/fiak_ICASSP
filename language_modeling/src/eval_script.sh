# Evaluation
CUDA_VISIBLE_DEVICES="0" python eval_sliding_window.py --cuda --data ../data/wikitext-103/ \
--dataset wt103 --split 'test' --batch_size 1 --tgt_len 256 --mem_len 0  \
--clamp_len 256 --work_dir checkpoints/8head-gmm-gd-seed-1111-matrix