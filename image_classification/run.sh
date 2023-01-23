##baseline
python -m torch.distributed.launch --master_port 190 --nproc_per_node=2 --use_env main.py --model deit_tiny_patch16_224 --batch-size 256 --data-path path/data --output_dir path/output

##FiAK
python -m torch.distributed.launch --master_port 190 --nproc_per_node=2 --use_env main.py --model deit_tiny_gmm_gd_patch16_224 --batch-size 256 --data-path path/data --output_dir path/output

##Attention score pruning via FiAK with pruning fraction is 70%
python -m torch.distributed.launch --master_port 190 --nproc_per_node=2 --use_env main1.py --model deit_tiny_gmm_gd_patch16_224 --batch-size 256 --data-path path/data --output_dir path/output --epochs 100 --amount 0.7 --lr 5e-5 --prune_type 'entry'

##Mixed pruning via FiAK with total pruning fraction is 70% and key pruning fraction is 15%
python -m torch.distributed.launch --master_port 190 --nproc_per_node=2 --use_env main1.py --model deit_tiny_gmm_gd_patch16_224 --batch-size 256 --data-path path/data --output_dir path/output --epochs 100 --amount 0.7 --lr 5e-5 --prune_type 'entry_token' --amount_token 0.15