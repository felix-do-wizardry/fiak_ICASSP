### finetuning FiAKformer from pretrained Swin-T
python -m torch.distributed.launch --master_port 1 --nproc_per_nod8 \
tools/train.py configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py --work-dir /data/coco/swin_chks/fiak_test --launcher pytorch ${@:3}

### attention score pruning via FiAK
python -m torch.distributed.launch --master_port 1 --nproc_per_nod8 \
tools/train.py configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco_prune.py --work-dir /data/coco/swin_chks/fiak_test --launcher pytorch ${@:3}