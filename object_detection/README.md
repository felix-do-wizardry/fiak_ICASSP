# FiAK - Object Detection with SWin transformer

## Installation
Please refer to [get_started.md](docs/get_started.md) for installation and dataset preparation.


## Experiments

### Finetuning FiAKformer from pretrained Swin-T
```bash
python -m torch.distributed.launch --master_port 1 --nproc_per_nod8 \
tools/train.py configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py --work-dir /data/coco/swin_chks/fiak_test --launcher pytorch ${@:3}
```

### Attention score pruning via FiAK
```bash
python -m torch.distributed.launch --master_port 1 --nproc_per_nod8 \
tools/train.py configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco_prune.py --work-dir /data/coco/swin_chks/fiak_test --launcher pytorch ${@:3}
```
