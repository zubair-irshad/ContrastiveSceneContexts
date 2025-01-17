# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash

MODEL=PointNet2Backbone

python ddp_train.py -m \
    net.model=$MODEL \
    net.conv1_kernel_size=3 \
    net.model_n_out=32 \
    opt.lr=0.1 \
    opt.max_iter=300000 \
    data.dataset=ScanNetMatchPairDataset \
    data.voxelize=False \
    data.num_points=20000 \
    data.world_space=True \
    trainer.trainer=PartitionPointNCELossTrainerPointNet \
    trainer.batch_size=128 \
    trainer.stat_freq=5 \
    trainer.checkpoint_freq=500 \
    trainer.lr_update_freq=1000 \
    shape_context.r1=0.05 \
    shape_context.r2=0.5 \
    shape_context.nbins_xy=2 \
    shape_context.nbins_zy=2 \
    shape_context.weight_inner=False \
    shape_context.fast_partition=True \
    misc.num_gpus=7 \
    misc.train_num_thread=64 \
    misc.npos=4096 \
    misc.nceT=0.4 \
    misc.out_dir=${OUT_DIR} \
    # hydra.launcher.partition=priority \
    # hydra.launcher.timeout_min=3600 \
    # hydra.launcher.max_num_timeout=5 \
    # hydra.launcher.comment=criticalExp \
