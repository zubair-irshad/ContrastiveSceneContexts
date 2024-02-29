# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import torch
import hydra
import numpy as np
from lib.ddp_trainer import DetectionTrainer
from lib.distributed import multi_proc_run
from torch.distributed import init_process_group, destroy_process_group
import lib.distributed as du
import multiprocessing as mp

def single_proc_run(config):
  if not torch.cuda.is_available():
    raise Exception('No GPUs FOUND.')
  trainer = DetectionTrainer(config)
  if config.net.is_train:
    trainer.train()
  else:
    trainer.test()

def multi_proc_run(rank, world_size, config):
  du.ddp_setup(rank, world_size)
  if not torch.cuda.is_available():
    raise Exception('No GPUs FOUND.')
  trainer = DetectionTrainer(config)
  if config.net.is_train:
    trainer.train()
  else:
    trainer.test()
  destroy_process_group()
  
  # from lib.ddp_data_loaders import make_data_loader
  # train_loader = make_data_loader(
  #     config,
  #     int(config.trainer.batch_size / config.misc.num_gpus),
  #     num_threads=int(config.misc.train_num_thread / config.misc.num_gpus))
  # Trainer = get_trainer(config.trainer.trainer)
  # trainer = Trainer(config=config, data_loader=train_loader, rank=rank)
  # if config.misc.is_train:
  #   trainer.train()
  # else:
  #   trainer.test()



@hydra.main(config_path='config', config_name='default.yaml')
def main(config):
  # fix seed
  # np.random.seed(config.misc.seed)
  # torch.manual_seed(config.misc.seed)
  # torch.cuda.manual_seed(config.misc.seed)
  # port = random.randint(10001, 20001)

  print("=======================\n\n\n")
  print("config: ", config)
  print("=======================\n\n\n")

  # # Convert to dict
  if config.misc.num_gpus > 1:
      # mpu.multi_proc_run(config.misc.num_gpus,
      #         fun=single_proc_run, fun_args=(config,))
      # world_size = torch.cuda.device_count()
      print("config.misc.num_gpus: ", config.misc.num_gpus)
      mp.spawn(multi_proc_run, args=(config.misc.num_gpus, config), nprocs=config.misc.num_gpus)
  else:
      single_proc_run(config)

  # if config.misc.num_gpus > 1:
  #     multi_proc_run(config.misc.num_gpus, port, fun=single_proc_run, fun_args=(config,))
  # else:
  #     single_proc_run(config)
   
if __name__ == '__main__':
  __spec__ = None
  os.environ['MKL_THREADING_LAYER'] = 'GNU'
  main()
