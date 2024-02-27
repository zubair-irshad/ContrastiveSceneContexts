# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os
import json
import logging
import torch
from omegaconf import OmegaConf

from easydict import EasyDict as edict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
# import lib.multiprocessing_utils as mpu

import multiprocessing as mp
import hydra

from lib.ddp_trainer import PointNCELossTrainer, PartitionPointNCELossTrainer, PartitionPointNCELossTrainerPointNet

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "123568"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3,4,5,6,7"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])

torch.manual_seed(0)
torch.cuda.manual_seed(0)

logging.basicConfig(level=logging.INFO, format="")

def get_trainer(trainer):
  if trainer == 'PointNCELossTrainer':
    return PointNCELossTrainer
  elif trainer == 'PartitionPointNCELossTrainer':
    return PartitionPointNCELossTrainer
  elif trainer == 'PartitionPointNCELossTrainerPointNet':
    return PartitionPointNCELossTrainerPointNet
  else:
    raise ValueError(f'Trainer {trainer} not found')

@hydra.main(config_path='config', config_name='defaults.yaml')
def main(config):

  print("=======================\n\n\n")
  print("config: ", config)
  print("=======================\n\n\n")
  if os.path.exists('config.yaml'):
    logging.info('===> Loading exsiting config file')
    config = OmegaConf.load('config.yaml')
    logging.info('===> Loaded exsiting config file')
  logging.info('===> Configurations')
  # logging.info(config.pretty())
  logging.info(config)

  if config.misc.local_rank == 0:
      wandb.init(
          project="scene_context"
      )

  # # Convert to dict
  if config.misc.num_gpus > 1:
      # mpu.multi_proc_run(config.misc.num_gpus,
      #         fun=single_proc_run, fun_args=(config,))
      # world_size = torch.cuda.device_count()

      print("config.misc.num_gpus: ", config.misc.num_gpus)
      mp.spawn(multi_proc_run, args=(config.misc.num_gpus, config), nprocs=config.misc.num_gpus)
  else:
      single_proc_run(config)
  
  single_proc_run(config)

# def single_proc_run(config):
#     from lib.ddp_data_loaders import make_data_loader

#     # Initialize distributed training environment if multiple GPUs are available
#     if config.misc.num_gpus > 1:
#         # torch.cuda.set_device(config.misc.local_rank)
#         os.environ['MASTER_ADDR'] = 'localhost'
#         os.environ['MASTER_PORT'] = '12355'
#         os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3,4,5,6,7"
#         dist.init_process_group(backend='nccl', init_method='env://', rank=torch.cuda.device_count(),  world_size = 1)

#     train_loader = make_data_loader(
#         config,
#         int(config.trainer.batch_size / config.misc.num_gpus),
#         num_threads=int(config.misc.train_num_thread / config.misc.num_gpus)
#     )

#     Trainer = get_trainer(config.trainer.trainer)

#     if config.misc.num_gpus > 1:
#         model = Trainer.build_model()
#         model = DDP(model, device_ids=[config.misc.local_rank], output_device=config.misc.local_rank)
#         trainer = Trainer(config=config, model=model, data_loader=train_loader)
#     else:
#         trainer = Trainer(config=config, data_loader=train_loader)

#     if config.misc.is_train:
#         trainer.train()
#     else:
#         trainer.test()

#     # Cleanup
#     if config.misc.num_gpus > 1:
#         dist.destroy_process_group()
  
def multi_proc_run(rank, world_size, config):

  ddp_setup(rank, world_size)
  
  from lib.ddp_data_loaders import make_data_loader

  train_loader = make_data_loader(
      config,
      int(config.trainer.batch_size / config.misc.num_gpus),
      num_threads=int(config.misc.train_num_thread / config.misc.num_gpus))

  Trainer = get_trainer(config.trainer.trainer)
  trainer = Trainer(config=config, data_loader=train_loader)

  if config.misc.is_train:
    trainer.train()
  else:
    trainer.test()

  destroy_process_group()

  
def single_proc_run(config):
  
  from lib.ddp_data_loaders import make_data_loader

  train_loader = make_data_loader(
      config,
      int(config.trainer.batch_size / config.misc.num_gpus),
      num_threads=int(config.misc.train_num_thread / config.misc.num_gpus))

  Trainer = get_trainer(config.trainer.trainer)
  trainer = Trainer(config=config, data_loader=train_loader)

  if config.misc.is_train:
    trainer.train()
  else:
    trainer.test()


if __name__ == "__main__":
  print("HEREE")
  os.environ['MKL_THREADING_LAYER'] = 'GNU'
  main()
