import os
import io
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import webdataset as wds

from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torchvision import transforms
from PIL import Image

import logging
import wandb
import json

from torch import nn

from ema_pytorch import EMA

# Import the translator model
from lat2lat.models.translator import LatentTranslator
from lat2lat.data.pairs_s3 import get_loader


class LatentTranslatorTrainer(LightningModule):
    def __init__(self,
                 input_channels: int = 128,
                 output_channels: int = 16,
                 input_size: int = 4,
                 output_size: int = 64,
                 lr_adamw: float = 0.0002,
                 b1: float = 0.5,
                 b2: float = 0.999,
                 weight_decay: float = 0,
                 batch_size: int = 64, 
                 num_workers: int = 0,
                 log_every_n_steps: int = 100,
                 train_url: Optional[str] = None,
                 cosine_scheduler: bool = False,
                 max_steps: int = -1,
                 datapoints_per_epoch: Optional[int] = None,
                 max_epochs: Optional[int] = None,
                 gradient_clip_val: float = 0.0,
                 use_ema: bool = False,
                 ema_decay: float = 0.999,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.input_size = input_size
        self.output_size = output_size
        
        self.train_url = train_url
        
        self.cosine_scheduler = cosine_scheduler
        self.datapoints_per_epoch = datapoints_per_epoch
        self.max_steps = max_steps
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        if self.datapoints_per_epoch is not None and self.max_epochs is not None:
            self.max_steps = self.max_epochs * self.datapoints_per_epoch // self.batch_size 
        self.lr_adamw = lr_adamw
        self.b1 = b1
        self.b2 = b2
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.log_every_n_steps = log_every_n_steps
        self.scheduler = None
        self.optimizer = None
        self.automatic_optimization = True
        self.gradient_clip_val = gradient_clip_val
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        
        # Build the translator model
        self.model = LatentTranslator(
            input_channels=input_channels,
            output_channels=output_channels,
            input_size=input_size,
            output_size=output_size
        )
        self.model.to(torch.bfloat16)
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.compile()
        
        # Initialize EMA if enabled
        if self.use_ema:
            self.ema = EMA(
                    self.model,
                    beta = self.ema_decay,              # exponential moving average factor
                    update_after_step = 1000,    # only after this number of .update() calls will it start updating
                    update_every = 10,          # how often to actually update, to save on compute (updates every 10th .update() call)
                )

    def forward(self, x):
        # Input: [B, 128, 4, 4] -> Output: [B, 16, 64, 64]
        return self.model.translate_batch(x)

    def training_step(self, batch, batch_idx):            
        wan, dcae = batch
        # Forward pass
        predicted_wan = self(dcae)

        # Calculate loss (MSE)
        loss = F.mse_loss(predicted_wan, wan)
        
        # Log metrics
        self.log('epoch', float(self.current_epoch), on_step=True, on_epoch=True)
        self.log('loss', loss.item(), on_step=True, on_epoch=True)
        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            self.log('lr-adamw', optimizer[0].param_groups[0]['lr'], on_step=True, on_epoch=True)
        else:
            self.log('lr-adamw', optimizer.param_groups[0]['lr'], on_step=True, on_epoch=True)
        
        # Update EMA after optimizer step if enabled
        if self.use_ema:
            self.ema.update()
        
        return loss
    
     
    def configure_optimizers(self):
        lr_adamw = self.lr_adamw
        b1 = self.b1
        b2 = self.b2
        weight_decay = self.weight_decay

        self.adamw = torch.optim.AdamW(self.model.parameters(), lr=lr_adamw, betas=(b1, b2), weight_decay=weight_decay)
        
        if self.cosine_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.adamw, T_max=self.max_steps)
            return {
                "optimizer": self.adamw,
                "lr_scheduler": {
                    "scheduler": self.scheduler,
                    "interval": "step"
                }
            }
        else:
            return [self.adamw]

    def train_dataloader(self):
        return get_loader(self.batch_size, deterministic=False, prefetch_size=100, url=self.train_url)
    

def main(args: Namespace) -> None:
    os.environ['WANDB_DIR'] = args.wandb_dir
    torch.set_float32_matmul_precision('high')
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = LatentTranslatorTrainer(**vars(args))

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    # If use distubuted training  PyTorch recommends to use DistributedDataParallel.
    # See: https://pytorch.org/docs/stable/nn.html#torch.nn.DataParallel
    
    if args.devices is not None:
        trainer = Trainer(max_epochs=args.max_epochs, max_steps=args.max_steps, accelerator=args.accelerator, devices=args.devices, 
                        strategy=args.strategy, log_every_n_steps=args.log_every_n_steps)
    else:
        trainer = Trainer(max_epochs=args.max_epochs, max_steps=args.max_steps, accelerator=args.accelerator, strategy=args.strategy, 
                          log_every_n_steps=args.log_every_n_steps)

    if trainer.is_global_zero:
        # Wandb logging
        os.makedirs(args.checkpoint_path, exist_ok=True)
        wandb_logger = WandbLogger(project=args.wandb_project, 
                                log_model=False, 
                                save_dir=args.checkpoint_path,
                                config=vars(args),
                                name=args.experiment_name)
        run_name = wandb_logger.experiment.name

        # Watch model weights with wandb
        wandb.watch(model.model, log="all", log_freq=args.log_every_n_steps)

        # Configure the ModelCheckpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(args.checkpoint_path, run_name),  # Define the path where checkpoints will be saved
            save_top_k=args.checkpoint_top_k,  # Set to -1 to save all epochs
            verbose=True,  # If you want to see a message for each checkpoint
            every_n_train_steps=args.checkpoint_every_n_examples//(args.batch_size * trainer.world_size)+1,
            monitor='loss',
        )
        lr_monitor = LearningRateMonitor(logging_interval='step')
        trainer.callbacks.append(checkpoint_callback)
        trainer.callbacks.append(lr_monitor)
        trainer.logger = wandb_logger


    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    # model parameters
    parser.add_argument("--input_channels", type=int, default=128, help="number of input channels")
    parser.add_argument("--output_channels", type=int, default=16, help="number of output channels")
    parser.add_argument("--input_size", type=int, default=4, help="input size")
    parser.add_argument("--output_size", type=int, default=64, help="output size")
    
    # data parameters
    parser.add_argument("--train_url", type=str, default='s3://cod-yt-latent-pairs/pairs/train2')
    
    # training parameters
    parser.add_argument("--gradient_clip_val", type=float, default=0.0)
    parser.add_argument("--cosine_scheduler", default=False, action="store_true")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=10, help="number of epochs of training")
    parser.add_argument("--max_steps", type=int, default=-1, help="number of steps of training")
    parser.add_argument("--lr_adamw", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="weight decay for optimizer")
    parser.add_argument("--datapoints_per_epoch", type=int, default=None, help="number of data points per epoch")
    parser.add_argument("--use_ema", default=False, action="store_true", help="use exponential moving average")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="decay rate for exponential moving average")
    
    # checkpoint and logging parameters
    parser.add_argument("--checkpoint_path", type=str, default="models/latent_translator")
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--checkpoint_every_n_examples", type=int, default=100000)
    parser.add_argument("--checkpoint_top_k", type=int, default=5)
    
    # hardware parameters
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument("--strategy", type=str, default="ddp", help="ddp, ddp2, ddp_spawn, etc.")
    parser.add_argument("--accelerator", type=str, default="auto", help="auto, gpu, tpu, mpu, cpu, etc.")
    
    # logging parameters
    parser.add_argument("--wandb_project", type=str, default="latent_translator")
    parser.add_argument("--wandb_dir", type=str, default=".")
    parser.add_argument("--experiment_name", type=str, default=None, help="optional name for the experiment")
    
    hparams = parser.parse_args()

    main(hparams) 