#!/usr/bin/env python3
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import argparse
import datetime
import json
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress

import torch
import torchvision
import yaml
from timm.data import (AugMixDataset, FastCollateMixup, Mixup, create_dataset,
                       create_loader, resolve_data_config)
from timm.loss import *
from timm.models import (convert_splitbn_model, create_model, load_checkpoint,
                         model_parameters, resume_checkpoint, safe_model_name)
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import *
from timm.utils import ApexScaler, NativeScaler
from torch import nn
from torch.nn.parallel import DistributedDataParallel as NativeDDP

try:
    from mmcv.runner import load_checkpoint as load_ckpt
except:
    from mmengine.runner import load_checkpoint as load_ckpt

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

import models

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')


def get_args_parser():
    # The first arg parser parses out only the --config argument, this argument is used to
    # load a yaml file containing key-values that override the defaults for the main parser below
    # config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    # parser.add_argument('-c', '--config', default='', type=str, metavar='FILE', help='YAML config file specifying default arguments')
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training', add_help=False)

    # Dataset parameters
    parser.add_argument('--data-dir', metavar='DIR', help='path to dataset')
    parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                        help='dataset type (default: ImageFolder/ImageTar if empty)')
    parser.add_argument('--train-split', metavar='NAME', default='train',
                        help='dataset train split (default: train)')
    parser.add_argument('--val-split', metavar='NAME', default='validation',
                        help='dataset validation split (default: validation)')
    parser.add_argument('--dataset-download', action='store_true', default=False,
                        help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
    parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                        help='path to class to idx mapping file (default: "")')

    # Model parameters
    parser.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                        help='Name of model to train (default: "resnet50"')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='Start with pretrained version of specified network (if avail)')
    parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                        help='Initialize model from this checkpoint (default: none)')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='Resume full model and optimizer state from checkpoint (default: none)')
    parser.add_argument('--finetune', default=None, type=str, metavar='PATH',
                        help='Fine-tune model from this checkpoint (default: none)')
    parser.add_argument('--no-resume-opt', action='store_true', default=False,
                        help='prevent resume of optimizer state when resuming model')
    parser.add_argument('--num-classes', type=int, default=1000, metavar='N',
                        help='number of label classes (Model default if None)')
    parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                        help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
    parser.add_argument('--img-size', type=int, default=None, metavar='N',
                        help='Image patch size (default: None => model default)')
    parser.add_argument('--input-size', default=None, nargs=3, type=int,
                        metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
    parser.add_argument('--crop-pct', default=None, type=float,
                        metavar='N', help='Input image center crop percent (for validation only)')
    parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                        help='Override mean pixel value of dataset')
    parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                        help='Override std deviation of of dataset')
    parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                        help='Image resize interpolation type (overrides model)')
    parser.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                        help='validation batch size override (default: None)')
    parser.add_argument('--grad-checkpoint', action='store_true', default=False,
                        help='Using gradient checkpointing for saving GPU memory')
    parser.add_argument('--ckpt-stg', default=[0, 0, 0, 0], type=int, nargs='+',
                        help='stage for using grad checkpoint')
    parser.add_argument('--val-freq', default=1, type=int,
                        help='do evaluation per n epoch')
    parser.add_argument('--val-start-epoch', default=0, type=int,
                        help='do evaluation per epoch after n-th epoch')
    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: None, use opt default)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Optimizer momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip-mode', type=str, default='norm',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')
    parser.add_argument('--aux-loss-ratio', type=float, default=0.4,
                        help='Aux loss weight')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "step"')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--auto-lr', action='store_true',
                        default=False, help='auto scaling learning rate')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                        help='learning rate cycle len multiplier (default: 1.0)')
    parser.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                        help='amount to decay each learning rate cycle (default: 0.5)')
    parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                        help='learning rate cycle limit, cycles enabled if > 1')
    parser.add_argument('--lr-k-decay', type=float, default=1.0,
                        help='learning rate k-decay for cosine/poly (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                        help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
    parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--decay-epochs', type=float, default=100, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation & regularization parameters
    parser.add_argument('--no-aug', action='store_true', default=False,
                        help='Disable all training augmentation, override other train aug args')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--hflip', type=float, default=0.5,
                        help='Horizontal flip training aug probability')
    parser.add_argument('--vflip', type=float, default=0.,
                        help='Vertical flip training aug probability')
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default="rand-m9-mstd0.5-inc1", metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". (default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--aug-repeats', type=int, default=0,
                        help='Number of augmentation repetitions (distributed training only) (default: 0)')
    parser.add_argument('--aug-splits', type=int, default=0,
                        help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
    parser.add_argument('--jsd-loss', action='store_true', default=False,
                        help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
    parser.add_argument('--bce-loss', action='store_true', default=False,
                        help='Enable BCE loss w/ Mixup/CutMix use.')
    parser.add_argument('--bce-target-thresh', type=float, default=None,
                        help='Threshold for binarizing softened BCE targets (default: None, disabled)')
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                        help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='random',
                        help='Training interpolation (random, bilinear, bicubic default: "random")')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                        help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
    parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                        help='Drop path rate (default: None)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')

    # Batch norm parameters (only works with gen_efficientnet based models currently)
    parser.add_argument('--bn-tf', action='store_true', default=False,
                        help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
    parser.add_argument('--bn-momentum', type=float, default=None,
                        help='BatchNorm momentum override (if not None)')
    parser.add_argument('--bn-eps', type=float, default=None,
                        help='BatchNorm epsilon override (if not None)')
    parser.add_argument('--no-sync-bn', action='store_true', default=False,
                        help='Disable NVIDIA Apex or Torch synchronized BatchNorm.')
    parser.add_argument('--dist-bn', type=str, default='reduce',
                        help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
    parser.add_argument('--split-bn', action='store_true',
                        help='Enable separate BN layers per augmentation split.')

    # Model Exponential Moving Average
    parser.add_argument('--model-ema', action='store_true', default=False,
                        help='Enable tracking moving average of model weights')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                        help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
    parser.add_argument('--model-ema-decay', type=float, default=0.99984,
                        help='decay factor for model weights moving average (default: 0.99996)')

    # Misc
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--worker-seeding', type=str, default='all',
                        help='worker seed mode (default: all)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                        help='how many batches to wait before writing recovery checkpoint')
    parser.add_argument('--checkpoint-hist', type=int, default=5, metavar='N',
                        help='number of checkpoints to keep')
    parser.add_argument('--checkpoint-freq', type=int, default=50, metavar='N',
                        help='freq of saving checkpoints')
    parser.add_argument('-j', '--workers', type=int, default=8, metavar='N',
                        help='how many training processes to use (default: 8)')
    parser.add_argument('--save-images', action='store_true', default=False,
                        help='save images of input bathes every log interval for debugging')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
    parser.add_argument('--apex-amp', action='store_true', default=False,
                        help='Use NVIDIA Apex AMP mixed precision')
    parser.add_argument('--native-amp', action='store_true', default=False,
                        help='Use Native Torch AMP mixed precision')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='Use torch.compile to accelerate training')
    parser.add_argument('--debug-loss', action='store_true', default=False,
                        help='Use Anomaly Detection')
    parser.add_argument('--no-ddp-bb', action='store_true', default=False,
                        help='Force broadcast buffers for native DDP to off.')
    parser.add_argument('--channels-last', action='store_true', default=False,
                        help='Use channels_last memory layout')
    parser.add_argument('--no-pin-mem', action='store_true', default=False,
                        help='Disable Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-prefetcher', action='store_true', default=False,
                        help='disable fast prefetcher')
    parser.add_argument('--output', default='', type=str, metavar='PATH',
                        help='path to output folder (default: none, current dir)')
    parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                        help='name of train experiment, name of sub-folder for output')
    parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                        help='Best metric (default: "top1"')
    parser.add_argument('--tta', type=int, default=0, metavar='N',
                        help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
    # parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                        help='use the multi-epochs-loader to save time at the beginning of every epoch')
    parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                        help='convert model torchscript for inference')
    parser.add_argument('--log-wandb', action='store_true', default=False,
                        help='log training and validation metrics to wandb')
    parser.add_argument('--gpu-limit', default=1, type=float)
    # parser.add_argument('--local-rank', default=0, type=int)
    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    return parser


def main(args):

    # Cache the args as a text string to save them in the output dir later
    args.device_name = torch.cuda.get_device_name()
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    setup_default_logging()
    # args, args_text = _parse_args()
    if args.local_rank == 0:
        _logger.info(args_text)
    if args.log_wandb:
        if has_wandb:
            wandb.init(project=args.experiment, config=args)
        else:
            _logger.warning("You've requested to log metrics to wandb but package not found. "
                            "Metrics not being logged to wandb, try `pip install wandb`")

    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    # args.device = 'cuda:0'
    # args.world_size = 1
    # args.rank = 0  # global rank
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.distributed = True
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        # args.device = 'cuda:%d' % args.local_rank
        
    elif 'SLURM_PROCID' in os.environ:
        args.distributed = True
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
        
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl', init_method=args.dist_url,
                                             world_size=args.world_size, rank=args.rank)
        # args.world_size = torch.distributed.get_world_size()
        # args.rank = torch.distributed.get_rank()
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        _logger.info('Training with a single process on 1 GPUs.')
    assert args.rank >= 0

    if args.gpu_limit < 1:
        torch.cuda.set_per_process_memory_fraction(args.gpu_limit)
    
    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning("Neither APEX or native PyTorch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")

    random_seed(args.seed, args.rank)

    
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        # img_size=args.input_size,
        use_checkpoint=args.ckpt_stg if args.grad_checkpoint else [0] * 4,
    )
    
    if args.finetune:
        load_ckpt(model, args.finetune)
    
    if args.num_classes is None:
        assert hasattr(
            model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        # FIXME handle model default vs config num_classes more elegantly
        args.num_classes = model.num_classes

    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    if args.local_rank == 0:
        _logger.info(model)
        _logger.info(
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')
        model_str = str(model)
           
    # move model to GPU, enable channels last layout if set
    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and not args.no_sync_bn:
        assert not args.split_bn
        if has_apex and use_amp == 'apex':
            # Apex SyncBN preferred unless native amp is activated
            model = convert_syncbn_model(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.local_rank == 0:
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)

    if args.auto_lr:
        args.lr = (args.batch_size * args.world_size / 1024) * 1e-3
        args_text = yaml.load(args_text, Loader=yaml.FullLoader)
        args_text['lr'] = args.lr
        args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    if args.local_rank == 0:
        _logger.info(f'Initail learning rate: {args.lr}.')
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info(
                'Using native PyTorch AMP. Training in mixed precision.')
    else:
        if args.local_rank == 0:
            _logger.info('AMP not enabled. Training in float32.')

    if args.grad_checkpoint and args.local_rank == 0:
        _logger.info(
            f'Using gradient checkpointing, checkpoint stage: {args.ckpt_stg}.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model, args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0)

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        # model_ema_decay = args.model_ema_decay ** (args.batch_size * args.world_size / 512.0)
        model_ema = ModelEmaV2(model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp == 'apex':
            # Apex DDP preferred unless native amp is activated
            model = ApexDDP(model, delay_allreduce=True)
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
        else:
            model = NativeDDP(model, device_ids=[args.gpu], broadcast_buffers=not args.no_ddp_bb)
            if args.local_rank == 0:                
                _logger.info("Using native PyTorch DistributedDataParallel.")
        # NOTE: EMA model does not need to be wrapped by DDP

    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)


    _logger.info("Creating Dataset ...")
    # create the train and eval datasets
    dataset_train = create_dataset(
        args.dataset, root=args.data_dir, split=args.train_split, is_training=True,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size,
        repeats=args.epoch_repeats)
    dataset_eval = create_dataset(
        args.dataset, root=args.data_dir, split=args.val_split, is_training=False,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size)

    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        if args.prefetcher:
            # collate conflict (need to support deinterleaving in collate mixup)
            assert not num_aug_splits
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
        

    _logger.info("Creating Dataloader ...")
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=not args.no_pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        worker_seeding=args.worker_seeding,
    )

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size or args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=not args.no_pin_mem,
    )

    
    # setup loss function
    if args.jsd_loss:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(
            num_splits=num_aug_splits, smoothing=args.smoothing)
    elif mixup_active:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(
                target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(
                smoothing=args.smoothing, target_threshold=args.bce_target_thresh)
        else:
            # train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
            train_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    if args.model_ema:
        ema_save_tag = 'ema'
        eval_metric_ema = f'{args.eval_metric}_{ema_save_tag}'
    
    best_metric = None
    best_metric_ema = None
    best_epoch = None
    best_epoch_ema = None
    saver = None
    saver_ema = None
    output_dir = None
    resume_mode = args.resume
    
    if args.rank == 0:
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
        output_dir = get_outdir(
            args.output if args.output else './output/train', exp_name)
        decreasing = True if eval_metric == 'loss' else False
        
        checkpoint_dir = f'{output_dir}/checkpoints/'
        os.makedirs(checkpoint_dir, exist_ok=True)
        saver = CheckpointSaver(model=model, optimizer=optimizer, 
                                args=args, model_ema=model_ema, 
                                amp_scaler=loss_scaler, checkpoint_dir=checkpoint_dir, 
                                recovery_dir=output_dir, decreasing=decreasing, 
                                max_history=args.checkpoint_hist)
        
        
        if model_ema is not None:
            checkpoint_dir = f'{output_dir}/checkpoints_ema/'
            os.makedirs(checkpoint_dir, exist_ok=True)
            saver_ema = CheckpointSaver(model=model, optimizer=optimizer, 
                                        args=args, model_ema=model_ema,
                                        checkpoint_prefix='checkpoint-ema',
                                        amp_scaler=loss_scaler, checkpoint_dir=checkpoint_dir, 
                                        recovery_dir=output_dir, decreasing=decreasing, 
                                        max_history=args.checkpoint_hist)

        
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)
        with open(os.path.join(output_dir, 'model.log'), 'w') as f:
            f.write(model_str)

        if args.compile:
            _logger.info("Compiling model ...")
            model = torch.compile(model)
    
        _logger.info("Start Training.")
        _logger.info('Scheduled epochs: {}'.format(num_epochs))
            
    try:        
        for epoch in range(start_epoch, num_epochs):
            if args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)

            train_metrics = train_one_epoch(
                epoch, model, loader_train, 
                optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, 
                saver=saver, output_dir=output_dir,
                amp_autocast=amp_autocast, 
                loss_scaler=loss_scaler, 
                model_ema=model_ema, 
                mixup_fn=mixup_fn,
                eta_meter=AverageMeter())

            if (epoch % args.val_freq == 0) or epoch > args.val_start_epoch or resume_mode:
                resume_mode = False
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    if args.local_rank == 0:
                        _logger.info("Distributing BatchNorm running means and vars")
                    distribute_bn(model, args.world_size,
                                  args.dist_bn == 'reduce')

                eval_metrics = validate(model, loader_eval, validate_loss_fn, 
                                        args, amp_autocast=amp_autocast)

                if model_ema is not None and not args.model_ema_force_cpu:
                    if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                        distribute_bn(model_ema, args.world_size,
                                      args.dist_bn == 'reduce')
                    ema_eval_metrics = validate(model_ema.module, loader_eval,
                                                validate_loss_fn, args,
                                                amp_autocast=amp_autocast,
                                                log_suffix=f'_{ema_save_tag}')
                    eval_metrics.update(ema_eval_metrics)
            else:
                eval_metrics = {key: float('-inf') for key in eval_metrics}

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            if output_dir is not None:
                update_summary(epoch,
                               train_metrics,
                               eval_metrics,
                               os.path.join(output_dir, 'summary.csv'),
                               write_header=best_metric is None,
                               log_wandb=args.log_wandb and has_wandb)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

            if saver_ema is not None:
                # save proper checkpoint with eval metric (ema)
                save_metric = eval_metrics[eval_metric_ema]
                best_metric_ema, best_epoch_ema = saver_ema.save_checkpoint(epoch, metric=save_metric)

            
            if best_metric is not None:
                
                if args.local_rank == 0:
                    _logger.info(f"Currently Best Accuracy: {best_metric:.2f} at Epoch {best_epoch}")
                    if best_metric_ema is not None:
                        _logger.info(f"Currently Best Accuracy (EMA): {best_metric_ema:.2f} at Epoch {best_epoch_ema}")
                    _logger.info('\n')
                    
                with open(os.path.join(output_dir, 'best-metric.json'), 'w') as f:
                    
                    best_metric_info = {
                        eval_metric: best_metric,
                        'epoch': best_epoch,
                    }
                    
                    if best_metric_ema is not None:
                        best_metric_info[eval_metric_ema] = best_metric_ema
                        best_metric_info['epoch_ema'] = best_epoch_ema
                        
                    json.dump(best_metric_info, f, indent=4)

    except KeyboardInterrupt:
        pass
    
    if best_metric is not None:
        # result_info = f'*** Best metric ({eval_metric}): {best_metric} (epoch {best_epoch}) ***, 
        #                 *** Best metric (ema) ({eval_metric_ema}): {best_metric_ema} (epoch {best_epoch_ema}) ***'
        
        best_metric_info = {
            eval_metric: best_metric,
            'epoch': best_epoch,
        }
        
        if best_metric_ema is not None:
            best_metric_info[eval_metric_ema] = best_metric_ema
            best_metric_info['epoch_ema'] = best_epoch_ema        
        
        # _logger.info(f'best_metric: {best_metric_info}')
        print(f'best_metric: {best_metric_info}')
        

def train_one_epoch(epoch, model, loader, optimizer, loss_fn, args,
                    lr_scheduler=None, saver=None, output_dir=None,
                    amp_autocast=suppress, loss_scaler=None,
                    model_ema=None, mixup_fn=None, eta_meter=AverageMeter()):

    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    losses_aux_m = AverageMeter()
    is_ds = False

    model.train()
    # torch.cuda.empty_cache()

    end = time.time()
    end_epoch = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if not args.prefetcher:
            input = input.cuda()
            target = target.cuda()
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        if args.debug_loss:
            detect_anomaly = torch.autograd.detect_anomaly
        else:
            detect_anomaly = suppress
        
        with amp_autocast():
            with detect_anomaly():
               output = model(input)
               if isinstance(output, (tuple, list, dict)):
                   is_ds = True
                   output_main = output['main']
                   output_aux = output['aux']
                   loss_main = loss_fn(output_main, target)
                   loss_aux = loss_fn(output_aux, target)
                   loss = loss_main + args.aux_loss_ratio * loss_aux
               else:
                   loss = loss_fn(output, target) 

        if not args.distributed:
            
            if is_ds:
                losses_m.update(loss_main.item(), input.size(0))
                losses_aux_m.update(loss_aux.item(), input.size(0))
            else:
                losses_m.update(loss.item(), input.size(0))
            
        optimizer.zero_grad()
        
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=model_parameters(
                    model, exclude_head='agc' in args.clip_mode),
                create_graph=second_order)
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(
                        model, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode)
            optimizer.step()
        
        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        
        if eta_meter is not None:
            eta_meter.update(time.time() - end)
            
        if last_batch or batch_idx % args.log_interval == 0:
            
            if args.distributed:
                
                if is_ds:
                    reduced_loss = reduce_tensor(loss_main.data, args.world_size)
                    reduced_loss_aux = reduce_tensor(loss_aux.data, args.world_size)
                    losses_m.update(reduced_loss.item(), input.size(0))
                    losses_aux_m.update(reduced_loss_aux.item(), input.size(0))
                else:
                    reduced_loss = reduce_tensor(loss.data, args.world_size)
                    losses_m.update(reduced_loss.item(), input.size(0))
                
                
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            
            eta_remaining_batches = (args.epochs + args.cooldown_epochs - epoch - 1) * len(loader) + len(loader) - batch_idx - 1
            batch_time_avg = eta_meter.avg if eta_meter is not None else batch_time_m.avg
            eta_remaining_time = eta_remaining_batches * batch_time_avg
            eta_td = datetime.timedelta(seconds=eta_remaining_time)
            eta_str = str(eta_td)
            current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            gpu_memory = torch.cuda.max_memory_allocated()
            
            if args.local_rank == 0:
                
                base_info = (
                    '[{}]  Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'LR: {lr:.3e}  '
                    'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                )

                aux_info = (
                    'Loss (Aux): {loss_aux.val:#.4g} ({loss_aux.avg:#.3g})  '
                )

                other_info = (
                    'ETA: {eta}  '
                    'Memory: {gpu_memory:d}  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})  '
                )

                if is_ds:
                    base_info += aux_info

                base_info += other_info

                _logger.info(
                    base_info.format(
                        current_time,
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        loss_aux=losses_aux_m if is_ds else None,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m,
                        eta=eta_str.split('.')[0],
                        gpu_memory=int(gpu_memory/(1024**2))
                    )
                )
                
                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)
        
        end = time.time()
        # end for

    epoch_time = time.time() - end_epoch
    
    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()
    
    train_info_dict = {
        'epoch_time(min)': round(epoch_time / 60, 1),
        'lr': lr,
        'loss': losses_m.avg,
    }
    
    if is_ds:
        train_info_dict['loss_aux'] = losses_aux_m.avg
    
    return OrderedDict(train_info_dict)


def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
                if isinstance(output, (tuple, list)):
                    output = output[0]
                loss = loss_fn(output, target)

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(
                    0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))

    metrics = OrderedDict([(f'loss{log_suffix}', losses_m.avg),
                           (f'top1{log_suffix}', top1_m.avg),
                           (f'top5{log_suffix}', top5_m.avg)])

    return metrics


if __name__ == '__main__':
    torch.cuda.empty_cache()
    args = get_args_parser().parse_args()
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    main(args)