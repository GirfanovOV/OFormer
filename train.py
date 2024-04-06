import os
import argparse

from models.MNIST_GPT import TransformerModel
import torch
from torch.nn.utils import clip_grad_norm_
import torchvision
from torchvision.utils import make_grid
import torchvision.transforms as tvf
from torchvision.transforms import v2

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

import wandb
from uitl import save_checkpoint, load_checkpoint, gradient_norm, prep_generated_img

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoch", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_warmup_steps", type=float, default=100)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--log_to_wandb", type=bool, default=True)
    parser.add_argument("--wandb_api_key", type=str)
    parser.add_argument("--wandb_logs_per_epoch", type=int, default=2)
    # parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--transformer_num_layers", type=int, default=9)
    parser.add_argument("--transformer_model_dim", type=int, default=512)
    parser.add_argument("--transformer_num_heads", type=int, default=8)
    parser.add_argument("--transformer_ff_dim", type=int, default=2048)
    parser.add_argument("--transformer_dropout", type=float, default=.1)
    parser.add_argument("--continue_training", type=bool, default=False)
    parser.add_argument("--cpkt_load_path", type=str, default="model.cpkt")
    parser.add_argument("--cpkt_save_path", type=str, default="model.cpkt")
    parser.add_argument("--save_every", type=int, default=10)
    return parser

def check_args(args):
    if args.log_to_wandb and args.wandb_api_key is None:
        return False
    return True

def get_model(args):
    num_img_toks = 256
    special_toks = [i for i in range(10)] # num cls in MNIST
    max_seq_len = 28 * 28

    model = TransformerModel(
        num_img_toks=num_img_toks,
        special_toks=special_toks,
        num_layers=args.transformer_num_layers,
        max_seq_len=max_seq_len,
        model_dim=args.transformer_model_dim,
        num_heads=args.transformer_num_heads,
        ff_dim=args.transformer_ff_dim,
        dropout=args.transformer_dropout
    )
    return model

def get_tarining_config(args):
    config = {
        "epochs": args.num_epoch,
        "lr": args.lr,
        "lr_warmup_steps" : args.lr_warmup_steps,
        "batch_size": args.batch_size,
    }
    return config

# def 
# run = wandb.init(
#     project="Simple ViT",
#     notes="Softmax Attention based vision transformer",
#     tags=["GPT", "MNIST", "RoPE"],
#     config=config
# )

def prepare_dataset(args):
    ds_path = "MNIST/"
    transforms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.long)])
    mnist_train = torchvision.datasets.MNIST(ds_path, train=True, download=True, transform=transforms)

    train_dataloader = DataLoader(
        mnist_train,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        sampler=DistributedSampler(mnist_train)
    )
    return train_dataloader

def setup(rank, world_size):
    "Sets up the process group and configuration for PyTorch Distributed Data Parallelism"
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    "Cleans up the distributed environment"
    dist.destroy_process_group()

# TODO: add AMP and GradScaler
def train_step(rank, model, optimizer, batch, loss_fn):
    optimizer.zero_grad(set_to_none=True)

    x = batch[0]
    y = batch[1]

    x = x.to(rank).flatten(start_dim=1)
    y = model.encode_cls(y.tolist()).to(rank)

    model_in = x[:,:-1]
    model_out(model_in, y)
    model_out = model_out.reshape((-1, model.model_out_vocab_size))

    target = x
    target = target.flatten()

    loss = loss_fn(model_out, target)
    loss.backward()

    clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    loss_val = loss.to('cpu').item()

    return loss_val

# TODO: add train & test dataloaders
# TODO: add Scheduler
# TODO: add GardNorm
def train_epoch(rank, epoch, model, optimizer, dataloader, loss_fn, args):
    if args.log_to_wandb:
        log_every = len(dataloader) // args.wandb_log_every
    t = tqdm(dataloader, total=len(dataloader)) if rank == 0 else dataloader
    
    if rank == 0:
        t.set_description(f"Epoch: {epoch: >3}/{args.num_epoch}")
    
    for n, batch in t:
        loss_val = train_step(rank, model, optimizer, batch, loss_fn)
        
        if rank == 0:
            t.set_postfix_str(f"Loss: {loss_val:.4}")
        
        if rank == 0 and args.log_to_wandb and n != 0 and n % log_every == 0:
            log = {
                "loss" : loss_val,
                # "lr" : scheduler.get_last_lr()[0],
                # "grad_norm" : grad_norm,
                "epoch" : epoch
            }
            wandb.log(log)

    if rank == 0 and args.log_to_wandb:
        model_out = model.generate(temperature=.8)
        prepd_img = prep_generated_img(model_out)
        log_img = wandb.Image(prepd_img, caption="Generated images")
        log = {'generated' :log_img, "epoch" : epoch}
        wandb.log(log)

def train(rank, world_size, args):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    model = get_model(args)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr_warmup_steps)

    epoch = 0
    if args.continue_training:
        checkpoint = load_checkpoint(args.cpkt_load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

    ddp_model = DDP(model, device_ids=[rank])
    dataloader = prepare_dataset(args)
    
    cls_weights = torch.ones((256))
    cls_weights[0] = .1
    loss_fn = torch.nn.CrossEntropyLoss(cls_weights).to(rank)

    while epoch < args.num_epoch:
        train_epoch(rank, epoch, ddp_model, optimizer, dataloader, loss_fn, args)
        if rank == 0 and epoch != 0 and epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, args.cpkt_save_path)
        epoch += 1

    cleanup()

if "__name__" == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, args), nprocs=world_size)