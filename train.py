import os
import argparse

from models.MNIST_GPT import TransformerModel, MNIST_GPT_tokenizer
import torch
from torch.nn.utils import clip_grad_norm_
import torchvision
from torchvision.transforms import v2

from torch.utils.data import Dataset, DataLoader

import torch.optim as optim
from tqdm import tqdm

from uitl import save_checkpoint, load_checkpoint, gradient_norm, prep_generated_img
from accelerate import Accelerator

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoch", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler_lr_warmup_steps", type=float, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--log_to_wandb", type=bool, default=True)
    parser.add_argument("--wandb_api_key", type=str)
    parser.add_argument("--wandb_logs_per_epoch", type=int, default=2)
    # parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--transformer_num_layers", type=int, default=4)
    parser.add_argument("--transformer_model_dim", type=int, default=128)
    parser.add_argument("--transformer_num_heads", type=int, default=4)
    parser.add_argument("--transformer_ff_dim", type=int, default=512)
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

def get_model_and_tokenizer(args):
    num_img_toks = 256
    special_toks = [i for i in range(10)] # num cls in MNIST
    max_seq_len = 28 * 28

    tokenizer = MNIST_GPT_tokenizer(num_img_toks, special_toks)
    model = TransformerModel(
        model_in_vocab_size=tokenizer.model_in_vocab_size,
        model_out_vocab_size=tokenizer.model_out_vocab_size,
        num_layers=args.transformer_num_layers,
        max_seq_len=max_seq_len,
        model_dim=args.transformer_model_dim,
        num_heads=args.transformer_num_heads,
        ff_dim=args.transformer_ff_dim,
        dropout=args.transformer_dropout
    )
    return model, tokenizer

def get_tarining_config(args):
    config = {
        "epochs": args.num_epoch,
        "lr": args.lr,
        "scheduler_lr_warmup_steps" : args.scheduler_lr_warmup_steps,
        "batch_size": args.batch_size,
    }
    return config

# TODO: set num_workers
def prepare_dataset(args, tokenizer):
    ds_path = "MNIST/"
    transforms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.long)])
    mnist_train = torchvision.datasets.MNIST(ds_path, train=True, download=True, transform=transforms)

    
    class MNIST_GPT_Dataset(Dataset):
        def __init__(self, mnist_dataset):
            x = mnist_dataset.data
            y = mnist_dataset.targets
            self.data = tokenizer.encode(x.flatten(start_dim=1), y.tolist())
        
        def __getitem__(self, index):
            return self.data[index]
        
        def __len__(self):
            return self.data.shape[0]
        
    ds = MNIST_GPT_Dataset(mnist_train)

    train_dataloader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        # num_workers=...
    )
    return train_dataloader

# TODO: add AMP and GradScaler
def train_step(accelerator, tokenizer, model, optimizer, batch, loss_fn):
    optimizer.zero_grad(set_to_none=True)
    
    model_in = batch[:,:-1 ]
    model_out = model(model_in)
    model_out = model_out.reshape((-1, tokenizer.model_out_vocab_size))

    target = batch[:,1:].flatten()
    loss = loss_fn(model_out, target)
    accelerator.backward(loss)

    clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    loss_val = loss.to('cpu').item()

    return loss_val

# TODO: add train & test dataloaders
# TODO: add Scheduler
# TODO: add GardNorm
def train_epoch(accelerator, epoch, tokenizer, model, optimizer, dataloader, loss_fn, args):
    t = tqdm(dataloader)
    t.set_description(f"Epoch: {epoch: >3}/{args.num_epoch}")
    
    for n, batch in enumerate(t):
        loss_val = train_step(accelerator, tokenizer, model, optimizer, batch, loss_fn)        
        t.set_postfix_str(f"Loss: {loss_val:.4}")

def train(accelerator, args):

    model, tokenizer = get_model_and_tokenizer(args)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    epoch = 0
    if args.continue_training:
        checkpoint = load_checkpoint(args.cpkt_load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

    dataloader = prepare_dataset(args, tokenizer)
    
    cls_weights = torch.ones((256))
    cls_weights[0] = .1
    loss_fn = torch.nn.CrossEntropyLoss(cls_weights).to(accelerator.device)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    while epoch < args.num_epoch:
        train_epoch(accelerator, epoch, tokenizer, model, optimizer, dataloader, loss_fn, args)
        if epoch != 0 and epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, args.cpkt_save_path)
        epoch += 1

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    accelerator = Accelerator()
    train(accelerator, args)