import torch
from torchvision.utils import make_grid
import torchvision.transforms as tvf
import numpy as np

def save_checkpoint(accelerator, model, optimizer, epoch, path="model.cpkt"):
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    cpkt = {
        "epoch" : epoch,
        "model_state_dict" : unwrapped_model.state_dict(),
        "optimizer_state_dict" : optimizer.state_dict()
    }
    accelerator.save(cpkt, path)
    if accelerator.is_local_main_process:
        print(f"Epoch {epoch} | Training checkpoint saved at {path}")

def load_checkpoint(path="model.cpkt"):
    checkpoint = torch.load(path)
    return checkpoint

def gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def prep_generated_img(model_out):
    bsz = model_out.shape[0]
    model_out = model_out.to('cpu').view(bsz, 1, 28, 28)
    img_grid = make_grid(model_out, nrow=8).to(torch.uint8)
    img = tvf.functional.to_pil_image(img_grid)
    return np.asarray(img)
