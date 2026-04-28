import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import get_cosine_schedule_with_warmup

import os
from mini_gpt.model_configs.config import Config
from mini_gpt.src.model import MiniGPT
from mini_gpt.data.dataset import TinyStoriesDataset

def train():
    cfg = Config()
    base_dir = os.path.dirname(
        os.path.abspath(__file__))
    vocab_path = os.path.join(
        base_dir, "data", "tokenizer", "vocab.json")
    merges_path = os.path.join(
        base_dir,"data","tokenizer", "merges.txt")
    device = torch.device("cuda"
        if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    tokenizer = ByteLevelBPETokenizer.from_file(
        vocab_path, merges_path)
    hf_dataset = load_dataset(
        "roneneldan/TinyStories",split = "train[:200000]")
    dataset = TinyStoriesDataset(
        hf_dataset,
        tokenizer,
        context_len = cfg.context_len,)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        pin_memory = True, shuffle=True)
    model = MiniGPT(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500,
        num_training_steps=cfg.max_steps
    )
    criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    model.train()
    step = 0
    for x,y in dataloader:
        x = x.to(device, non_blocking = True)
        y = y.to(device, non_blocking = True)
        optimizer.zero_grad(set_to_none = True)

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                logits = model(x)
                loss = criterion(
                    logits.reshape(-1,cfg.vocab_size),
                    y.reshape(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(
                logits.reshape(-1, cfg.vocab_size),
                y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
        scheduler.step()

        if step % cfg.eval_interval == 0:
            ppl = torch.exp(loss.detach()).item()
            print(f"step {step} "
                  f"| loss {loss.item():.4f}"
                  f" | ppl {ppl:.4f}")
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,},
            f"mini_gpt_{step}.pt")
        step += 1
        if step >= cfg.max_steps:
            break

if __name__ == "__main__":
    train()





