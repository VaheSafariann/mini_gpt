import torch
from tokenizers import ByteLevelBPETokenizer
import os
from mini_gpt.model_configs.config import Config
from mini_gpt.src.model import MiniGPT

def generate(prompt,max_new_tokens = 200,temperature=0.7):
    cfg = Config()
    device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))
    vocab_path = os.path.join(
        base_dir, "data","tokenizer","vocab.json")
    merges_path = os.path.join(
        base_dir, "data","tokenizer","merges.txt"
    )
    tokenizer = ByteLevelBPETokenizer.from_file(
        vocab_path, merges_path)
    model = MiniGPT(cfg).to(device)
    checkpoint_path = os.path.join(
        base_dir, "mini_gpt_4600.pt")
    checkpoint = torch.load(
        checkpoint_path,map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"сhekpoint is loaded from {checkpoint_path}")
    tokens = tokenizer.encode(prompt).ids
    x = torch.tensor(
        tokens,
        dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if x.size(1) > cfg.context_len:
                x = x[:,-cfg.context_len:]

            logits = model(x)
            logits = logits[:,-1,:]
            logits = logits/temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            x = torch.cat([x, next_token],dim = 1)
    output_ids = x[0].tolist()
    full_text = tokenizer.decode(output_ids)
    return full_text

if __name__ == "__main__":
    prompt = "Once upon a time, there was a little boy"
    generated_story = generate(prompt,max_new_tokens = 100,temperature=0.7)
    print(f"Generated story: {generated_story}")






