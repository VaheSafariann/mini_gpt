from dataclasses import dataclass
import os

BASE_DIR = os.path.dirname(
    os.path.dirname(
    os.path.abspath(__file__)))
@dataclass
class Config:

    vocab_size:     int = 8192
    embed_dim:      int = 192
    n_heads:        int = 6
    n_layers:       int = 6
    context_len:    int = 128


    batch_size:     int = 16
    learning_rate:  float = 3e-4
    dropout:        float = 0.1
    weight_decay:   float = 1e-5
    eval_interval:  int = 100
    max_steps:      int = 5000

    data_dir:       str = os.path.join(BASE_DIR, 'data')
    checkpoint_dir: str = os.path.join(BASE_DIR, 'checkpoints')

'''
vocab_size = 8192 → сколько разных токенов существует вообще
context_len = 128 → сколько токенов в одном предложении максимум
👉 Пример:
"hello bro how are you"
после токенизации → например:
[345, 812, 91, 27, 600]
каждое число ∈ [0, 8191]
длина = 5 (<= 128)
'''