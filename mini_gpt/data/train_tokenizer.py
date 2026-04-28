from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset
from mini_gpt.model_configs.config import Config
import os

def main():

    cfg = Config()
    print(f"Loading dataset...")
    dataset = load_dataset("roneneldan/TinyStories",split = "train")

    def get_training_corpus():
        for i in range(0,len(dataset),1000):
            yield dataset[i:i+1000]["text"]

    tokenizer = ByteLevelBPETokenizer()
    print(f"Training tokenizer (vocab_size={cfg.vocab_size})")
    tokenizer.train_from_iterator(
        get_training_corpus(),
        vocab_size     = cfg.vocab_size,
        min_frequency  = 2,
        special_tokens = ["<|endoftext|>", "<pad>"]
                                  )

    save_path = os.path.join(
        os.path.dirname(__file__),"tokenizer"
    )
    os.makedirs(
        save_path, exist_ok=True
    )
    tokenizer.save_model(save_path)
    print(f"Tokenizer saved to {save_path}")

if __name__ == "__main__":
    main()


