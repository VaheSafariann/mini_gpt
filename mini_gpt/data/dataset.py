import torch
from torch.utils.data import Dataset



class TinyStoriesDataset(Dataset):

    def __init__(self, dataset, tokenizer,context_len):
        self.context_len = context_len
        all_tokens = []
        eos_id = tokenizer.token_to_id("<|endoftext|>")
        if eos_id is None:
            eos_id = tokenizer.token_to_id("</s>")

        print(f"End Of Sequence id : {eos_id}")
        for item in dataset:
            text = item['text']
            ids = tokenizer.encode(text).ids
            all_tokens.extend(ids)
            all_tokens.append(eos_id)
        self.tokens = torch.tensor(all_tokens,dtype=torch.long)
        print("Done!")

    def __len__(self):
        return len(self.tokens)//self.context_len
    def __getitem__(self, idx):

        start = idx*self.context_len
        end = (idx+1)*self.context_len
        chunk = self.tokens[start:end+1]
        x = chunk[:self.context_len]
        y = chunk[1:]
        return x,y


