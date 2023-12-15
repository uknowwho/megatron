import torch


class Data:
    def __init__(self, file_name):
        self.file_name = file_name
        self.stoi = dict()
        self.itos = dict()

    def encode(self, s):
        return [self.stoi[char] for char in s]

    def decode(self, l):
        return "".join(self.itos[i] for i in l)

    def load_data(self):
        with open(self.file_name, "r") as f:
            all_articles = f.read()

        chars = sorted(list(set(all_articles)))
        vocab_size = len(chars)

        self.stoi = {char: i for i, char in enumerate(chars)}
        self.itos = {i: char for i, char in enumerate(chars)}

        data = torch.tensor(self.encode(all_articles), dtype=torch.long)
        return data, vocab_size

    def train_test_split(self, all_data, ratio=0.7):
        return all_data[: int(all_data.size(dim=0) * ratio)], all_data[
            int(all_data.size(dim=0) * (ratio)) :
        ]


# TODO: if name == main
