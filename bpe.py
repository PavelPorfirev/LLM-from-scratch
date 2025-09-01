class Naive_BPE:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.vocab = []
        self.id2token = {}
        self.token2id = {}

    def fit(self, text: str):
        self.vocab = sorted(set(text))
        tokens = list(text)

        while len(self.vocab) < self.vocab_size:
            pair_counts = {}
            for a, b in zip(tokens, tokens[1:]):
                pair = a + b
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                break

            max_pair = max(pair_counts, key=pair_counts.get)
            self.vocab.append(max_pair)

            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] + tokens[i+1] == max_pair:
                    new_tokens.append(max_pair)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        for i in range(self.vocab_size):
            self.id2token[i] = self.vocab[i]
            self.token2id[self.vocab[i]] = i

class BPE:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.vocab = []
        self.id2token = {}
        self.token2id = {}

    def fit(self, text: str):
        self.vocab = sorted(set(text))
        tokens = list(text)

        while len(self.vocab) < self.vocab_size:
            pair_counts = {}
            for i in range(len(tokens) - 1):
                pair = tokens[i] + tokens[i + 1]
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                break

            max_pair = max(pair_counts, key=pair_counts.get)
            self.vocab.append(max_pair)

            new_tokens = []
            skip = False
            for i in range(len(tokens)):
                if skip:
                    skip = False
                    continue
                if i < len(tokens) - 1 and tokens[i] + tokens[i + 1] == max_pair:
                    new_tokens.append(max_pair)
                    skip = True
                else:
                    new_tokens.append(tokens[i])
            tokens = new_tokens

        self.id2token = {i: tok for i, tok in enumerate(self.vocab)}
        self.token2id = {tok: i for i, tok in enumerate(self.vocab)}
