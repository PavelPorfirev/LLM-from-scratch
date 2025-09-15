class BPE:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = []
        self.token2id = {}
        self.id2token = {}

    def fit(self, text):
        tokens = list(text)
        vocab = sorted(set(tokens))
        while len(vocab) < self.vocab_size:
            pair_counts = {}
            for i in range(len(tokens)-1):
                pair = tokens[i] + tokens[i+1]
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
            if not pair_counts:
                break
            max_pair = max(pair_counts, key=pair_counts.get)
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens)-1 and tokens[i]+tokens[i+1] == max_pair:
                    new_tokens.append(max_pair)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            if max_pair not in vocab:
                vocab.append(max_pair)
        self.vocab = vocab
        self.token2id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id2token = {i: tok for tok, i in self.token2id.items()}

    def encode(self, text):
        out = []
        i = 0
        max_len = max((len(t) for t in self.vocab), default=1)
        while i < len(text):
            end = min(len(text), i + max_len)
            found = False
            while end > i:
                piece = text[i:end]
                if piece in self.token2id:
                    out.append(self.token2id[piece])
                    i = end
                    found = True
                    break
                end -= 1
            if not found:
                out.append(self.token2id.get(text[i], 0))
                i += 1
        return out

    def decode(self, ids):
        return ''.join(self.id2token[i] for i in ids)
