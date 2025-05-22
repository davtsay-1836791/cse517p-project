import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
import re
from utils.normalize import normalize_v2

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=1):
        super(CharLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        output, hidden = self.lstm(x, hidden)
        output = self.fc(output)
        return output, hidden

class CharLSTMWrapper:
    def __init__(self, model=None, vocab=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab = vocab or {}
        self.idx2char = {i: c for c, i in self.vocab.items()}
        self.model = model or CharLSTM(vocab_size=len(self.vocab)).to(self.device)

    def save(self, path):
        torch.save(self.model.state_dict(), os.path.join(path, 'char_lstm.pt'))
        with open(os.path.join(path, 'vocab.pkl'), 'wb') as f:
            pickle.dump(self.vocab, f)

    @classmethod
    def load(cls, path):
        with open(os.path.join(path, 'vocab.pkl'), 'rb') as f:
            vocab = pickle.load(f)
        model = CharLSTM(vocab_size=len(vocab))
        model.load_state_dict(torch.load(os.path.join(path, 'char_lstm.pt'), map_location='cpu'))
        return cls(model=model, vocab=vocab)

    @staticmethod
    def preprocess_conversations(conversation_str_list):
        """
        Extracts 'value' texts using regex, normalizes them with normalize_v2,
        and joins with <sos> and <eos> tokens.
        """
        pattern = re.compile(r"'value'\s*:\s*'(.*?)'", re.DOTALL)
        normalized_texts = []
        START_TOKEN = '<sos>'
        END_TOKEN = '<eos>'

        for row in conversation_str_list:
            matches = pattern.findall(row)
            for text in matches:
                norm = normalize_v2(text)
                normalized_texts.append(f"{START_TOKEN}{norm}{END_TOKEN}")

        # Join all normalized sentences into one long string
        all_text = ''.join(normalized_texts)
        return all_text
    
    def train_model(self, conversations, epochs=5, lr=0.003, batch_size=64, seq_len=100):
        """
        Override train_model to accept raw conversations (list of strings).
        Normalize, build vocab, encode, and train.
        """
        # Preprocess raw conversations
        all_text = self.preprocess_conversations(conversations)

        # Build vocab if not already built
        if not self.vocab:
            self.vocab = self.build_vocab([all_text])
            self.idx2char = {i: c for c, i in self.vocab.items()}
            self.model = CharLSTM(vocab_size=len(self.vocab)).to(self.device)

        # Encode text into tensor of indices
        encoded = torch.tensor([self.vocab.get(c, self.vocab['<unk>']) for c in all_text], dtype=torch.long)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        self.model.train()

        for epoch in range(epochs):
            for i in range(0, len(encoded) - seq_len, batch_size):
                batch_x = []
                batch_y = []
                for b in range(batch_size):
                    start = i + b
                    if start + seq_len >= len(encoded):
                        break
                    batch_x.append(encoded[start:start + seq_len])
                    batch_y.append(encoded[start + 1:start + seq_len + 1])
                if not batch_x:
                    continue

                batch_x = torch.stack(batch_x).to(self.device)
                batch_y = torch.stack(batch_y).to(self.device)

                optimizer.zero_grad()
                output, _ = self.model(batch_x)
                loss = criterion(output.view(-1, len(self.vocab)), batch_y.view(-1))
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch + 1} completed")

    def predict_next(self, input_seq, top_k=5):
        self.model.eval()
        input_encoded = torch.tensor([self.vocab.get(c, self.vocab['<unk>']) for c in input_seq], dtype=torch.long).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output, _ = self.model(input_encoded)
        output = output[0, -1]
        probs = F.softmax(output, dim=0)
        top_chars = torch.topk(probs, k=top_k)
        return [self.idx2char[i.item()] for i in top_chars.indices]

    @staticmethod
    def build_vocab(data):
        chars = sorted(set(''.join(data)))
        vocab = {ch: i for i, ch in enumerate(chars)}
        vocab['<unk>'] = len(vocab)
        return vocab
