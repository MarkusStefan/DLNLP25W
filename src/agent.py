import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from pathlib import Path


global MODEL_NR
MODEL_NR = 0

if MODEL_NR == 0:
    class Encoder(nn.Module):
        def __init__(self, vocab_size, emb_dim, hid_dim, num_layers=1, dropout=0.1):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)

        def forward(self, src, src_lens):
            emb = self.emb(src)
            packed = nn.utils.rnn.pack_padded_sequence(emb, src_lens.cpu(), batch_first=True, enforce_sorted=False)
            out, (h, c) = self.rnn(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            return out, (h, c)

    class Decoder(nn.Module):
        def __init__(self, vocab_size, emb_dim, hid_dim, num_layers=1, dropout=0.1):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
            self.proj = nn.Linear(hid_dim, vocab_size)

        def forward(self, tgt_in, hidden):
            emb = self.emb(tgt_in)
            out, hidden = self.rnn(emb, hidden)
            return self.proj(out), hidden

    class Seq2Seq(nn.Module):
        def __init__(self, enc, dec):
            super().__init__()
            self.encoder = enc
            self.decoder = dec

        def forward(self, src, src_lens, tgt_in):
            _, h = self.encoder(src, src_lens)
            logits, _ = self.decoder(tgt_in, h)
            return logits

        @torch.no_grad()
        def greedy_decode(self, src, src_lens, max_len, sos_id, eos_id):
            B = src.size(0)
            _, h = self.encoder(src, src_lens)
            inputs = torch.full((B, 1), sos_id, dtype=torch.long, device=src.device)
            outs = []
            for _ in range(max_len):
                logits, h = self.decoder(inputs[:, -1:].contiguous(), h)
                nxt = logits[:, -1, :].argmax(-1, keepdim=True)
                outs.append(nxt)
                inputs = torch.cat([inputs, nxt], dim=1)
            
            seqs = torch.cat(outs, dim=1)
            for i in range(B):
                row = seqs[i]
                if (row == eos_id).any():
                    idx = (row == eos_id).nonzero(as_tuple=False)[0].item()
                    row[idx + 1:] = eos_id
            return seqs
        
elif MODEL_NR == 1:
    class Encoder(nn.Module):
        def __init__(self, vocab_size, emb_dim, hid_dim, num_layers=2, dropout=0.1):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            self.rnn = nn.GRU(emb_dim, hid_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 0 else 0.0)

        def forward(self, src, src_lens):
            emb = self.emb(src)
            packed = nn.utils.rnn.pack_padded_sequence(emb, src_lens.cpu(), batch_first=True, enforce_sorted=False)
            out, h = self.rnn(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            return out, h


    class Decoder(nn.Module):
        def __init__(self, vocab_size, emb_dim, hid_dim, num_layers=1, dropout=0.1):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            self.rnn = nn.GRU(emb_dim, hid_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 0 else 0.0)
            self.fc1 = nn.Linear(hid_dim, vocab_size)

        def forward(self, tgt_in, hidden):
            emb = self.emb(tgt_in)
            out, hidden = self.rnn(emb, hidden)
            return self.fc1(out), hidden


    class Seq2Seq(nn.Module):
        def __init__(self, enc, dec):
            super().__init__()
            self.encoder = enc
            self.decoder = dec

        def forward(self, src, src_lens, tgt_in):
            _, h = self.encoder(src, src_lens)
            logits, _ = self.decoder(tgt_in, h)
            return logits

        @torch.no_grad()
        def greedy_decode(self, src, src_lens, max_len, sos_id, eos_id):
            B = src.size(0)
            _, h = self.encoder(src, src_lens)
            inputs = torch.full((B, 1), sos_id, dtype=torch.long, device=src.device)
            outs = []
            for _ in range(max_len):
                logits, h = self.decoder(inputs[:, -1:].contiguous(), h)
                nxt = logits[:, -1, :].argmax(-1, keepdim=True)
                outs.append(nxt)
                inputs = torch.cat([inputs, nxt], dim=1)
            
            seqs = torch.cat(outs, dim=1)
            for i in range(B):
                row = seqs[i]
                if (row == eos_id).any():
                    idx = (row == eos_id).nonzero(as_tuple=False)[0].item()
                    row[idx + 1:] = eos_id
            return seqs





class Agent:
    """Submitter agent for ML-Arena."""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_nr = MODEL_NR
        self._load_checkpoint()


    def _build_model(self, cfg: dict, src_vocab_size: int, tgt_vocab_size: int) -> nn.Module:
        emb_dim = cfg.get('emb', 256)
        hid_dim = cfg.get('hid', 512)
        layers = cfg.get('layers', 1)
        dropout = cfg.get('dropout', 0.1)
        encoder = Encoder(src_vocab_size, emb_dim, hid_dim, num_layers=layers, dropout=dropout)
        decoder = Decoder(tgt_vocab_size, emb_dim, hid_dim, num_layers=layers, dropout=dropout)
        return Seq2Seq(encoder, decoder)

    def _load_checkpoint(self):
        candidates = [
            Path(f'checkpoint_last_{self.model_nr}.pt'),
            Path(f'checkpoints/checkpoint_last_{self.model_nr}.pt'),
            Path(f'../checkpoints/checkpoint_last_{self.model_nr}.pt'),
            Path(f'../../checkpoints/checkpoint_last_{self.model_nr}.pt'),
            Path(__file__).parent.parent / 'checkpoints' / f'checkpoint_last_{self.model_nr}.pt'
        ]
        path = next((p for p in candidates if p.exists()), None)
        if not path:
            raise FileNotFoundError(f"Checkpoint for model {self.model_nr} not found.")
        
        ckpt = torch.load(path, map_location=self.device)
        # str to id mappings
        self.src_stoi = ckpt['src_stoi']
        self.tgt_stoi = ckpt['tgt_stoi']
        # id to str mappings
        self.src_itos = {v: k for k, v in self.src_stoi.items()}
        self.tgt_itos = {v: k for k, v in self.tgt_stoi.items()}
        
        model_cfg = ckpt.get('model_cfg', {})
        # Ensure vocab sizes match what was saved
        src_vocab_size = max(self.src_stoi.values()) + 1
        tgt_vocab_size = max(self.tgt_stoi.values()) + 1
        
        self.model = self._build_model(model_cfg, src_vocab_size, tgt_vocab_size).to(self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.model.eval()

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        preds = []
        unk = self.src_stoi.get('<unk>', 0)
        eos_src = self.src_stoi['<eos>']
        sos_tgt = self.tgt_stoi['<sos>']
        eos_tgt = self.tgt_stoi['<eos>']

        for text in X_test:
            tokens = text.strip().lower().split()
            ids = [self.src_stoi.get(t, unk) for t in tokens] + [eos_src]
            src = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
            src_len = torch.tensor([len(ids)], dtype=torch.long, device=self.device)
            
            out = self.model.greedy_decode(src, src_len, 100, sos_tgt, eos_tgt)
            
            out_toks = []
            for idx in out[0].tolist():
                if idx == eos_tgt:
                    break
                if idx in self.tgt_itos:
                    out_toks.append(self.tgt_itos[idx])
            preds.append(" ".join(out_toks))
            
        return np.array(preds, dtype=object)
        





if __name__ == '__main__':
    
    agent = Agent()

    X_test = np.array([
        "Hello, how are you?",
        "The weather is nice today.",
        "I love machine learning."
    ], dtype=object)

    predictions = agent.predict(X_test)
    for src, pred in zip(X_test, predictions):
        print(f"Source: {src}")
        print(f"Prediction: {pred}")
        print()
