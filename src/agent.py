import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from pathlib import Path


global MODEL_NR
MODEL_NR = 3

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




# MODEL 2: Using GRU, torch.float16 and Attention
if MODEL_NR == 2:

    class BahdanauAttention(nn.Module):
        def __init__(self, hidden_size):
            super(BahdanauAttention, self).__init__()
            self.Wa = nn.Linear(hidden_size, hidden_size)
            self.Ua = nn.Linear(hidden_size, hidden_size)
            self.Va = nn.Linear(hidden_size, 1)

        def forward(self, query, keys):
            # module() calls .forward() aka matmul A@X + b
            scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
            scores = scores.squeeze(2).unsqueeze(1)

            weights = F.softmax(scores, dim=-1)
            context = torch.bmm(weights, keys) # batch matrix-matrix product

            return context, weights

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
            self.attention = BahdanauAttention(hid_dim)
            self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 0 else 0.0)
            self.fc1 = nn.Linear(hid_dim, vocab_size)

        def forward(self, tgt_in, hidden, encoder_outputs):
            tgt_in = tgt_in.unsqueeze(1)
            emb = self.emb(tgt_in)
            
            query = hidden[-1].unsqueeze(1)
            context, weights = self.attention(query, encoder_outputs)
            
            rnn_input = torch.cat((emb, context), dim=2)
            out, hidden = self.rnn(rnn_input, hidden)
            return self.fc1(out.squeeze(1)), hidden


    class Seq2Seq(nn.Module):
        def __init__(self, enc, dec):
            super().__init__()
            self.encoder = enc
            self.decoder = dec

        def forward(self, src, src_lens, tgt_in):
            encoder_outputs, hidden = self.encoder(src, src_lens)
            
            if self.encoder.rnn.num_layers != self.decoder.rnn.num_layers:
                hidden = hidden[-self.decoder.rnn.num_layers:]

            batch_size = src.shape[0]
            tgt_len = tgt_in.shape[1]
            vocab_size = self.decoder.fc1.out_features
            
            outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(src.device)
            
            for t in range(tgt_len):
                output, hidden = self.decoder(tgt_in[:, t], hidden, encoder_outputs)
                outputs[:, t] = output
            
            return outputs

        @torch.no_grad()
        def greedy_decode(self, src, src_lens, max_len, sos_id, eos_id):
            B = src.size(0)
            encoder_outputs, hidden = self.encoder(src, src_lens)
            
            if self.encoder.rnn.num_layers != self.decoder.rnn.num_layers:
                hidden = hidden[-self.decoder.rnn.num_layers:]

            inputs = torch.full((B,), sos_id, dtype=torch.long, device=src.device)
            outs = []
            for _ in range(max_len):
                output, hidden = self.decoder(inputs, hidden, encoder_outputs)
                nxt = output.argmax(1)
                outs.append(nxt.unsqueeze(1))
                inputs = nxt
            
            seqs = torch.cat(outs, dim=1)
            for i in range(B):
                row = seqs[i]
                if (row == eos_id).any():
                    idx = (row == eos_id).nonzero(as_tuple=False)[0].item()
                    row[idx + 1:] = eos_id
            return seqs




if MODEL_NR == 3:
    class BahdanauAttention(nn.Module):
        def __init__(self, enc_hid_dim, dec_hid_dim):
            super().__init__()
            # enc_hid_dim will be doubled due to bidirectional encoder
            self.Wa = nn.Linear(enc_hid_dim, dec_hid_dim)
            self.Ua = nn.Linear(dec_hid_dim, dec_hid_dim)
            self.Va = nn.Linear(dec_hid_dim, 1)

        def forward(self, hidden, encoder_outputs):
            # hidden: [1, batch_size, dec_hid_dim]
            # encoder_outputs: [batch_size, src_len, enc_hid_dim]
            hidden = hidden.permute(1, 0, 2)  # [batch, 1, dec_hid_dim]
            # calc Energy — broadcast addition aligns time dim
            energy = torch.tanh(self.Wa(encoder_outputs) + self.Ua(hidden))
            scores = self.Va(energy)  # [batch, src_len, 1]
            # softmax for weights
            weights = F.softmax(scores, dim=1)  # [batch, src_len, 1]
            # context vector: weighted sum of encoder outputs
            context = torch.bmm(weights.permute(0, 2, 1), encoder_outputs)  # [batch, 1, enc_hid_dim]
            return context, weights

    class Encoder(nn.Module):
        def __init__(self, vocab_size, emb_dim, hid_dim, num_layers=1, dropout=0.1):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            self.rnn = nn.GRU(
                emb_dim, hid_dim, num_layers=num_layers, batch_first=True, bidirectional=True,
                dropout=dropout if num_layers > 1 else 0
            )
            self.fc_hidden = nn.Linear(hid_dim * 2, hid_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, src, src_lens):
            emb = self.dropout(self.emb(src))
            packed = nn.utils.rnn.pack_padded_sequence(emb, src_lens.cpu(), batch_first=True, enforce_sorted=False)
            out, hidden = self.rnn(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

            # hidden shape: [num_layers * 2, batch, hid_dim] --> grab final forward/backward states
            fwd_last = hidden[-2, :, :]
            bwd_last = hidden[-1, :, :]
            merged_hidden = torch.tanh(self.fc_hidden(torch.cat((fwd_last, bwd_last), dim=1)))  # [batch, hid_dim]

            return out, merged_hidden

    class Decoder(nn.Module):
        def __init__(self, vocab_size, emb_dim, hid_dim, num_layers=1, dropout=0.1):
            super().__init__()
            self.emb_dim = emb_dim
            self.hid_dim = hid_dim
            self.vocab_size = vocab_size
            self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            self.attention = BahdanauAttention(hid_dim * 2, hid_dim)
            self.rnn = nn.GRU(
                emb_dim + (hid_dim * 2), hid_dim, num_layers=num_layers, batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
            self.fc1 = nn.Linear(hid_dim + (hid_dim * 2), vocab_size)
            self.dropout = nn.Dropout(dropout)

            # optionally: weight tying when projection size matches embedding size
            if emb_dim == (hid_dim + (hid_dim * 2)):
                self.fc1.weight = self.emb.weight

        def forward(self, tgt_in, hidden, encoder_outputs):
            # tgt_in: [batch]
            tgt_in = tgt_in.unsqueeze(1)  # [batch, 1]
            emb = self.dropout(self.emb(tgt_in))  # [batch, 1, emb_dim]
            # attention uses the top decoder layer as query
            context, _ = self.attention(hidden[-1:].contiguous(), encoder_outputs)  # [batch, 1, hid_dim * 2]
            rnn_input = torch.cat((emb, context), dim=2)
            out, hidden = self.rnn(rnn_input, hidden)  # out: [batch, 1, hid_dim]

            prediction_input = torch.cat((out, context), dim=2)
            prediction = self.fc1(prediction_input.squeeze(1))  # [batch, vocab_size]

            return prediction, hidden

    class Seq2Seq(nn.Module):
        def __init__(self, enc, dec):
            super().__init__()
            self.encoder = enc
            self.decoder = dec

        def _prepare_decoder_hidden(self, hidden):
            # hidden can be [batch, hid_dim] (from encoder merge) or [layers, batch, hid_dim]
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)
            layers_available = hidden.size(0)
            layers_needed = self.decoder.rnn.num_layers
            if layers_available == layers_needed:
                return hidden.contiguous()
            if layers_available > layers_needed:
                return hidden[-layers_needed:].contiguous()
            pad = hidden[-1:].repeat(layers_needed - layers_available, 1, 1)
            return torch.cat((hidden, pad), dim=0).contiguous()

        def forward(self, src, src_lens, tgt_in):
            encoder_outputs, hidden = self.encoder(src, src_lens)
            hidden = self._prepare_decoder_hidden(hidden)

            batch_size = src.size(0)
            tgt_len = tgt_in.size(1)
            vocab_size = self.decoder.vocab_size
            outputs = torch.zeros(batch_size, tgt_len, vocab_size, device=src.device)

            for t in range(tgt_len):
                output, hidden = self.decoder(tgt_in[:, t], hidden, encoder_outputs)
                outputs[:, t] = output

            return outputs

        @torch.no_grad()
        def greedy_decode(self, src, src_lens, max_len, sos_id, eos_id):
            encoder_outputs, hidden = self.encoder(src, src_lens)
            hidden = self._prepare_decoder_hidden(hidden)

            batch_size = src.size(0)
            inputs = torch.full((batch_size,), sos_id, dtype=torch.long, device=src.device)
            outs = []
            finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)

            for _ in range(max_len):
                output, hidden = self.decoder(inputs, hidden, encoder_outputs)
                next_tokens = output.argmax(1)
                outs.append(next_tokens.unsqueeze(1))
                finished |= (next_tokens == eos_id)
                inputs = next_tokens
                if finished.all():
                    break

            if len(outs) == 0:
                return torch.zeros(batch_size, 0, dtype=torch.long, device=src.device)

            seqs = torch.cat(outs, dim=1)
            for i in range(batch_size):
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
