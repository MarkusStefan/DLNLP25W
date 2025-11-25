import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Transformers library not found. Please install it or include it in the submission.")

# --- Model Architecture (Must match the trained model) ---

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers=1, dropout=0.1, pad_id=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.rnn = nn.GRU(emb_dim, hid_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)

    def forward(self, src, src_lens):
        emb = self.emb(src)
        packed = nn.utils.rnn.pack_padded_sequence(emb, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        outputs, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        # Concatenate last forward and backward hidden states
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = torch.tanh(self.fc(hidden_cat))
        
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 3, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch, hid_dim]
        # encoder_outputs: [batch, src_len, hid_dim * 2]
        src_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers=1, dropout=0.1, pad_id=0):
        super().__init__()
        self.attention = Attention(hid_dim)
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.rnn = nn.GRU(hid_dim * 2 + emb_dim, hid_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc_out = nn.Linear(hid_dim + emb_dim + hid_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.emb(input))
        
        a = self.attention(hidden[-1], encoder_outputs)
        a = a.unsqueeze(1)
        
        weighted = torch.bmm(a, encoder_outputs)
        
        rnn_input = torch.cat((embedded, weighted), dim=2)
        
        output, hidden = self.rnn(rnn_input, hidden)
        
        embedded = embedded.squeeze(1)
        output = output.squeeze(1)
        weighted = weighted.squeeze(1)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.encoder = enc
        self.decoder = dec

    def forward(self, src, src_lens, tgt_in):
        # Not used for inference
        pass

    @torch.no_grad()
    def greedy_decode(self, src, src_lens, max_len, sos_id, eos_id):
        batch_size = src.shape[0]
        device = src.device
        encoder_outputs, hidden = self.encoder(src, src_lens)
        
        # Prepare hidden for decoder (num_layers)
        num_layers = self.decoder.rnn.num_layers
        hidden = hidden.unsqueeze(0).repeat(num_layers, 1, 1)
        
        inputs = torch.tensor([sos_id] * batch_size).to(device)
        outs = []
        
        for _ in range(max_len):
            output, hidden = self.decoder(inputs, hidden, encoder_outputs)
            pred = output.argmax(1)
            outs.append(pred.unsqueeze(1))
            inputs = pred
            
        seqs = torch.cat(outs, dim=1)
        return seqs

# --- Agent Class ---

class Agent:
    """Agent that loads a trained NMT model (EN->DE) and generates translations."""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        
        # 1. Load Tokenizer
        # We use the same tokenizer as training (BERT tokenizer)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_en_de")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            return

        # 2. Locate Checkpoint
        # Check common locations
        checkpoint_path = 'checkpoint_last_2.pt' # Expected location for submission
        possible_paths = [
            checkpoint_path,
            os.path.join('checkpoints', 'checkpoint_last_2.pt'),
            os.path.join('..', 'notebooks', 'checkpoints', 'checkpoint_last_2.pt'),
            os.path.join('..', 'checkpoints', 'checkpoint_last_2.pt')
        ]
        
        found_path = None
        for p in possible_paths:
            if os.path.exists(p):
                found_path = p
                break
        
        if not found_path:
            print(f"Warning: Checkpoint not found. Searched: {possible_paths}")
            return
            
        print(f"Loading checkpoint from {found_path}")
        checkpoint = torch.load(found_path, map_location=self.device)
        cfg = checkpoint['model_cfg']
        
        # 3. Initialize Model
        pad_id_src = self.tokenizer.pad_token_id
        pad_id_tgt = self.tokenizer.pad_token_id
        self.sos_id = self.tokenizer.cls_token_id
        self.eos_id = self.tokenizer.sep_token_id
        
        # Note: vocab size might be slightly different if special tokens were added, 
        # but here we used the tokenizer's vocab directly.
        src_vocab_size = self.tokenizer.vocab_size
        tgt_vocab_size = self.tokenizer.vocab_size
        
        enc = Encoder(src_vocab_size, cfg['emb'], cfg['hid'], cfg['layers'], cfg['dropout'], pad_id=pad_id_src)
        dec = Decoder(tgt_vocab_size, cfg['emb'], cfg['hid'], cfg['layers'], cfg['dropout'], pad_id=pad_id_tgt)
        
        self.model = Seq2Seq(enc, dec).to(self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()

    def reset(self):
        pass

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Generate translations for the given input sentences.
        X_test: NumPy array of English sentences.
        Returns: NumPy array of German translations.
        """
        if self.model is None:
            return np.array(["Error: Model not loaded"] * len(X_test))
            
        translations = []
        self.model.eval()
        
        for sentence in X_test:
            if not isinstance(sentence, str):
                translations.append("")
                continue
                
            # Tokenize
            # Note: In training we did: encode(src) + [eos]
            # tokenizer(..., add_special_tokens=False) gives ids without special tokens.
            inputs = self.tokenizer(sentence, return_tensors="pt", add_special_tokens=False)
            src = inputs.input_ids.to(self.device)
            
            # Append EOS token
            src = torch.cat([src, torch.tensor([[self.eos_id]], device=self.device)], dim=1)
            src_len = torch.tensor([src.size(1)]).to(self.device)
            
            # Greedy Decode
            try:
                out_ids = self.model.greedy_decode(src, src_len, max_len=100, sos_id=self.sos_id, eos_id=self.eos_id)
                
                # Detokenize
                out_ids = out_ids.squeeze(0).tolist()
                if self.eos_id in out_ids:
                    out_ids = out_ids[:out_ids.index(self.eos_id)]
                
                trans = self.tokenizer.decode(out_ids, skip_special_tokens=True)
                translations.append(trans)
            except Exception as e:
                print(f"Error predicting sentence '{sentence}': {e}")
                translations.append("")
            
        return np.array(translations, dtype=object)