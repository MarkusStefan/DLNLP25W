```markdown
# NMT Homework (EN→DE)

Train an English→German translation model. Save a checkpoint. Report **perplexity** and **sacreBLEU**. Show a few example translations. Optionally export predictions for ML-Arena.

---

## Data

- Folder: `dataset_splits/`
  - `train.txt`, `val.txt`, `public_test.txt`
- Format: tab-separated `EN<TAB>DE`, UTF-8.
```

i am happy .	ich bin glücklich .
how are you ?	wie geht es dir ?

````

---

## Requirements

- Python 3.10+
- PyTorch, sacreBLEU, tqdm
```bash
pip install torch sacrebleu tqdm
````

GPU is optional but recommended.

---

## Structure

```
.
├─ dataset_splits/
│  ├─ train.txt
│  ├─ val.txt
│  └─ public_test.txt
├─ model.py              # you implement
├─ nmt_homework.ipynb    # train + eval + export
├─ checkpoints/          # created by training
└─ submissions/          # exported TSV
```

---

## Your tasks

1. **Implement `model.py`.**
   Provide tokens/utilities and a model that fits the notebook’s I/O.

   Required symbols:

   ```python
   SPECIAL_TOKENS = {"pad": "<pad>", "sos": "<sos>", "eos": "<eos>", "unk": "<unk>"}

   def simple_tokenize(s: str) -> list[str]
   def encode(tokens: list[str], stoi: dict[str,int], add_sos_eos: bool=False) -> list[int]
   ```

   Required **model interface** (architecture is your choice: LSTM/GRU/Transformer/etc.):

   ```python
   class ModelOrWrapper(nn.Module):
       def forward(self, src, src_lens, tgt_in):
           """Return logits [B, T, V] for teacher forcing."""

       @torch.no_grad()
       def greedy_decode(self, src, src_lens, max_len, sos_id, eos_id):
           """Return LongTensor[B, max_len] predicted ids."""
   ```

   If you prefer separate components (e.g., `Encoder`, `Decoder`, `Seq2Seq`), expose a final class with the same `forward` and `greedy_decode`. Update the notebook import accordingly.

2. **Run `nmt_homework.ipynb`.**
   The notebook:

   * Reads data, builds vocab, encodes `(tgt_in, tgt_out)` for teacher forcing.
   * Batches with padding and sequence lengths.
   * Trains with tqdm progress bars and gradient clipping.
   * Saves `checkpoints/checkpoint_last.pt` with vocab and config.
   * Evaluates PPL and sacreBLEU.
   * Prints example translations.
   * Exports `submissions/public_predictions.tsv`.

3. **Report metrics.**

   * Validation perplexity
   * Public test perplexity
   * Public test BLEU (sacreBLEU, shown as `%`)

4. **Show examples.**
   Print 5–10 `(SRC, REF, HYP)` triplets.

5. **(Optional) Export for LM-Arena.**

   * TSV is created by the notebook: `submissions/public_predictions.tsv`.
   * For agent-based submission or API usage, see:

     * GitHub agent: [https://github.com/ml-arena/translate](https://github.com/ml-arena/translate)
     * Competition page: [https://ml-arena.com/viewcompetition/14](https://ml-arena.com/viewcompetition/14)

---

## Standard evaluation

* **Perplexity (PPL)**
  `CrossEntropyLoss(ignore_index=pad, reduction='sum')` → `PPL = exp(total_loss / nonpad_tokens)`.

* **BLEU**
  `sacrebleu.corpus_bleu(hyp_strings, references)` (BLEU-4). The notebook prints `BLEU * 100`.

Both evaluation and export display tqdm progress bars.


## Checkpoint format

Saved at `checkpoints/checkpoint_last.pt`:

```python
{
  "model_state": ...,
  "optimizer_state": ...,
  "epoch": ...,
  "src_stoi": ...,
  "tgt_stoi": ...,
  "model_cfg": {
    "emb": ..., "hid": ..., "layers": ..., "dropout": ...,
    "sos_id": ..., "eos_id": ...
  }
}
```

Reload with:

```python
state = torch.load("checkpoints/checkpoint_last.pt", map_location=device)
model.load_state_dict(state["model_state"], strict=False)
```

---

## Deliverables

* `model.py` with your implementation.
* Notebook run showing:

  * Validation PPL
  * Public test PPL
  * Public test BLEU
  * 5–10 sample translations
* `submissions/public_predictions.tsv` (optional)
* checkpoint file so I can verify perplexity, BLEU
---

## Notes

* Architecture is flexible. Keep the model I/O contract unchanged.
* Keep `ignore_index=pad_id` and `reduction='sum'` for correct PPL.
* Use `reshape` on logits before loss.
* Use `clip_grad_norm_` to stabilize training.

```
::contentReference[oaicite:0]{index=0}
```
