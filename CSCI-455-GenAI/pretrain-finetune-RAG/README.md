# Assignment 3 — Pre-training, Fine-tuning, and RAG for Bug Fixing

**CSCI 455/555: GenAI for Software Development — Spring 2026**

This project investigates whether pre-training helps a small T5 model learn bug-fixing, and compares the fine-tuned variants against a RAG-augmented prompting strategy using Qwen2.5-Coder.

## Pipelines

| # | Notebook | Produces |
|---|---|---|
| 1 | `1_tokenizer_and_pretrain.ipynb` | SentencePiece tokenizer (Unigram, 16,384 vocab) + pre-trained T5-small checkpoint |
| 2 | `2_finetune_and_evaluate.ipynb`  | Two fine-tuned models (A = pre-trained init, B = scratch init) + CodeBLEU / exact-match results |
| 3 | `3_rag.ipynb`                    | CodeBERT + FAISS knowledge base + RAG & zero-shot Qwen predictions + final four-way comparison |

## Setup

Designed for **Google Colab** (T4 GPU or better). To reproduce:

1. Open each notebook in Colab.
2. Run cell `0.1` — installs pinned dependencies:
   - `transformers==4.46.0`, `tokenizers==0.20.3`, `sentencepiece==0.2.0`
   - `datasets==2.14.0`, `torch`, `tqdm`, `matplotlib`
   - `codebleu`, `tree-sitter==0.22.3`, `tree-sitter-java` (notebooks 2 & 3)
   - `faiss-cpu`, `accelerate` (notebook 3)
3. Run cell `0.3` — mounts Google Drive and sets `BASE_DIR` to `/content/drive/MyDrive/W&M/GenAI/Assignment_3`. Adjust `BASE_DIR` if running elsewhere.
4. Run all cells top-to-bottom. Notebook 1 must finish before Notebook 2; Notebook 2 must finish before Notebook 3 (the final comparison cell reads Notebook 2's metrics).

## Where outputs are written

All outputs live under `${BASE_DIR}`:

```
/Assignment_3/
├── data/                   # pre-training corpus (.txt)
├── tokenizer/              # sp_model.model + T5Tokenizer files
├── pretrained/             # Pipeline A checkpoint (pre-training only)
├── finetuned_A/            # Pipeline A final checkpoint (+ history.json)
├── finetuned_B/            # Pipeline B final checkpoint (+ history.json)
├── rag_kb/                 # index.faiss, pairs.pkl, config.json
└── results/
    ├── preds_A.json preds_B.json preds_RAG.json preds_ZeroShot.json
    ├── metrics_A.json metrics_B.json metrics_RAG.json metrics_ZeroShot.json
    ├── final_comparison.json
    ├── finetune_loss.png
    └── pretrain_loss.png
```

## Key hyperparameters

| Setting | Value | Rationale |
|---|---|---|
| SP vocab | 16,384 (16,284 pieces + 100 sentinels) | Per assignment spec |
| T5-small config | `d_model=512, d_ff=2048, d_kv=64, heads=8, 6/6 layers` | Per assignment spec |
| Pre-training | 3 epochs, lr=1e-4, bs=16, span-rate=15% | Per assignment spec — no validation, no early stopping |
| Fine-tuning (A & B) | 10 epochs, lr=1e-4, bs=16, beams=4 | Same across pipelines — only variable is pre-training |
| Retriever | CodeBERT mean-pooling + FAISS L2 | Contextual embeddings, no AST parsing |
| RAG shots | 3 | Minimum per assignment |

## Evaluation

CodeBLEU (n-gram + weighted n-gram + syntax + dataflow, `lang="java"`, uniform weights) and exact-match (whitespace-normalized) on the CodeXGLUE `code_refinement` **medium** test split (~6.5K instances). To run on a subset (faster), set `TEST_SUBSET_SIZE = 500` in Notebook 2 *and* Notebook 3 and re-run the generation cells.

## Notes

- Random seeds (`42`) are set in every notebook for reproducibility.
- The tokenizer from Notebook 1 is the *single source of truth*; Notebooks 2 & 3 load from `tokenizer/` without retraining.
- Pipeline B uses an **identical** `T5Config` to Pipeline A — the only difference is that B initializes from random weights.
