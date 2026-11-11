# ProofGap

Implementation of ProofGap, a learned proof-deficit estimator for multi-hop evidence retrieval. The repository contains source code only. Datasets, model weights, experiment outputs, and manuscript files are not included.

## Requirements

Python 3.10+ and a CUDA-capable GPU are recommended. Core dependencies are PyTorch, Transformers, NumPy, SciPy, pandas, and scikit-learn. LLM-based baselines additionally require vLLM and compatible model weights. Install versions compatible with your CUDA setup.

```bash
python -m venv .venv
pip install torch transformers numpy scipy pandas scikit-learn
```

## Data

Download MuSiQue-Ans, HotpotQA, and 2WikiMultiHopQA from their official sources. `prep.py` converts each dataset into JSONL records with `id`, `question`, `answers`, `ctxs`, `gold`, `hops`, and `type` fields. The input and output locations are configured by `PROOFGAP_RAW_DIR` and `PROOFGAP_DATA_DIR`.

```bash
export PROOFGAP_RAW_DIR=/path/to/raw-data
export PROOFGAP_DATA_DIR=/path/to/processed-data
python prep.py musique
python prep.py hotpotqa
python prep.py 2wiki
```

Open-corpus preparation uses `open_prep.py` and `build_open_train.py`; it requires the corresponding retrieved candidate pools and an external first-stage retriever.

## Train and infer

The main model uses a DeBERTa-v3 encoder. Set `PROOFGAP_DATA_DIR` to the directory produced above. The commands below illustrate the member-view ordinal configuration; adjust batch size and paths for your hardware.

```bash
export PROOFGAP_DATA_DIR=/path/to/processed-data
python train.py --data musique --head ladder --v4 --views consensus --rungs 5 --open_mix 0.5 --enc /path/to/deberta-v3-base --out ckpt/musique --epochs 2 --bs 2 --accum 4
python infer.py --ckpt ckpt/musique --split musique_dev --pg4 2 --views consensus --out predictions.jsonl
```

`evaluate.py` provides retrieval metrics. Additional scripts cover baseline runs, stopping, calibration, and diagnostics. Some experiment-specific scripts still contain local path defaults; inspect and configure those before running them in another environment.

## Scope

This release contains no dataset copies, pretrained weights, saved checkpoints, generated results, manuscript, or submission PDF. Please obtain each benchmark and model under its own license. No license for this code is granted until one is added.
