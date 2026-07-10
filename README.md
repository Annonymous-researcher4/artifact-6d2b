# SUMMER: Seamlessly Upgradable Multimodal Entity Resolution

This repository reproduces the results starting from **precomputed per-modality pair
embeddings** , which are included under `data/`.

The upstream stages that produce those embeddings are also included as installable
packages: **Stage 1** (per-modality pair encoders) under `unimodal_pair_encoders/`,
and the **Stage 2** intermediate-fusion baseline under `concat_mlp/` (register both
with `pip install -e .`). See `unimodal_pair_encoders/README.md` for how to
regenerate the embeddings from raw data (which lives under `data/raw/`). Stage 3
(this directory) is self-contained and does not require them.

## Installation

Python 3.9 is recommended.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

See `requirements.txt` for CPU vs. CUDA notes. The code automatically uses an Apple
MPS or NVIDIA CUDA device if available, and otherwise runs on CPU.

## Data

Each dataset directory under `data/` contains the precomputed pair embeddings and a
small config file read by the training script:

| Dataset     | Path            | Modalities                                   |
|-------------|-----------------|----------------------------------------------|
| WDC Shoes   | `data/shoes`    | text, image                                  |
| Zalando     | `data/zalando`  | text, image                                  |
| Hague       | `data/hague`    | image, feature_vector, point_cloud, normals  |

`intermediate_embeddings.pkl` holds `train` / `val` / `test` splits, each with a
`{modality}_hidden_embeddings` array (`N × 256`, `float32`) and a `labels` array.

## Reproducing the results

Run all commands from the repository root.

### Main results

The three per-dataset commands are in `scripts/recreate_main_results.txt`. 

### Experiments

```bash
# Section 6.2 — neighborhood size
bash scripts/section6_2_neighborhood_size.sh

# Section 6.3 — edge types (both / intra_only / inter_only)
bash scripts/section6_3_edge_types.sh

# Section 6.4 — number of modalities
bash scripts/section6_4_num_modalities.sh

# Section 6.5 — sequential training
bash scripts/section6_5_sequential_training.sh
```

Each script accepts `--dry-run` to print the commands without executing them.

## Outputs

Each run creates a timestamped directory under `results/` containing the trained
model, a metrics summary (`gnn_results_summary.json`), and a log.

## Repository layout

```
run_gnn.py            # training / evaluation entry point
gnn_models.py         # HAN model definition
graph_builder.py      # multiplex heterogeneous graph construction
metrics.py            # precision / recall / F1
utils.py              # experiment-directory helper
requirements.txt
pyproject.toml           # declares the Stage 1/2 packages (pip install -e .)
scripts/                 # reproduction scripts
data/                    # precomputed pair embeddings per dataset (+ raw/ inputs for Stage 1)
unimodal_pair_encoders/  # Stage 1 — per-modality pair encoders (installable package)
concat_mlp/              # Stage 2 — concat-MLP intermediate-fusion baseline (installable package)
```
