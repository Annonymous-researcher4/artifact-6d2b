# Stage 1 — Unimodal Pair Encoders

This directory is an installable Python package that reproduces the upstream stage
that precedes the GNN:

- **Stage 1 — Unimodal pair encoders:** train the per-modality encoders and export
  the per-pair embeddings (`intermediate_embeddings.pkl`) plus a
  `config_and_results.json`. This is the input consumed by the GNN (Stage 3).

The **Stage 2 concat-MLP baseline** (the reported *intermediate-fusion baseline*)
lives in the sibling `concat_mlp/` package; it consumes Stage 1 outputs and is
**not** on the path to the GNN.

The precomputed embeddings shipped for Stage 3 (`../data/<dataset>/`) were produced
by running Stage 1 exactly as described below. Raw inputs (product images, 3D mesh
object dicts, dataset partitions) live under `../data/raw/`.

## Layout

```
unimodal_pair_encoders/                    # installable package (Stage 1)
├── __init__.py
├── run_embedding_branches.py              # WDC Shoes / Zalando (text + image)
├── run_hague_embedding_branches.py        # Hague (image, feature_vector, point_cloud, normals)
├── tf_utils.py, device_config.py, models_mps.py, utils.py
├── bimodal/                               # text/image branch code
│   ├── __init__.py
│   └── data_preparation.py, models.py
└── hague/                                 # 3D branch code
    ├── __init__.py
    └── config.py, utils.py, object_properties.py, process_pairs.py

concat_mlp/                                # Stage 2 baseline (sibling package)
├── __init__.py
└── run_concat_mlp.py
pyproject.toml                             # declares the two packages above
../data/raw/{bimodal,hague}/               # raw inputs
```

## Installation

Python 3.9 is recommended. From the **repository root**:

```bash
python -m venv venv
source venv/bin/activate
pip install -r unimodal_pair_encoders/requirements.txt
pip install git+https://github.com/openai/CLIP.git   # OpenAI CLIP (imported as `clip`)
pip install -e .                                      # register the packages
```

`pip install -e .` reads `pyproject.toml` and makes `unimodal_pair_encoders` and
`concat_mlp` importable. First run downloads pretrained weights (`roberta-base`,
`openai/clip-vit-base-patch32`, and the CLIP ViT-B/32 checkpoint) from
HuggingFace / OpenAI.

## Running

Run from the repository root. Results are written to `results/<stage1_dir>/`
relative to the current directory.

### Stage 1 — unimodal pair encoders

```bash
# WDC Shoes
python -m unimodal_pair_encoders.run_embedding_branches --dataset "WDC Shoes"

# Zalando
python -m unimodal_pair_encoders.run_embedding_branches --dataset "Zalando"

# Hague
python -m unimodal_pair_encoders.run_hague_embedding_branches
```

Each run creates `results/<stage1_dir>/` containing `intermediate_embeddings.pkl`
and `config_and_results.json` (and the trained branch models).

### Stage 2 — concat-MLP fusion (baseline)

Point it at a Stage 1 output directory; it selects the TensorFlow or PyTorch
fusion path automatically from that run's `config_and_results.json`:

```bash
python -m concat_mlp.run_concat_mlp --embedding-branches-dir results/<stage1_dir>
```

Results are written to `results/intermediate_fusion_<timestamp>/fusion_results_summary.json`.

### Feeding Stage 1 output into the GNN (Stage 3)

```bash
python run_gnn.py --embedding-branches-dir results/<stage1_dir>
```

(The bundled `data/<dataset>/` directories are exactly such Stage 1 outputs, trimmed
to the two files the GNN reads.)

