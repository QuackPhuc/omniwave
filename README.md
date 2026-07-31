# OmniWeave

A vision backbone that replaces convolutions and self-attention with **structured two-sided tiled GEMMs** — jointly mixing spatial tokens and channel features through deterministic routing patterns.

## Architecture

OmniWeave-T is the Tiny variant (10–20M parameters), comparable to ConvNeXt-T, DeiT-S, and Swin-T.

```
Input [B, 3, 224, 224]
  │
  ├─ Stem: space-to-depth(4) → Linear(48, 128)
  │
  ├─ Stage 1:  56×56, 128-d,  2 blocks
  ├─ Stage 2:  28×28, 256-d,  3 blocks
  ├─ Stage 3:  14×14, 512-d,  8 blocks
  ├─ Stage 4:   7×7, 1024-d,  2 blocks
  │
  └─ Head: global avg pool → Linear(1024, 1000)
```

Each block: **RMSNorm → BiGEMM → residual**. Routes cycle through local → shifted → radix patterns.

## Setup

```bash
pip install -e .            # base
pip install -e ".[dev]"     # + pytest, ruff, mypy
pip install -e ".[triton]"  # + Triton (NVIDIA GPU)
```

## Usage

```python
from omniweave import create_model

model = create_model("omniweave_t")
logits = model(torch.randn(1, 3, 224, 224))  # → [1, 1000]
```

## Training

```bash
python scripts/train.py --config configs/train/overfit.yaml        # sanity check
python scripts/train.py --config configs/train/imagenet100.yaml    # ImageNet-100
torchrun --nproc_per_node=8 scripts/train.py \
  --config configs/train/imagenet1k_300e.yaml                      # full run
```

## Benchmarking

```bash
python scripts/benchmark.py \
  --config configs/benchmark/operator.yaml \
  --output results/operator.json

python scripts/check_gates.py \
  --operator-results results/operator.json \
  --output results/gates.json
```

## Kaggle / Colab validation notebooks

The local-only notebooks under `notebooks/` are intentionally ignored by Git.
Run them in this order after setting `OMNIWAVE_ROOT` and `IMAGENET100_ROOT`:

1. `00_environment_preflight.ipynb`
2. `01_operator_gate_triton.ipynb`
3. `02_train_smoke_single_gpu.ipynb`
4. `03_train_ddp_resume.ipynb`
5. `04_train_imagenet100.ipynb`

Use `backend: reference` for smoke training. Enable Triton training only after
the operator Gate A/B notebook passes.

## Layout

```
omniweave/           Python package
  models/            routing, block, backbone, registry
  ops/               BiGEMM operator (reference, compile, Triton)
  data/              ImageNet validation and transforms
  training/          engine, checkpointing, DDP
  evaluation/        metrics, benchmarking, profiling
  utils/             config, logging, environment
configs/             YAML configurations (model, train, benchmark)
scripts/             CLI entry points
tests/               test suite
```

## License

Apache-2.0
