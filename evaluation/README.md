# Evaluation

This directory contains:

- `evaluation.py`: main evaluator for HomeSafe-Bench.
- `eval_earbench.py`, `eval_mssbench.py`, `eval_pasbench.py`, `eval_sabench.py`: evaluation entry points for public benchmarks.
- `inference.py`, `judgement.py`, `prompt.py`, `visualization.py`: shared utilities.
- `scripts/`: example launch scripts to adapt to your own endpoints.

Public benchmark metadata is bundled in this release, but public benchmark images are not. Place downloaded benchmark images under `data/public_benches/earbench/images/`, `data/public_benches/mssbench/embodied/`, and `data/public_benches/pasbench/combine_images/` as described in the main README.
