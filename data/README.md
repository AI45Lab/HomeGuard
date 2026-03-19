# Data

This directory separates HomeGuard data assets into three parts:

- `pipeline/`: code and notebooks for data construction.
- `metadata/`: JSON, TXT, and notebook assets describing generated samples and benchmark rewrites.
- `images/`: placeholder directories for images that are not redistributed in this repository.

Expected image layout:

- `data/images/homeguard/base_image`
- `data/images/homeguard/edit_image`
- `data/images/homeguard/annotate_image`
- `data/images/homeguard/test`
- `data/public_benches/earbench/images` for EARBench images from https://huggingface.co/datasets/ZihaoZhu/EARDataset
- `data/public_benches/mssbench/embodied` for MSSBench images from https://huggingface.co/datasets/kzhou35/mssbench/
- `data/public_benches/pasbench/combine_images` for PaSBench images from https://huggingface.co/datasets/Youliang/PaSBench/

Only metadata and pipeline code are included in this open-source release. Public benchmark image assets should be downloaded separately into the directories above.