# PaSBench

This directory stores the released PaSBench metadata used by HomeGuard evaluation.

Included in this repository:
- `multi_modal_eval_hugging.json`: multimodal evaluation metadata.
- `text_eval_hugging.json`: text-only metadata from the upstream release.
- `severity.json`: severity annotations from the upstream release.
- `UPSTREAM_README.md`: copied from the original benchmark release for reference.

Not included in this repository:
- `combine_images/`: benchmark images. Download them from Hugging Face: https://huggingface.co/datasets/Youliang/PaSBench/

Expected layout:
```text
data/public_benches/pasbench/
├── multi_modal_eval_hugging.json
├── text_eval_hugging.json
├── severity.json
└── combine_images/
```
