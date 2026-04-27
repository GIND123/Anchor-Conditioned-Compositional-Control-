# Anchor-Conditioned Compositional Control for High-Resolution Landscape Generation

Research artifact and code release for the ICCC 2026 submission:

**Anchor-Conditioned Compositional Control for High-Resolution Landscape Image Generation**

This repository documents a two-stage landscape generation pipeline that treats composition as an explicit conditioning variable rather than a side effect of prompt priors. A 4D compositional anchor is extracted from landscape images, injected into a fine-tuned diffusion model through a decoupled cross-attention path, and then tracked during 4K upscaling to measure compositional drift.

![Sample 4K output](results/samples/IMAX_sample_results/stage4k.png)

## Contents

- [Abstract](#abstract)
- [Scientific Contributions](#scientific-contributions)
- [Repository Scope](#repository-scope)
- [Method Overview](#method-overview)
- [Main Results](#main-results)
- [Reproducibility and Environment](#reproducibility-and-environment)
- [How to Use This Repository](#how-to-use-this-repository)
- [Directory Structure](#directory-structure)
- [Included Artifacts](#included-artifacts)
- [Known Limitations](#known-limitations)
- [Citation](#citation)
- [License](#license)

## Abstract

This project introduces a multi-stage pipeline for landscape image generation with explicit compositional control. Each training image is mapped to a 4D anchor vector consisting of horizon position, horizon confidence, average saliency, and foreground ratio. The anchor is Fourier-encoded and injected into a fine-tuned diffusion model through a decoupled cross-attention mechanism with three-way classifier-free guidance dropout. The paper reports that this architecture outperforms an unmodified baseline, a LoRA-only fine-tune, and a text-concatenation anchor ablation on horizon detection, horizon placement error, and rule-of-thirds alignment.

The second stage is a training-free 4K refinement pipeline combining Real-ESRGAN pre-upscaling, tiled latent diffusion refinement, and Haar wavelet low-frequency color correction. The central research question is not only whether the model can place composition deliberately at generation time, but whether that compositional intent survives progressive upscaling.

## Scientific Contributions

The paper makes four main contributions:

- It defines a compact **4D compositional anchor** that turns landscape layout into an explicit conditioning signal.
- It shows that **decoupled cross-attention** is materially more effective than anchor-token concatenation for spatial control.
- It demonstrates that **category-specific training** can improve compositional precision beyond mixed-scene training.
- It treats high-resolution synthesis as a **composition-preservation problem**, quantifying drift across the full generation-to-upscaling pipeline rather than reporting only final-image aesthetics.

## Repository Scope

This repository is best understood as a **research artifact** rather than a polished end-user package.

- The **primary artifact** is the notebook-based implementation under `notebooks/`, plus the saved run directories under `models/`.
- The `scripts/` package is a **lightweight reference scaffold**. `scripts/anchor_extraction.py` is directly useful; `scripts/pipeline.py` sketches the pipeline structure but still contains placeholder model-loading and image-generation code.
- The repository already includes **sample outputs**, **saved training summaries**, **train configs**, **curve plots**, and **forest/desert ablation checkpoints**.
- The manuscript discusses more experiment variants than are bundled as standalone directories here. In particular, the paper tables include baseline, LoRA, concatenation, mixed-data, and category-specific comparisons, while the current checkout mainly ships notebook artifacts plus forest and desert run folders.
- Minor version drift exists between manuscript prose, notebook comments, and per-run configs. When reproducing a specific experiment, treat the saved `train_config.json` and `training_summary.json` files inside each run directory as the authoritative record for that run.

## Method Overview

### 1. Compositional anchor

Each image is mapped to a 4D vector:

- `horizon_y`: normalized vertical horizon position
- `horizon_conf`: confidence of detected horizon
- `avg_saliency`: average saliency score from spectral residual analysis
- `fg_ratio`: coarse foreground-content ratio

The anchor is designed to encode composition compactly enough to condition layout while remaining simple to extract offline across the full dataset.

### 2. Stage 1: anchor-conditioned diffusion

Stage 1 performs controlled landscape generation with an explicit anchor pathway:

- A 4D anchor is Fourier-encoded with 16 log-spaced frequencies.
- The encoded anchor is projected into a single token with an `AnchorProjModel`.
- That token is injected through **decoupled cross-attention**, following the IP-Adapter design pattern rather than competing inside the normal text-token sequence.
- Training uses **three-way classifier-free guidance dropout**:
  - both text and anchor dropped
  - text only dropped
  - anchor only dropped
  - both retained

The paper argument is that composition fails when the anchor is appended directly to text, because a single token is too weak relative to the pretrained text-conditioning budget. The decoupled pathway is the key architectural intervention.

### 3. Stage 2: 4K wavelet-guided refinement

Stage 2 is training-free and aims to preserve composition while increasing detail:

- **Real-ESRGAN pre-upscaling** produces an initial 4K image.
- **Tiled latent diffusion refinement** hallucinates high-frequency texture on overlapping tiles.
- **Per-tile horizon renormalization** re-expresses the global horizon target in each tile's local coordinate frame.
- **Haar wavelet color correction** replaces the low-frequency component with the Stage 1 reference to reduce color and luminance drift after tiled VAE decoding.

This stage is deliberately analyzed as a compositional-preservation problem, not only a perceptual-upscaling problem.

### 4. Evaluation protocol

The paper evaluates:

- horizon detection rate
- absolute horizon deviation
- rule-of-thirds alignment
- sharpness
- CLIP score
- NIQE
- Stage 2 compositional drift between the Stage 1 image and the final 4K output

## Main Results

### Four-model comparison

Average over 24 images per model, using 6 prompts x 4 seeds.

| Metric | Baseline | LoRA | Concat | Proposed |
| --- | ---: | ---: | ---: | ---: |
| Detection rate | 0.667 | 0.792 | 0.667 | **0.833** |
| Horizon deviation | 0.073 | 0.077 | 0.073 | **0.048** |
| Rule-of-thirds alignment | 0.500 | 0.625 | 0.500 | **0.917** |
| Sharpness | 1296 | 935 | 1296 | **3632** |
| CLIP score | 0.294 | 0.280 | 0.294 | **0.297** |
| NIQE | 8.79 | 8.51 | 8.79 | **7.99** |

The key conclusion is that the concatenation ablation collapses back to baseline behavior, while the decoupled anchor pathway improves both composition accuracy and image quality metrics.

### Category-specific ablation

| Model | Images | Steps | Detection rate | Horizon deviation |
| --- | ---: | ---: | ---: | ---: |
| Proposed (mixed) | 4,713 | 4,000 | 0.833 | 0.048 |
| Mountain | 1,534 | 1,301 | **1.000** | 0.033 |
| Forest | 1,149 | 1,301 | **1.000** | **0.029** |
| Desert | 1,042 | 1,301 | **1.000** | 0.033 |

This supports one of the paper's strongest practical claims: compositionally homogeneous subsets can improve anchor-conditioning precision beyond a single mixed model.

### Stage 2 compositional drift

| Target | Target horizon | Stage 1 horizon | Stage 2 horizon | Drift |
| --- | ---: | ---: | ---: | ---: |
| Rule of thirds | 0.333 | 0.333 | 0.538 | 0.205 |
| High sky | 0.150 | 0.150 | 0.392 | 0.242 |
| Low horizon | 0.600 | 0.600 | 0.488 | 0.112 |
| Center | 0.500 | 0.500 | 0.440 | 0.060 |

The paper's interpretation is that training-free refinement works best near the pretrained model's compositional prior and drifts more strongly when the requested horizon is far from that prior.

## Reproducibility and Environment

### Dataset

The manuscript describes training on the **Cropped-1901 Landscape Dataset** with 4,713 landscape images aggregated from multiple public sources and standardized to a 1.90:1 aspect ratio.

### Reported software stack

- PyTorch 2.x
- Hugging Face Diffusers
- PEFT
- safetensors
- OpenCV
- OpenCLIP for CLIP score
- LPIPS with AlexNet backbone

### Reported hardware

- NVIDIA A100 40 GB for Stage 1 training
- NVIDIA T4 16 GB for ablations and evaluation

### Installed Python dependencies

The repository ships a `requirements.txt` covering the main Python stack:

```bash
pip install -r requirements.txt
```

For a clean local setup:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Saved run metadata

The most reproducible experiment records in this repository are the per-run JSON files in `models/.../logs/` and `models/.../final_model/`. For example, the bundled forest and desert runs record:

- base model: `runwayml/stable-diffusion-v1-5`
- LoRA rank: `16`
- IP-Adapter rank: `8`
- mixed precision: `bf16`
- max train steps: `1301` for those ablation runs
- saved train curves and metric JSON files

## How to Use This Repository

### Quick navigation

- Main Stage 1 notebook: [notebooks/stage1/main_stage1.ipynb](notebooks/stage1/main_stage1.ipynb)
- Forest ablation notebook: [notebooks/stage1/ablation_forest.ipynb](notebooks/stage1/ablation_forest.ipynb)
- Desert ablation notebook: [notebooks/stage1/ablation_desert.ipynb](notebooks/stage1/ablation_desert.ipynb)
- Stage 2 notebook: [notebooks/stage2/inference_stage2.ipynb](notebooks/stage2/inference_stage2.ipynb)
- Anchor extraction utility: [scripts/anchor_extraction.py](scripts/anchor_extraction.py)

### Recommended workflow

1. Install the Python environment with `requirements.txt`.
2. Start with [notebooks/stage1/main_stage1.ipynb](notebooks/stage1/main_stage1.ipynb) for the main training implementation and experiment logic.
3. Use the forest and desert ablation notebooks for the category-specific experiments reported in the paper.
4. Use [notebooks/stage2/inference_stage2.ipynb](notebooks/stage2/inference_stage2.ipynb) for the high-resolution refinement pipeline.
5. Inspect the saved run folders in `models/` to compare configs, training summaries, curve plots, and qualitative test outputs.

### Minimal anchor extraction example

The anchor extractor is the most directly reusable standalone module in `scripts/`:

```python
from scripts.anchor_extraction import AnchorExtractor

extractor = AnchorExtractor()
anchor = extractor.extract_anchor("path/to/landscape.jpg")
print(anchor)
```

Expected output keys:

- `horizon_y`
- `horizon_conf`
- `avg_saliency`
- `fg_ratio`

### About `scripts/pipeline.py`

`scripts/pipeline.py` mirrors the paper pipeline conceptually, but it is not the canonical experiment implementation. It still contains placeholder logic for model loading, generation, and refinement. For paper-faithful reproduction, prefer the notebooks and saved run directories.

## Directory Structure

```text
.
|-- data/
|   `-- anchor_cache_9k.json
|-- docs/
|   `-- images/
|-- models/
|   |-- desert/
|   |   `-- stage1_ablation_desert_sd15_20260409_075018/
|   |       |-- checkpoints/
|   |       |   `-- checkpoint-001000/
|   |       |       |-- anchor_proj.pt
|   |       |       |-- ip_adapter.pt
|   |       |       `-- lora_weights/
|   |       |-- final_model/
|   |       |   |-- anchor_proj.pt
|   |       |   |-- ip_adapter.pt
|   |       |   |-- training_summary.json
|   |       |   `-- lora_weights/
|   |       |-- logs/
|   |       |   |-- train_config.json
|   |       |   |-- training_curves.png
|   |       |   `-- training_metrics.json
|   |       `-- test_outputs/
|   |           |-- testA_checkpoint_comparison.png
|   |           |-- testB_horizon_sweep.png
|   |           |-- testC_anchor_scale.png
|   |           |-- testD_prompt_variety.png
|   |           `-- testE_training_curves.png
|   `-- forest/
|       `-- stage1_ablation_forest_sd15_20260407_173621/
|           |-- checkpoints/
|           |-- final_model/
|           |-- logs/
|           `-- test_outputs/
|-- notebooks/
|   |-- stage1/
|   |   |-- ablation_desert.ipynb
|   |   |-- ablation_forest.ipynb
|   |   `-- main_stage1.ipynb
|   `-- stage2/
|       `-- inference_stage2.ipynb
|-- results/
|   `-- samples/
|       `-- IMAX_sample_results/
|           |-- archB_high_sky_153907_final.png
|           |-- stage1_1k.png
|           |-- stage4k.png
|           `-- stage4k_desert.png
|-- scripts/
|   |-- __init__.py
|   |-- anchor_extraction.py
|   |-- pipeline.py
|   `-- utils.py
|-- .gitignore
|-- LICENSE
|-- README.md
|-- requirements.txt
`-- stage1_LAST (1).zip
```

## Included Artifacts

### `data/`

- `anchor_cache_9k.json`
  - precomputed anchor cache shipped with the repository

### `models/`

- Category-specific run directories for **desert** and **forest**
- Saved adapter weights:
  - `anchor_proj.pt`
  - `ip_adapter.pt`
  - `lora_weights/adapter_config.json`
  - `lora_weights/adapter_model.safetensors`
- Per-run metadata:
  - `training_summary.json`
  - `logs/train_config.json`
  - `logs/training_metrics.json`
  - `logs/training_curves.png`
- Qualitative evaluation grids in `test_outputs/`

### `notebooks/`

- Full notebook implementation for the main Stage 1 workflow
- Separate forest and desert ablation notebooks
- Stage 2 high-resolution inference notebook

### `results/`

- Exported sample results for qualitative inspection
- Includes Stage 1 and 4K outputs used for README and paper-style presentation

### Legacy export

- `stage1_LAST (1).zip`
  - zipped notebook export present at repository root

## Known Limitations

- The repository is **not yet a fully packaged training library**. The notebooks are the ground-truth implementation path.
- `scripts/pipeline.py` is useful for orientation, but not sufficient by itself for exact paper reproduction.
- The current checkout does not appear to bundle every paper variant as a standalone run directory.
- Some exact dimensions and settings differ across manuscript text, notebook comments, and saved configs. For reproducibility, prefer the per-run JSON config files in `models/.../logs/`.
- Stage 2 is intentionally analyzed as a limitation point: the paper itself shows measurable drift for high-sky and rule-of-thirds horizon targets during training-free 4K refinement.

## Citation

If you use this repository, please cite the paper.

```bibtex
@inproceedings{anchor_conditioned_compositional_control_2026,
  title     = {Anchor-Conditioned Compositional Control for High-Resolution Landscape Image Generation},
  author    = {Anonymous},
  booktitle = {Proceedings of the International Conference on Computational Creativity (ICCC)},
  year      = {2026}
}
```

## License

This repository is released under the MIT License. See [LICENSE](LICENSE).
