# Anonymous VGP Completion Reproduction

This package accompanies the anonymous submission **Seeing Where to Complete: Multimodal Prototypes against Missing-Region Dilution**. It contains the public Teeth3DS data-construction path, Spatial Coordinate Colorization (SCC), visual--geometric prototype prediction, fixed-prototype export, and prototype-conditioned SymmCompletion training.

The paper names the two multimodal predictors:

- **DINO-VGP**: a frozen DINOv3 view encoder plus a PointNet-style geometric branch.
- **C-VGP**: a compact shared SCC encoder plus a PointNet-style geometric branch. During training only, its normalized per-view visual features are aligned to frozen DINO features.

Both predictors output a 30-point unordered coordinate prototype. The dense true missing support is the only coordinate target. C-VGP does not distill DINO-VGP coordinates or D2P fields, and it does not load DINO at deployment.

## Scope

```text
authorized Teeth3DS source + labels + split
  -> partial / complete / true missing support
  -> three SCC views (-y, +y, +z)
  -> DINO-VGP or C-VGP
  -> fixed case-aligned prototype
  -> SymmCompletion + D2P modulation + MR supervision
```

No weights, raw or derived dataset files, SCC images, cached features, exported prototypes, patient data, logs, predictions, or machine-specific paths are included. The private dental dataset used in the paper is not released. The ShapeNet-55 experiment follows the protocol documented in the supplementary material but is outside this minimal Teeth3DS reproduction package.

## Compatibility note

The internal Python package and several configuration filenames retain the historical strings `litevcp`, `litedino`, and `dinov3` so that existing checkpoints and scripts remain loadable. They map to the paper terminology as follows:

| Paper name | Internal implementation identifier |
|---|---|
| DINO-VGP | `DINOv3SetKD`, `model.type: dinov3` |
| C-VGP | `LiteDINOPrototypeStudent`, `model.type: litedino` |
| Point-only ablation | `DINOv3SetKD(use_view_encoder=False)` |
| DINO view-only ablation | `DINOv3SetKD(use_point_encoder=False)` |

The package exports the clearer Python aliases `DINOVGP` and `CVGP`. Historical identifiers are implementation compatibility names, not additional methods.

## Installation

Use Linux, an NVIDIA GPU, and a CUDA-compatible PyTorch build. One example is:

```bash
conda create -n vgp_completion python=3.11 -y
conda activate vgp_completion
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
cd extensions/Pointnet2/pointnet2
python setup.py install
cd ../../..
export CHAMFER_BACKEND=torch
export CHAMFER_TORCH_CHUNK_SIZE=512
```

Copy `env.example` to `.env`, replace each placeholder with an authorized local path, and run `source .env`. The exact PyTorch/CUDA versions should match the local compiler. `CHAMFER_BACKEND=torch` selects the included exact PyTorch Chamfer implementation; PointNet++ still requires its CUDA extension.

## 1. Construct the public Teeth3DS benchmark

Obtain Teeth3DS under its original terms. The expected authorized input layout is:

```text
$RAW_T3DS_ROOT/
  upper/<case>/<case>_upper.obj
  upper/<case>/<case>_upper.json
  lower/<case>/<case>_lower.obj
  lower/<case>/<case>_lower.json
```

The label JSON must contain a `labels` array aligned one-to-one with OBJ vertices. A split JSON contains `upper` and `lower` lists with `train` and `test` case identifiers.

```bash
python data_processing/build_t3ds_completion_dataset.py \
  --split-json /absolute/path/to/tooth_splits.json \
  --raw-root "$RAW_T3DS_ROOT" \
  --output-root "$DENTAL_DATA_ROOT" \
  --seed 42 --train-variants 8

python data_processing/generate_category_json.py \
  --root "$DENTAL_DATA_ROOT" \
  --out data/Tooth/Tooth.json --splits train,test --pretty
```

The builder forms a local patch from the target tooth, two nearest neighboring teeth, and local gingiva, normalizes it to a common local frame, and removes one contiguous target region. Its output has the following structure:

```text
$DENTAL_DATA_ROOT/
  train/{partial,gt,gt_missing}/11/...
  test/{partial,gt,gt_missing}/11/...
```

`gt_missing` is used only for prototype-coordinate supervision, the missing-region (MR) loss, and evaluation. It is never a deployment input.

## 2. Render SCC views

SCC encodes every normalized point `p=(x,y,z)` as `RGB=clip((p+1)/2,0,1)` and renders the partial cloud from three fixed cameras. It is a deterministic image representation of the partial geometry, not a clinical photograph.

```bash
python data_processing/render_scc_views.py \
  --pcd-root "$DENTAL_DATA_ROOT" \
  --image-root "$DENTAL_SCC_ROOT" --width 640 --height 640
```

The renderer refuses to overwrite an existing image root so that different rendering conventions cannot be mixed accidentally.

## 3. Train prototype predictors

Train DINO-VGP. Obtain DINOv3 weights separately from their official source and set `DINO_V3_CHECKPOINT`.

```bash
python -m litevcp.train \
  --config cfgs/Tooth_models/DINO_VGP.yaml \
  --exp_name dino_vgp --output_root experiments/DINO_VGP
```

Cache only DINO per-view tokens, then train C-VGP. The legacy configuration filename is kept for compatibility.

```bash
python -m litevcp.cache_dino_reference \
  --config cfgs/Tooth_models/DINO_VGP.yaml \
  --ckpt experiments/DINO_VGP/dino_vgp/ckpt-best.pth \
  --output-root "$DINO_REFERENCE_ROOT" --tokens-only

python -m litevcp.train \
  --config cfgs/Tooth_models/C_VGP.yaml \
  --exp_name c_vgp --output_root experiments/C_VGP
```

The point-only and DINO view-only ablations use `PointOnly_MissingGT.yaml` and `DINOv3_SCCOnly_MissingGT.yaml`, respectively.

## 4. Export fixed prototypes

Export C-VGP predictions to a separate root; never overwrite `gt_missing`.

```bash
python -m litevcp.export_prototypes \
  --config cfgs/Tooth_models/C_VGP.yaml \
  --ckpt experiments/C_VGP/c_vgp/ckpt-best.pth \
  --output_root "$CVGP_PROTOTYPE_ROOT" --splits train test
```

The exported hierarchy is `split/missing/11/<case>/<view>.pcd`. DINO-VGP can be exported with the same command and its own configuration/checkpoint.

## 5. Train and evaluate completion

The supplied paper-facing SymmCompletion configuration reads the C-VGP prototype root directly.

```bash
python main.py \
  --config cfgs/Tooth_models/SymmCompletion_C_VGP.yaml \
  --exp_name symm_c_vgp --deterministic

python main.py \
  --config cfgs/Tooth_models/SymmCompletion_C_VGP.yaml \
  --test \
  --ckpts experiments/SymmCompletion_C_VGP/Tooth_models/symm_c_vgp/ckpt-best.pth \
  --exp_name symm_c_vgp_k10 --deterministic
```

The default completion configuration reports CD-$L_2$ and CDMiss@10. `SymmCompletion_geometry_only_missinggt.yaml` is the strict global-loss-only control. The companion `_k1` file is retained for audit use but CDMiss@10 is the paper's reported local metric.

## Release and licensing

Third-party notices are in `NOTICE.md`. Teeth3DS, DINOv3, PyTorch, Open3D, `timm`, and PointNet++ remain subject to their own licenses and terms. This anonymous artifact does not grant redistribution rights for external data or weights. See `OPEN_SOURCE_CHECKLIST.md` before any later public release.
