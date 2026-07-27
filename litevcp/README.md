# Visual--Geometric Prototype Predictors

This module implements the predictor family used in the paper:

- `DINOVGP` (compatibility class `DINOv3SetKD`) fuses frozen DINOv3 SCC features with a PointNet-style partial-cloud descriptor.
- `CVGP` (compatibility class `LiteDINOPrototypeStudent`) replaces DINO with a compact shared CNN. Frozen DINO tokens supervise only normalized per-view feature directions during training.
- The point-only and DINO view-only controls disable one branch of `DINOVGP`.

All current-route coordinate outputs are supervised directly by the dense `gt_missing` set. C-VGP does not imitate DINO-VGP coordinates, language-model output, or a D2P field. Historical SetKD code is retained only for loading older internal configurations and is not part of the paper method.

See the repository-level `README.md` for the complete two-stage workflow.
