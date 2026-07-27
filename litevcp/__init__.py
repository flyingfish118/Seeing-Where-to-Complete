"""MissingGT prototype-prediction components."""

from .model import DINOv3SetKD, LiteDINOPrototypeStudent, LiteVCPSetKD

# Paper-facing aliases. Legacy names remain importable for old checkpoints and
# configuration files; they do not denote extra methods.
DINOVGP = DINOv3SetKD
CVGP = LiteDINOPrototypeStudent

__all__ = [
    "DINOVGP",
    "CVGP",
    "DINOv3SetKD",
    "LiteDINOPrototypeStudent",
    "LiteVCPSetKD",
]
