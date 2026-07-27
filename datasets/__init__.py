from .build import build_dataset_from_cfg

# The release intentionally ships only the dental MissingGT dataset.
import datasets.ToothDataset_missing_points
