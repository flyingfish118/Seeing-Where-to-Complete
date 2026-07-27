# Checklist for a Later Public Release

The anonymous artifact already excludes weights, datasets, rendered images, feature caches, logs, predictions, and compiled binaries. Before publishing it as a permanent public repository:

1. Confirm that the Teeth3DS terms permit publication of the data-processing code, split description, and paper figures. Do not redistribute original or derived Teeth3DS files in this repository.
2. Review the licenses of DINOv3, PyTorch, Open3D, `timm`, and PointNet++. Users must obtain DINOv3 weights separately from the official source.
3. Select a project-level license only after the third-party compatibility review. The PointNet++ extension retains its bundled MIT license.
4. Repeat a secret/anonymity scan for absolute paths, accounts, tokens, patient identifiers, data files, weights, and experiment artifacts.
5. In a clean environment, compile PointNet++, run every `--help` entry point, and execute one end-to-end smoke test on authorized miniature data.
