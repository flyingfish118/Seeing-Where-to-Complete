import torch.utils.data as data
import numpy as np
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import data_transforms
from .io import IO
import random
import os
import json
from .build import DATASETS
from utils.logger import *


@DATASETS.register_module()
class Tooth_missing_points(data.Dataset):
    def __init__(self, config):
        self.partial_points_path = config.PARTIAL_POINTS_PATH
        self.complete_points_path = config.COMPLETE_POINTS_PATH
        self.missing_path = config.MISSING_PATH
        self.gt_missing_path = config.GT_MISSING_PATH
        self.category_file = config.CATEGORY_FILE_PATH
        self.npoints = config.N_POINTS
        self.subset = config.subset
        self.FDI_11 = config.FDI_11
        # A strict geometry-only condition must neither read nor pass an
        # exported prototype.  Keep the historical behavior as the default.
        self.load_missing = bool(getattr(config, 'LOAD_MISSING', True))
        self.taxonomy_ids = set(str(value) for value in getattr(config, 'TAXONOMY_IDS', []))
        # The historical tooth set uses eight training renderings. Derived
        # clinical mask sets can explicitly request one deterministic view.
        self.train_renderings = int(getattr(config, 'TRAIN_RENDERINGS', 8))
        self.gt_missing_npoints = getattr(config, 'GT_MISSING_N_POINTS', None)

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())
            if config.FDI_11:
                self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_id'] == '11']
            if self.taxonomy_ids:
                self.dataset_categories = [dc for dc in self.dataset_categories if str(dc['taxonomy_id']) in self.taxonomy_ids]
            if not self.dataset_categories:
                raise ValueError('No dataset categories remain after taxonomy filtering')

        self.n_renderings = self.train_renderings if self.subset == 'train' else 1
        self.file_list = self._get_file_list(self.subset, self.n_renderings)
        self.transforms = self._get_transforms(self.subset)

    def _get_transforms(self, subset):
        gt_missing_transform = []
        if self.gt_missing_npoints is not None:
            gt_missing_transform = [{
                'callback': 'RandomSamplePoints',
                'parameters': {'n_points': int(self.gt_missing_npoints)},
                'objects': ['gt_missing']
            }]
        if subset == 'train':
            transforms = [
                {   # ① partial 采样到 2048
                    'callback': 'RandomSamplePoints',
                    'parameters': {
                        'n_points': 2048
                    },
                    'objects': ['partial']
                },
                {   # The frozen VGP prototype has a fixed cardinality of 30.
                    'callback': 'RandomSamplePoints',
                    'parameters': {
                        'n_points': 30
                    },
                    'objects': ['missing']
                },
                {   # ③ 镜像（如果你还想要）
                    'callback': 'RandomMirrorPoints',
                    'objects': ['partial', 'gt', 'missing', "gt_missing"]
                },
                {   # ④ 转成 Tensor
                    'callback': 'ToTensor',
                    'objects': ['partial', 'gt', 'missing', "gt_missing"]
                }
            ]
            return data_transforms.Compose(transforms[:-1] + gt_missing_transform + transforms[-1:])
        else:
            transforms = [
                {   # 验证集也保持 partial=2048
                    'callback': 'RandomSamplePoints',
                    'parameters': {
                        'n_points': 2048
                    },
                    'objects': ['partial']
                },
                {
                    'callback': 'RandomSamplePoints',
                    'parameters': {
                        'n_points': 30
                    },
                    'objects': ['missing']
                },
                {
                    'callback': 'ToTensor',
                    'objects': ['partial', 'gt', 'missing', "gt_missing"]
                }
            ]
            return data_transforms.Compose(transforms[:-1] + gt_missing_transform + transforms[-1:])


    def _get_file_list(self, subset, n_renderings=1):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:
            print_log('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']), logger='PCNDATASET')
            samples = dc[subset]

            for s in samples:
                file_list.append({
                    'taxonomy_id':
                    dc['taxonomy_id'],
                    'model_id':
                    s,
                    'partial_path': [
                        self.partial_points_path % (subset, dc['taxonomy_id'], s, i)
                        for i in range(n_renderings)
                    ],
                    'gt_path':
                    self.complete_points_path % (subset, dc['taxonomy_id'], s),
                    'missing_path': [
                        self.missing_path  % (subset, dc['taxonomy_id'], s, i)
                        for i in range(n_renderings)
                    ],
                    'gt_missing_path': [
                        self.gt_missing_path  % (subset, dc['taxonomy_id'], s, i)
                        for i in range(n_renderings)
                    ]                    
                    
                })

        print_log('Complete collecting files of the dataset. Total files: %d' % len(file_list), logger='PCNDATASET')
        return file_list

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        rand_idx = random.randint(0, self.n_renderings - 1) if self.subset=='train' else 0

        for ri in ['partial', 'gt', 'missing', 'gt_missing']:
            if ri == 'missing' and not self.load_missing:
                # The common loader contract still returns a tensor, while
                # prototype_source=none prevents this placeholder from being
                # consumed by the completion model.
                data[ri] = np.zeros((30, 3), dtype=np.float32)
                continue
            file_path = sample['%s_path' % ri]
            if isinstance(file_path, list):
                file_path = file_path[rand_idx]
            data[ri] = IO.get(file_path).astype(np.float32)   # numpy

        assert data['gt'].shape[0] == self.npoints

        if self.transforms is not None:
            data = self.transforms(data)

        # Give DataLoader tensors independent, resizable storage.
        import torch
        for k in ['partial', 'gt', 'missing', 'gt_missing']:
            if isinstance(data[k], torch.Tensor):
                data[k] = data[k].detach().clone().to(dtype=torch.float32)
            else:
                data[k] = torch.as_tensor(data[k], dtype=torch.float32).clone()

        return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['gt']), data['missing'], data['gt_missing']


    def __len__(self):
        return len(self.file_list)
