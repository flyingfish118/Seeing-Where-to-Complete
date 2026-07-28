#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_t3ds_fdi_json.py

扫描 t3ds_pcn 数据目录并生成以 FDI（例如: 11,12...）为 taxonomy_id 的 JSON 文件。

用法示例：
    python generate_t3ds_fdi_json.py --root /path/to/t3ds_pcn --out dataset_fdi.json

输出格式：
[
  {
    "taxonomy_id": "11",
    "taxonomy_name": "11",
    "train": ["0EAKT1CU", ...],
    "test": [...],
    "val": [...]
  },
  ...
]

说明：
- 脚本只会在每个 split 下查找 partial/ 子目录（train/test/val）。
- 如果某个 split 或 partial 缺失，该 split 下将返回空列表。
- 为保证 JSON 可解析，输出为标准 JSON（没有尾随逗号）。
"""

import os
import json
import argparse
from collections import defaultdict


def collect_samples(root, splits=("train", "test", "val")):
    """收集 root/{split}/partial/{fdi}/{sample}/ 目录下的 sample 名称。

    返回: dict mapping fdi -> {"train":[], "test":[], "val":[]}
    """
    data = defaultdict(lambda: {s: [] for s in splits})

    for split in splits:
        partial_dir = os.path.join(root, split, 'partial')
        if not os.path.isdir(partial_dir):
            # 如果没有 partial 目录，跳过但保留空列表
            continue

        # 每个子目录（例如 11, 12, ...）代表一个 taxonomy_id
        for fdi in sorted(os.listdir(partial_dir)):
            fdi_path = os.path.join(partial_dir, fdi)
            if not os.path.isdir(fdi_path):
                continue

            # 每个样本通常以文件夹形式存在
            for sample in sorted(os.listdir(fdi_path)):
                sample_path = os.path.join(fdi_path, sample)
                if os.path.isdir(sample_path):
                    if sample not in data[fdi][split]:
                        data[fdi][split].append(sample)
                else:
                    # 也有可能直接是 pcd 文件（不常见），那就用文件名不带扩展
                    name, ext = os.path.splitext(sample)
                    if ext.lower() == '.pcd' and name:
                        if name not in data[fdi][split]:
                            data[fdi][split].append(name)

    return data


def to_list_of_dicts(mapping):
    """把 mapping 转换为用户期望的列表格式，并按 taxonomy_id 排序。"""
    out = []
    for fdi in sorted(mapping, key=lambda x: (len(x) != 2, x)):
        # 排序策略：尽量把双字符（如 '11','12'）与其它一致排序
        entry = {
            'taxonomy_id': fdi,
            'taxonomy_name': fdi,
            'train': mapping[fdi].get('train', []),
            'test': mapping[fdi].get('test', []),
            'val': mapping[fdi].get('val', []),
        }
        out.append(entry)
    return out


def main():
    parser = argparse.ArgumentParser(description='Generate dataset JSON by FDI (e.g. 11,12) from t3ds_pcn folder structure')
    parser.add_argument('--root', '-r', required=True, help='t3ds_pcn 根目录路径')
    parser.add_argument('--out', '-o', default='dataset_fdi.json', help='输出 JSON 文件路径')
    parser.add_argument('--splits', '-s', default='train,test', help='逗号分隔的 split 列表（默认: train,test）')
    parser.add_argument('--pretty', action='store_true', help='是否美化输出（缩进）')
    args = parser.parse_args()

    splits = [x.strip() for x in args.splits.split(',') if x.strip()]
    mapping = collect_samples(args.root, splits=tuple(splits))
    result = []

    # 如果用户自定义了 splits，to_list_of_dicts 里仍假定 train/test/val 字段，
    # 我们用映射中实际的 splits 来构建每个条目
    for fdi in sorted(mapping.keys(), key=lambda x: (len(x) != 2, x)):
        entry = {'taxonomy_id': fdi, 'taxonomy_name': fdi}
        for sp in splits:
            entry[sp] = mapping[fdi].get(sp, [])
        result.append(entry)

    # 保存 JSON
    with open(args.out, 'w', encoding='utf-8') as f:
        if args.pretty:
            json.dump(result, f, ensure_ascii=False, indent=2)
        else:
            json.dump(result, f, ensure_ascii=False, separators=(',', ':'))

    print(f'Wrote {len(result)} taxonomy entries to {args.out}')


if __name__ == '__main__':
    main()
