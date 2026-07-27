# 数据划分

将授权获取的完整 T3DS 派生数据集划分保存为 `Tooth.json`。格式为 JSON 数组，每一项包含 `taxonomy_id`、`taxonomy_name`、`train` 与 `test`。当前模板只保留 FDI 牙位 `11`，与论文主实验一致。

用 `data_processing/generate_category_json.py` 可从已处理的数据目录自动生成完整类别索引。例如：

```bash
python data_processing/generate_category_json.py \
  --root "$DENTAL_DATA_ROOT" \
  --out data/Tooth/Tooth.json \
  --splits train,test --pretty
```

原始 T3DS 文件、派生 PCD、SCC 图像、`gt_missing` 和学生原型均不随本仓库发布。请确认数据许可后自行获取和处理。
