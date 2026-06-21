# 数据集来源说明

本仓库只提供代码，不提供数据文件。v8 实验使用的训练数据需要从公开数据服务下载，并在本地预处理为 QA 感知的积雪覆盖序列。

## 主要遥感数据

主要数据源为 NASA MODIS/Terra 每日积雪产品，通过 NASA AppEEARS 服务下载。

使用到的核心图层包括：

- `NDSI_Snow_Cover`：归一化差值积雪指数派生的积雪覆盖百分比。
- `NDSI_Snow_Cover_Basic_QA`：基础质量控制图层。
- `NDSI_Snow_Cover_Algorithm_Flags_QA`：算法标志质量控制图层。

下载脚本：

```powershell
python scripts/download_qa_data.py all
```

数据服务：

- NASA AppEEARS: https://appeears.earthdatacloud.nasa.gov/
- NASA Earthdata: https://www.earthdata.nasa.gov/
- MODIS Snow Cover 产品说明: https://nsidc.org/data/modis/data_summaries

## 辅助气象数据

辅助气象变量来自 NASA POWER，按研究区和日期下载，用作外部环境上下文。

下载脚本：

```powershell
python scripts/download_power_regions.py
```

数据服务：

- NASA POWER: https://power.larc.nasa.gov/

## 预处理结果

原始 MODIS 图层会被预处理为模型训练所需的数组和元数据，典型文件包括：

- `snow.npy`
- `strict_mask.npy`
- `lenient_mask.npy`
- `observation_age.npy`
- `spatial_prior.npy`
- `metadata.json`

预处理脚本：

```powershell
python scripts/preprocess_qa_dataset.py --raw-root data_qa_raw --output-root data_qa_v7_256 --image-size 256
```

这些文件属于数据集或派生数据，不应提交到 GitHub。

## 数据划分

v8 默认配置中的时间划分位于 `configs/qa_experiment_v8.yaml`：

- 训练/验证滚动折：`fold_1`、`fold_2`、`fold_3`
- 冻结测试集：`2024-01-01` 至 `2024-12-31`

请不要根据冻结测试集结果反向选择模型或调参。

## 引用与合规

使用数据时应在论文或报告中引用对应数据产品和服务平台。建议至少说明：

- MODIS/Terra snow-cover data were accessed through NASA AppEEARS.
- Meteorological auxiliary variables were accessed through NASA POWER.
- All raw and processed data are excluded from this repository because of size, reproducibility, and data-service policy considerations.

