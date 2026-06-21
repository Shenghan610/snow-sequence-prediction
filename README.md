# QA 感知积雪覆盖预测 v8

本仓库包含 v8 实验所需的代码、配置、测试脚本和第三方基线适配代码，不包含任何原始数据、预处理数据、模型权重或实验输出。

v8 的目标是在 Transformer-CANN-Lyapunov 全局动力学分支基础上，引入高分辨率边界修正分支，用于提升雪线边界、SSIM、snow F1 和局部空间细节，同时保留覆盖率变化轨迹和 Lyapunov 风格稳定性约束。

## 核心内容

- `configs/qa_experiment_v8.yaml`：v8 主配置。
- `run_qa_experiment.py`：训练、验证、测试和多模型实验入口。
- `src/snow_attractor/transformer_cann.py`：Transformer-CANN-Lyapunov v8 主模型与边界修正模块。
- `src/snow_attractor/losses.py`：覆盖率、Lyapunov、边界和雪盖相关损失。
- `src/snow_attractor/data.py`：QA 感知序列数据加载。
- `scripts/download_qa_data.py`：通过 NASA AppEEARS 请求 MODIS QA 数据。
- `scripts/download_power_regions.py`：下载 NASA POWER 辅助气象变量。
- `scripts/preprocess_qa_dataset.py`：将原始 MODIS QA 图层重投影、裁剪并生成训练用数组。
- `scripts/run_qa_pipeline.py`：端到端 QA 实验流水线。
- `vendor/`：固定版本的 SimVPv2、SwinLSTM、VMRNN 第三方基线源码。

## 环境安装

建议使用 Python 3.9 或以上版本，并在独立虚拟环境中安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果使用 CUDA，请根据本机驱动和 CUDA 版本安装匹配的 PyTorch 版本。`requirements.txt` 中的版本是本实验环境使用过的参考版本。

## 数据准备

数据不随仓库上传，需要自行下载和预处理。默认配置期望数据位于：

```text
F:/new_experiment/data_qa_v7_256
```

可以修改 `configs/qa_experiment_v8.yaml` 中的 `data.root` 指向自己的数据目录。

典型流程如下：

```powershell
python scripts/download_qa_data.py all
python scripts/download_power_regions.py
python scripts/preprocess_qa_dataset.py --raw-root data_qa_raw --output-root data_qa_v7_256 --image-size 256
```

具体数据来源见 [DATASETS.md](DATASETS.md)。

## 运行 v8 实验

单次运行：

```powershell
python run_qa_experiment.py --config configs/qa_experiment_v8.yaml --model transformer_cann_lyapunov_boundary --split fold_3
```

流水线运行：

```powershell
python scripts/run_qa_pipeline.py --config configs/qa_experiment_v8.yaml
```

输出默认写入 `artifacts/qa_experiment_v8`。该目录被 `.gitignore` 排除，不应提交到仓库。

## v8 模型摘要

v8 使用两条互补路径：

```text
历史 snow/QA/context 序列
        |
        |-- Transformer-CANN-Lyapunov 全局分支
        |      - coverage/change/season 坐标
        |      - temporal transformer
        |      - window spatial transformer
        |      - Lyapunov energy 软约束
        |      -> global_prediction
        |
        |-- 高分辨率边界修正分支
               - last snow map
               - global prediction
               - residual/change cues
               - spatial prior/context
               -> boundary_residual

final_prediction = coverage-preserving calibration(global_prediction + boundary_residual)
```

## 重要说明

- 本仓库不包含 `data_qa*`、`artifacts`、`.pt/.pth/.ckpt`、`.npy/.npz`、GeoTIFF 或压缩数据包。
- 第三方基线源码保留在 `vendor/`，但移除了各自的 `.git` 历史和非必要数据产物。
- 如果只复现 v8 主模型，不运行官方第三方基线，可以忽略 `vendor/`。
- 若使用 NASA 或第三方数据，请遵守对应数据源的许可、引用和下载政策。

## 测试

可运行关键测试：

```powershell
pytest tests/test_v7_transformer_cann.py tests/test_benchmark_models.py tests/test_qa_pipeline.py
```

