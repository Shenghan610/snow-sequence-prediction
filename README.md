# Snow Sequence Prediction

本项目用于遥感积雪覆盖序列预测。模型读取连续多日的积雪覆盖栅格影像，并结合日期、季节先验、覆盖率统计和外部气象/地形特征，预测目标日期的积雪覆盖状态。

主训练入口是 `main.py`。在 PyCharm 中直接运行 `main.py` 即可开始训练，不需要额外填写命令行参数。

## 仓库文件说明

### `main.py`

轻量训练入口。它从 `snow_prediction` 包导入公开接口，并调用
`run_training()`。原有的 `python main.py` 和 PyCharm 运行方式保持不变。

### `snow_prediction/`

训练代码按职责拆分后的核心包：

- `config.py`：命令行参数、随机种子、EMA 和权重加载。
- `data.py`：外部特征读取、TIF 预处理、时间窗口和空间先验。
- `model.py`：空间编码器、动力学注意力和积雪热力图预测模型。
- `evaluation.py`：损失函数、基线、指标计算和结果可视化。
- `training.py`：数据划分、训练循环、验证、早停和模型保存。
- `__init__.py`：统一导出项目公共接口。

### 部署文件

- `inference.py`：可复用的权重加载和预测服务。
- `predict.py`：命令行预测入口。
- `api.py`：FastAPI HTTP 服务。
- `tests/test_inference.py`：模型、GeoTIFF 输出和真实权重集成测试。

### `download_external_features.py`

外部气象/地形表格特征生成脚本。

它会调用 NASA POWER daily point API，在研究区域内构建 `5 x 5` 经纬度网格，并下载 2020-2024 年逐日气象变量，包括：

- 2 米气温、最高气温、最低气温、露点温度
- 相对湿度
- 降水
- 10 米风速
- 短波和长波辐射

脚本还会根据网格点高程估算区域地形统计量，例如平均高程、坡度、坡向和地形辐射指数。

生成文件位于 `ExternalClimateTerrain/`：

- `external_daily_features.csv`：训练时实际读取的逐日外部特征表。
- `external_grid_points.csv`：采样网格点经纬度和高程。
- `terrain_summary.csv`：区域地形统计摘要。
- `power_grid_daily_raw/*.json`：NASA POWER 原始返回数据缓存。

### `gee_multisource_export.js`

Google Earth Engine Code Editor 脚本，用于从云端数据源生成多源遥感/地理影像。

脚本使用的数据源包括：

- `MODIS/061/MOD10A1`：MODIS 逐日积雪覆盖。
- `ECMWF/ERA5_LAND/DAILY_AGGR`：ERA5-Land 逐日气象数据。
- `USGS/SRTMGL1_003`：SRTM 地形高程数据。

它会将积雪、气象和地形变量融合为逐日 GeoTIFF，并导出到 Google Drive。该脚本主要用于复现或扩展原始地理数据生成流程；当前 `main.py` 训练默认读取的是已经整理好的 `Ali_SnowData/` 和 `ExternalClimateTerrain/external_daily_features.csv`。

### `snow_sequence_training_datasets.zip`

训练依赖数据压缩包，使用 Git LFS 上传。

压缩包包含：

- `Ali_SnowData/`：积雪覆盖 TIF 序列，是模型的主要影像输入。
- `ExternalClimateTerrain/`：外部气象/地形特征及 NASA POWER 原始 JSON 缓存。

解压后应得到如下目录结构：

```text
Ali_SnowData/
ExternalClimateTerrain/
main.py
```

### `.gitattributes`

Git LFS 配置文件。

其中声明 `snow_sequence_training_datasets.zip` 由 Git LFS 管理，避免将数百 MB 的数据包作为普通 Git 文件提交。

### `.gitignore`

忽略本地缓存、模型权重、训练日志、图片结果和解压后的数据目录。

常见被忽略文件包括：

- `__pycache__/`
- `.idea/`
- `runs/`
- `*.png`
- `*.pth`
- `training_history.csv`
- `Ali_SnowData/`
- `ExternalClimateTerrain/`

这些文件大多是本地实验产物，或者已经通过压缩包单独提供。

### 训练生成文件

运行 `main.py` 后，项目根目录可能生成：

- `best_highres_global_snow_model.pth`
- `highres_global_snow_model.pth`
- `training_history.csv`
- `Figure_1.png`

这些是训练输出，不作为普通代码更新上传。

## 数据依赖

训练至少需要：

```text
Ali_SnowData/
ExternalClimateTerrain/external_daily_features.csv
```

如果已经从 GitHub 拉取了 LFS 数据包，可以在项目根目录解压：

```powershell
Expand-Archive snow_sequence_training_datasets.zip -DestinationPath .
```

如果没有安装 Git LFS，需要先安装并启用：

```powershell
git lfs install
```

然后重新拉取仓库或执行：

```powershell
git lfs pull
```

## 完整复现流程

仓库通过 Git LFS 提供训练数据压缩包和最佳模型权重：

- `snow_sequence_training_datasets.zip`：完整训练数据。
- `best_highres_snow_heatmap_model.pth`：最佳热力图预测权重。

推荐使用 Python 3.9 或更高版本。Windows PowerShell 复现步骤：

```powershell
git clone https://github.com/Shenghan610/snow-sequence-prediction.git
cd snow-sequence-prediction
git lfs install
git lfs pull

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

Expand-Archive snow_sequence_training_datasets.zip -DestinationPath .
```

解压后应确认以下资源存在：

```text
Ali_SnowData/
ExternalClimateTerrain/external_daily_features.csv
best_highres_snow_heatmap_model.pth
```

文件 SHA-256：

```text
snow_sequence_training_datasets.zip
AFCB7AE50261F487C58BDAAC0402B46CAA9BDBF342FDFA13D11795186D9914EC

best_highres_snow_heatmap_model.pth
F1B6D3B63C41C19A8F23BD31F6CF8E3F16E723313701312A6BE0EA7232569F0A
```

校验文件：

```powershell
Get-FileHash snow_sequence_training_datasets.zip -Algorithm SHA256
Get-FileHash best_highres_snow_heatmap_model.pth -Algorithm SHA256
```

使用最佳权重预测：

```powershell
python predict.py --target-date 2025-01-01 --device auto
```

从头训练：

```powershell
python main.py
```

运行轻量测试：

```powershell
python -m unittest discover -s tests -v
```

运行真实数据和最佳权重集成测试：

```powershell
$env:RUN_MODEL_INTEGRATION_TEST="1"
python -m unittest discover -s tests -v
```

## 模型结构

核心模型为 `DataDrivenSnowPredictor`，主要由四部分组成。

### 1. 数据与先验特征

`AliSnowDatasetRAM` 会将 TIF 影像预加载到内存，并构建多类辅助特征：

- 影像覆盖率序列
- 空间标准差
- 积雪像元比例
- 强积雪像元比例
- 短期变化量与加速度
- 日期和季节周期编码
- 目标日气候态覆盖率
- 外部气象/地形特征

同时，程序会从历史影像中计算两类空间先验：

- 长期平均积雪分布图
- 积雪变化活跃度图

### 2. 空间编码器

`PriorSpatialEncoder` 使用卷积网络提取每一天影像的空间表示。输入不仅包含原始影像通道，还拼接了空间先验通道。

编码器会输出多类 token：

- 全局平均 token
- 先验加权 token
- 局部网格池化 token

### 3. 时序动力学模块

模型使用双向 GRU 编码覆盖率、变化量、季节特征等标量序列。同时，空间 token 会加入时间位置编码、token 类型编码和季节投影。

`PriorDynamicalSelfAttention` 会对时空 token 做多步自注意力迭代，并使用空间变化先验计算内部能量约束，鼓励模型学习更稳定的积雪演化表示。

### 4. 多基线候选与门控残差

预测头会构造多个可解释基线候选：

- 前一天覆盖率
- 最近窗口均值
- 最近窗口趋势外推
- 目标日气候态覆盖率
- 气候异常延续

模型先学习这些候选基线的融合权重，得到基础预测值。随后，残差分支拆成两个部分：

- `residual_direction_head`：学习修正方向。
- `residual_gate_head`：学习修正门控强度。

最终修正量为：

```text
delta = max_delta * gate * direction
```

这种设计保留了模型对复杂变化的修正能力，同时限制后期训练中残差过大带来的验证集崩落。

## 当前默认训练配置

`main.py` 中已经写入默认实验配置，适合直接在 PyCharm 中运行。

核心参数包括：

```text
epochs = 140
seq_len = 21
lr = 0.0005
weight_decay = 0.0003
d_model = 160
iterations = 8
base_window = 7
max_delta = 0.40
hidden_dropout = 0.10
feature_dropout = 0.0
head_dropout = 0.10
residual_l1_weight = 0.003
residual_gate_bias = -0.8
patience = 25
ema = False
```

如果显存压力较大，优先降低 `batch_size` 或 `img_size`。

## 本次补充更新

本次补充上传了外部地理/气象特征生成相关脚本，并完善了 README：

- 新增 `download_external_features.py`，用于生成 NASA POWER 外部特征表。
- 新增 `gee_multisource_export.js`，用于在 Google Earth Engine 中导出多源遥感/地理影像。
- README 新增“仓库文件说明”，解释每个代码文件、数据包和配置文件的作用。

## 运行方式

1. 克隆仓库并拉取 Git LFS 数据。
2. 解压 `snow_sequence_training_datasets.zip`。
3. 在 PyCharm 中打开项目。
4. 选择并运行 `main.py`。

程序会自动读取默认参数、加载数据、训练模型、保存最佳权重并生成训练历史。

## 模型部署与预测

部署使用 `best_highres_snow_heatmap_model.pth`，不会重新训练模型。

命令行预测：

```powershell
python predict.py
```

指定目标日期和计算设备：

```powershell
python predict.py --target-date 2025-01-01 --device cuda
```

结果默认保存在 `predictions/`，包括：

- `SnowPrediction_日期.tif`：保留原始地理坐标的 GeoTIFF。
- `SnowPrediction_日期.png`：热力图预览。
- `SnowPrediction_日期.json`：覆盖率、输入日期和模型输出摘要。

启动 HTTP API：

```powershell
pip install -r requirements-deploy.txt
uvicorn api:app --host 0.0.0.0 --port 8000
```

接口：

- `GET /health`：检查模型和数据是否加载成功。
- `POST /predict?target_date=2025-01-01`：运行预测。
- `GET /docs`：打开 FastAPI 自动接口文档。

运行单元测试：

```powershell
python -m unittest discover -s tests -v
```

运行包含真实最佳权重的集成测试：

```powershell
$env:RUN_MODEL_INTEGRATION_TEST="1"
python -m unittest discover -s tests -v
```
