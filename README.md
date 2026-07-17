# quantitative_abcm

ABCM（Alpha-Beta Co-mining，Alpha-Beta 协同挖掘）日频量化模型的非官方复现实现。

该项目使用个股日频价量序列，同时生成：

- `alpha` 因子：用于预测个股相对收益和截面排序。
- `beta` 风险因子：用于解释股票之间的共同波动，并约束方向性、相关性和跨期稳定性。

模型采用门控循环单元网络（GRU）编码个股时序，通过全连接分支输出 1 个 alpha 因子，通过非对称时空图神经网络（ASTGNN）输出 12 个 beta 风险因子。项目包含数据处理、模型、联合损失、时间块验证、LightGBM 基线、参数扫描、评价和结果汇总代码。

> 本仓库用于研究和复现，不构成投资建议。原始研究报告、训练数据、完整因子文件和模型权重未随仓库发布。

## 主要结果

实验使用 52 个 pickle 文件，覆盖 2005-01-04 至 2026-05-12，共 5,183 个交易日和约 1,606 万行数据。验证采用 5 个连续时间块，每个 ABCM 验证折至少训练一个完整轮次。

`h32`、`h48`、`h64` 是本项目的模型代号，数字表示 GRU 隐藏层维度。RankIC（秩信息系数）衡量因子排序与未来收益排序的相关性；ICIR（信息系数信息比率）衡量 RankIC 的稳定性；Top 组是 alpha 排名最高的 5%股票。

| 模型 | RankIC | ICIR | RankIC 胜率 | Top 年化近似值 | 截尾 Top 年化近似值 |
|---|---:|---:|---:|---:|---:|
| ABCM h32 | 11.85% | **0.926** | **82.72%** | 34.15% | 31.30% |
| ABCM h48 | 11.81% | 0.897 | 81.56% | **38.70%** | **34.93%** |
| ABCM h64 | **11.96%** | 0.875 | 81.05% | 38.57% | 34.83% |
| ABCM h32 PDF 损失 | 4.65% | 0.393 | 65.89% | 21.06% | 16.99% |
| LightGBM | 10.25% | 0.828 | 80.07% | **45.42%** | **40.90%** |
| PDF 数据集 1 | 12.69% | 0.960 | 86.63% | 34.51% | - |

结论概括：

- h32 的 alpha 综合指标最接近 PDF 参考值。
- h48 的 ABCM 验证收益最高。
- h64 的 RankIC 最高，收益接近 h48，跨验证折波动更低。
- PDF 基准损失提高了 beta 稳定性，同时 alpha 排序和收益较低。
- LightGBM 的 Top 收益高于 ABCM，ABCM h64 的 RankIC 高于 LightGBM。

详细结论见 [ABCM 模型复现结果汇报](docs/abcm_f52_final_report_2026-07-07.md)。

## 实现内容

- 按文件编号加载 `testdata_<N>.pkl`。
- 复权价格、12 个日频价量特征和前向收益标签。
- 按交易日进行中位数填补、5 倍 MAD 截尾和截面标准化。
- 防止跨长日期缺口构造 rolling 特征、标签和换手约束。
- GRU 时序编码器、alpha 多层感知机和非对称 ASTGNN beta 分支。
- MSE、R²残差、alpha 相关性、因子相关性和 beta 稳定性联合损失。
- 连续时间块交叉验证，训练日期按固定随机种子确定性打乱。
- RankIC、ICIR、胜率、命中率、Top 收益、long-short 收益、beta 自相关和滚动 R²评价。
- LightGBM tabular 基线、参数扫描、缺失运行补跑和候选比较。
- 运行摘要、模型状态哈希和候选级结果导出。

## 仓库结构

```text
abcm/                       数据、特征、模型、损失、评价和切分模块
configs/                    ABCM1 示例配置
scripts/                    训练、评价、扫描、基线和汇总入口
tests/                      单元测试和小型集成测试
docs/                       最终复现报告
results/                    已发布的候选比较 CSV
requirements.txt            Python 依赖
```

## 环境要求

- Python 3.10+
- Linux 推荐
- CUDA GPU 推荐；小规模检查可以使用 CPU
- 内存需求取决于数据规模和并行验证折数量

安装：

```bash
git clone git@github.com:leehm00/quantitative_abcm.git
cd quantitative_abcm

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

GPU 用户应根据本机 CUDA 版本安装对应的 PyTorch wheel，具体命令以 PyTorch 官方安装页面为准。

## 数据格式

默认数据目录为 `data/testdata/`，文件名格式为：

```text
data/testdata/testdata_0.pkl
data/testdata/testdata_1.pkl
...
```

每个 pickle 文件应包含以下字段：

| 字段 | 含义 |
|---|---|
| `S_INFO_WINDCODE` | 股票代码 |
| `TRADE_DT` | 交易日，格式 `YYYYMMDD` |
| `S_DQ_PRECLOSE` | 前收盘价 |
| `S_DQ_OPEN` | 开盘价 |
| `S_DQ_HIGH` | 最高价 |
| `S_DQ_LOW` | 最低价 |
| `S_DQ_CLOSE` | 收盘价 |
| `S_DQ_VOLUME` | 成交量 |
| `S_DQ_AMOUNT` | 成交额 |
| `S_DQ_AVGPRICE` | 均价 |
| `S_DQ_ADJFACTOR` | 复权因子 |

数据文件未包含在仓库中。使用其他字段体系时，需要先转换为上述列名或修改 `abcm/data.py` 和 `abcm/features.py`。

## 快速检查

先编辑 `configs/abcm1_daily_label_clip1.yaml`，确认：

- `data.root` 指向数据目录。
- `train.output_dir` 指向模型输出目录。
- `train.device` 为 `cuda`、`cuda:0` 或 `cpu`。

使用少量文件和步骤检查完整流程：

```bash
python scripts/train_abcm1.py \
  --config configs/abcm1_daily_label_clip1.yaml \
  --max-files 2 \
  --max-steps 5 \
  --stock-limit 128 \
  --device cpu
```

训练目录会生成：

- `checkpoints/best.pt`：训练结束时的模型权重。
- `training_log.csv`：逐步损失、epoch 和训练吞吐。
- `run_summary.json`：日期覆盖、运行时间、设备和显存信息。
- `validation_metrics.csv`：验证损失。
- `factors.csv`：导出的 alpha、beta 和标签。

评价导出的因子：

```bash
python scripts/evaluate_abcm1.py --factors-csv <run_dir>/factors.csv
```

## 完整候选示例

以下命令运行 h32 的五折完整训练。多个候选应使用不同的 `--output-root`，并分别设置隐藏层、学习率和损失权重。

```bash
python scripts/sweep_abcm1.py \
  --base-config configs/abcm1_daily_label_clip1.yaml \
  --output-root outputs/sweep_h32_cv \
  --hidden-dims 32 \
  --gru-layers 2 \
  --learning-rates 0.00035 \
  --stock-limits 1536 \
  --max-steps 2040 \
  --date-batch-sizes 2 \
  --lambda-mses 5.0 \
  --lambda-r2s 0.5 \
  --lambda-alpha-corrs 0.20 \
  --lambda-corrs 0.01 \
  --lambda-tos 0.01 \
  --validation-folds 0,1,2,3,4 \
  --dropouts 0.3 \
  --weight-decays 0.001 \
  --max-files 52 \
  --export-valid-dates -1 \
  --devices cuda:0,cuda:1,cuda:2 \
  --parallel 3
```

如果机器没有三块 GPU，请减少 `--devices` 和 `--parallel`。`max_steps=2040` 适用于本次数据规模；更换数据后，应根据每折训练日期数量和 `date_batch_size` 重新计算完整 epoch 所需步数。

## LightGBM 基线

```bash
python scripts/train_lightgbm_baseline.py \
  --config configs/abcm1_daily_label_clip1.yaml \
  --output-root outputs/baselines/lightgbm \
  --max-files 52 \
  --stock-limit 1536 \
  --validation-folds 0,1,2,3,4 \
  --export-valid-dates -1 \
  --sample-mode tabular \
  --n-estimators 300 \
  --learning-rate 0.03 \
  --num-leaves 31
```

## 汇总与比较

汇总单个 ABCM sweep：

```bash
python scripts/summarize_abcm_sweep.py \
  --sweep-root outputs/sweep_h32_cv
```

比较多个候选目录：

```bash
python scripts/compare_abcm_candidates.py \
  --output-dir outputs/global_candidate_comparison \
  --search-root outputs/sweep_h32_cv \
  --search-root outputs/baselines/lightgbm
```

生成指定候选的 alpha、beta、覆盖率和模型哈希汇总：

```bash
python scripts/summarize_full_epoch_results.py \
  --output-dir outputs/selected_results \
  --candidate h32=outputs/sweep_h32_cv
```

## 已发布结果

本次汇总结果位于：

```text
results/global_candidate_comparison_f52_full_epoch_20260717/
```

主要文件：

- `selected_candidate_summary.csv`：4 个 ABCM 候选的五折汇总。
- `selected_fold_metrics.csv`：20 个 ABCM 验证折的指标。
- `selected_model_manifest.csv`：原始运行中的模型路径、文件哈希和模型状态哈希。
- `global_candidate_leaderboard.csv`：候选级运行排名。
- `global_candidate_by_config.csv`：按配置聚合的 ABCM 与 LightGBM 比较。
- `stable_cv_clip_leaderboard.csv`：稳定五折且使用标签截尾的配置排名。
- `screening_clip_leaderboard.csv`：筛选实验排名。

CSV 中的绝对路径来自原始实验机器，仅作为审计记录。大型 checkpoint、完整 `factors.csv` 和训练数据未上传 GitHub。

## 测试

```bash
PYTHONPATH=. pytest -q
```

当前测试覆盖数据排序与复权、日期缺口、标签构造、张量形状、损失、评价指标、时间切分、LightGBM 样本、参数扫描和候选汇总。

## 结果口径与限制

- Top 年化近似值按 11 日标签均值乘以 `252/11` 计算。
- 相邻 11 日收益窗口存在重叠。
- 收益未扣除交易成本、冲击成本和成交限制。
- 五折结果参与了候选选择，不能代替独立前向测试。
- 模型输出用于研究评估，线上使用前应明确训练截止日、特征版本和可交易约束。

## License

本项目采用 [MIT License](LICENSE)。
