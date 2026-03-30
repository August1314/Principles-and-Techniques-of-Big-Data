# 大数据原理与技术 作业5：股票价格预测

本作业基于同一份 Apple 股票历史收盘价数据，分别使用 `ARIMA` 与 `LSTM` 完成：

- 测试集价格预测
- `MAE` / `RMSE` 误差对比
- 未来 7 个交易日价格预测
- 自动生成图表、结果汇总和实验报告 PDF

## 目录结构

- `data/stock_history.csv`：实验使用的本地股票历史数据
- `src/load_data.py`：CSV 加载与标准化
- `src/preprocess.py`：时间切分、滑动窗口、缩放处理
- `src/metrics.py`：误差指标
- `src/train_arima.py`：ARIMA 训练与预测
- `src/train_lstm.py`：LSTM 训练与预测
- `src/visualize.py`：图表生成
- `src/run_all.py`：一键运行主流程
- `tests/`：单元测试与烟雾测试
- `outputs/`：模型输出、CSV、图表、报告表格
- `report.tex` / `report.pdf`：实验报告源码与 PDF

## 环境

```bash
conda run -n ml python -m pip install -r requirements.txt
```

## 运行方式

运行测试：

```bash
conda run -n ml python -m unittest discover -s tests -v
```

执行完整实验：

```bash
conda run -n ml python src/run_all.py
```

生成 PDF 报告：

```bash
cd "/Users/lianglihang/Downloads/Principles-and-Techniques-of-Big-Data/作业5"
xelatex report.tex
xelatex report.tex
```

## 输出说明

运行完成后，`outputs/` 下会生成：

- `metrics_summary.json`
- `metrics_summary.md`
- `arima_test_prediction.csv`
- `lstm_test_prediction.csv`
- `future_7days_predictions.csv`
- `report_metrics_table.tex`
- `report_future_table.tex`
- `figures/raw_close_series.png`
- `figures/train_test_split.png`
- `figures/arima_vs_actual.png`
- `figures/lstm_vs_actual.png`
- `figures/model_comparison.png`
- `figures/error_bar_chart.png`
- `figures/future_7days_forecast.png`

## 数据来源

实验数据由 Plotly 公开示例 CSV 提取而来，原始文件保存在 `data/source_finance_charts_apple.csv`，建模时使用标准化后的 `data/stock_history.csv`。
