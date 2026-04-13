# 大数据原理与技术 — 作业4（电影评论情感分类）

## 作业要求（来自 `作业4要求.jpg`）

- 使用 IMDB 电影评论数据集（或其他数据）
- 使用 TF-IDF 进行特征提取，并用逻辑回归/SVM/RNN 实现分类
- 对比并分析准确率差异
- 提交：代码 + 简单报告（2 页以上）

## 本实现方案

- 传统机器学习：`TF-IDF + Logistic Regression`
- 传统机器学习：`TF-IDF + Linear SVM`
- 深度学习对照组：`Embedding + LSTM + Linear`

说明：按作业要求，TF-IDF 用于 LR/SVM；RNN 使用词序列嵌入输入，作为深度学习对照模型。

## 目录结构

- `src/load_data.py`：IMDB 自动下载、读取、清洗
- `src/features.py`：TF-IDF 特征构建
- `src/train_lr.py`：逻辑回归训练与评估
- `src/train_svm.py`：线性 SVM 训练与评估
- `src/train_rnn.py`：RNN（LSTM）训练与评估
- `src/run_all.py`：一键运行全部实验并导出结果
- `tests/`：单元测试与烟雾集成测试
- `outputs/`：指标结果输出目录
- `23336128-梁力航-作业4.pdf`：提交用实验报告 PDF
- `23336128-梁力航-作业4.md`：实验报告 Markdown 源文件
- `23336128-梁力航-作业4.tex`：实验报告 LaTeX 源文件

## 环境（Anaconda `ml`）

```bash
conda run -n ml python -m pip install -r requirements.txt
```

## 运行方式

### 1) 运行测试

```bash
conda run -n ml python -m unittest discover -s tests -v
```

### 2) 全量运行（默认 25k train + 25k test）

```bash
conda run -n ml python src/run_all.py
```

第一次运行若未检测到 `data/aclImdb`，会自动下载并解压官方 IMDB 数据集。

### 3) 快速烟雾运行（每类抽样）

```bash
conda run -n ml python src/run_all.py --sample-per-class 200
```

## 输出结果

运行结束后在 `outputs/` 下生成：

- `metrics_summary.json`：三模型汇总结果（JSON）
- `metrics_summary.md`：三模型对比表（Markdown）
- `logisticregression_metrics.json`
- `linearsvm_metrics.json`
- `rnn_lstm_metrics.json`

每条结果字段：

- `model`
- `accuracy`
- `precision`
- `recall`
- `f1`
- `train_seconds`
- `infer_seconds`

## 复现实验建议

在报告提交前，建议执行一次全量实验，然后将 `outputs/metrics_summary.md` 的结果表同步到 `23336128-梁力航-作业4.md` / `23336128-梁力航-作业4.tex`，保证报告与代码一致。
