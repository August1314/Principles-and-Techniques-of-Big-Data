# 大数据原理与技术 — 作业1（图像边缘检测与分类）

## 作业要求（来自 `作业1说明.jpg`）

- 使用 Python + OpenCV 完成：
  - 任意选择一张图像，使用 Sobel 算法检测图像边缘，展示原图与边缘结果。
- 选择一种**机器学习方法**（如 SVM、决策树等）与一种**深度学习方法**（如 MLP、CNN 等）实现图像分类：
  - 数据集自选（如 MNIST、CIFAR-10）。
  - 对比两种方法的**准确率**与**计算效率**（训练/推理耗时）。
- 提交：代码 + 简单报告（2 页以上）
- 压缩包命名格式：`学号+姓名+作业1.zip`
- 邮箱：`BDSysu2026Spring@163.com`

## 环境（Anaconda `ml`）

本作业默认使用你机器上的 conda 环境：`ml`。

### 依赖安装（仅需一次）

在当前目录执行：

```bash
conda run -n ml python -m pip install -r requirements.txt
```

## 目录结构

- `src/edge_sobel.py`：Sobel 边缘检测（保存结果到 `outputs/`）
- `src/train_svm_mnist.py`：机器学习方法：PCA + Linear SVM（MNIST）
- `src/train_cnn_mnist.py`：深度学习方法：简单 CNN（PyTorch，MNIST）
- `src/run_all.py`：一键运行（边缘检测 + 两种分类训练/评估 + 对比表）
- `report.md`：报告模板（已包含实验结果填写位置）

## 运行

### 1) Sobel 边缘检测

```bash
conda run -n ml python src/edge_sobel.py --image data/edge_input.png
```

如果你没有准备图片，可先生成一张示例图：

```bash
conda run -n ml python src/edge_sobel.py --make-demo data/edge_input.png
```

### 2) 训练并评估（MNIST）

```bash
conda run -n ml python src/run_all.py
```

运行完成后会在 `outputs/` 下生成：

- `edge_original.png` / `edge_sobel.png`
- `svm_metrics.json` / `cnn_metrics.json`
- `comparison.md`

## 打包提交

把以下内容放进压缩包：

- `src/`
- `outputs/`
- `report.md`（你补全实验结果与分析）

然后按要求命名：`学号+姓名+作业1.zip`

也可以用脚本自动打包：

```bash
conda run -n ml python src/package_submission.py --student-id 你的学号 --name 你的姓名 --outdir .
```
